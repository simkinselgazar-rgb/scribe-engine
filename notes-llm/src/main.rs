//! `scribe-notes-llm` — the on-device notes-generation sidecar.
//!
//! A standalone llama.cpp process. `scribe-engine` spawns it because
//! whisper.cpp and llama.cpp cannot share one binary — each bundles its
//! own `ggml` and the symbols collide. This crate links only llama.cpp.
//!
//! Protocol — fully local, no network:
//!   stdin  ← a transcript as JSON  (`{"segments":[{speaker,start_secs,end_secs,text}]}`)
//!   argv1  ← path to a Gemma GGUF instruction model
//!   stdout → the notes as JSON     (`{summary,decisions,action_items,billable}`)
//!   exit 1 + stderr on any failure.
//!
//! A short meeting is summarized in one pass. A long one (over an hour,
//! say) is chunked: each ~10-minute span is summarized on its own, with
//! a small overlap for continuity, and the parts are then combined —
//! decisions and action items merged and de-duplicated, the part
//! summaries synthesized into one. This keeps any meeting length within
//! a small, fixed context window, and a focused chunk also yields
//! sharper notes from a small model than one oversized context would.

use std::collections::HashSet;
use std::io::Read;
use std::num::NonZeroU32;
use std::ops::Range;
use std::path::Path;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use serde::{Deserialize, Serialize};

/// The llama.cpp context window.
const CONTEXT_TOKENS: u32 = 8192;
/// llama.cpp's default logical batch size — a single `decode` call may
/// not exceed it, so the prompt is fed in windows of this size.
const PROMPT_BATCH: usize = 2048;
/// Upper bound on generated tokens per pass.
const MAX_OUTPUT_TOKENS: usize = 2048;
/// A transcript whose rendered text is at most this many characters is
/// summarized in a single pass; longer transcripts are chunked.
const SINGLE_PASS_CHARS: usize = 14_000;
/// Target rendered characters of primary content per chunk.
const CHUNK_CHARS: usize = 11_000;
/// Segments from the end of one chunk repeated at the start of the next,
/// so a chunk does not begin mid-thought.
const OVERLAP_SEGMENTS: usize = 2;
/// Hard ceiling on operator-supplied context. A backstop; the app
/// enforces a tighter, model-dependent limit in the UI.
const MAX_CONTEXT_CHARS: usize = 1_000;
/// Byte buffer passed to `token_to_piece_bytes` — a UTF-8 BPE token is
/// at most a handful of bytes; 64 is a conservative upper bound.
const TOKEN_PIECE_BUF: usize = 64;

/// The notes JSON shape, asked for by both the per-chunk and the
/// final reduce passes.
const NOTES_SCHEMA: &str = r#"{
  "summary": "two or three sentences describing the meeting",
  "decisions": [{"text": "a decision that was made", "at_seconds": 0}],
  "action_items": [{"text": "a task someone committed to", "at_seconds": 0}],
  "billable_description": "one short line describing the work, for a time entry"
}"#;

fn main() {
    if let Err(err) = run() {
        eprintln!("scribe-notes-llm: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let model_path = std::env::args()
        .nth(1)
        .ok_or("usage: scribe-notes-llm <model.gguf> (transcript JSON on stdin)")?;

    let mut input = String::new();
    std::io::stdin()
        .read_to_string(&mut input)
        .map_err(|e| format!("could not read transcript from stdin: {e}"))?;
    let transcript: WireTranscript =
        serde_json::from_str(&input).map_err(|e| format!("invalid transcript JSON: {e}"))?;
    if transcript.segments.is_empty() {
        return Err("transcript is empty".into());
    }

    let session = LlamaSession::load(Path::new(&model_path))?;
    let context = clamped_context(&transcript.context);
    let scenario = transcript.scenario;

    // Short meeting: one pass. Long meeting: chunk and combine.
    let notes = if format_segments(&transcript.segments, scenario).len() <= SINGLE_PASS_CHARS {
        let prompt = build_notes_prompt(&transcript.segments, &context, scenario, None);
        run_and_parse(&session, &prompt, &transcript.segments)?
    } else {
        generate_chunked(&session, &transcript.segments, &context, scenario)?
    };

    let json =
        serde_json::to_string(&notes).map_err(|e| format!("could not encode notes: {e}"))?;
    println!("{json}");
    Ok(())
}

/// Run a notes prompt and parse the result, retrying with fresh
/// sampling if the JSON does not parse. A small model's structural
/// slips are stochastic, so a re-roll almost always lands clean.
fn run_and_parse(
    session: &LlamaSession,
    prompt: &str,
    segments: &[WireSegment],
) -> Result<WireNotes, String> {
    let mut last = String::from("no attempt ran");
    for seed in 0..3u32 {
        let raw = session.run(prompt, seed)?;
        match parse_notes(&raw, segments) {
            Ok(notes) => return Ok(notes),
            Err(e) => {
                eprintln!(
                    "scribe-notes-llm: attempt {} produced unparseable notes — {e}",
                    seed + 1
                );
                last = e;
            }
        }
    }
    Err(last)
}

// --- chunked map-reduce ----------------------------------------------

/// Summarize a long transcript chunk by chunk, then combine the parts.
fn generate_chunked(
    session: &LlamaSession,
    segments: &[WireSegment],
    context: &str,
    scenario: Scenario,
) -> Result<WireNotes, String> {
    let ranges = chunk_ranges(segments);
    let total = ranges.len();

    let mut parts: Vec<WireNotes> = Vec::with_capacity(total);
    for (i, range) in ranges.iter().enumerate() {
        let chunk = &segments[range.clone()];
        let prompt = build_notes_prompt(chunk, context, scenario, Some((i + 1, total)));
        // One unparseable chunk (after retries) should not lose the
        // whole meeting — skip it and let the rest carry the notes.
        match run_and_parse(session, &prompt, chunk) {
            Ok(notes) => parts.push(notes),
            Err(e) => eprintln!("scribe-notes-llm: skipping part {} — {e}", i + 1),
        }
    }
    if parts.is_empty() {
        return Err("no part of the meeting could be summarized".into());
    }

    // Reduce: one pass consolidates the part-notes into the final
    // notes. This is where cross-chunk connections are made and the
    // duplicates that overlap and revisited topics create are merged.
    let mut notes = run_and_parse(session, &build_reduce_prompt(&parts, context), segments)?;
    // A mechanical backstop for any exact duplicates the model leaves.
    notes.decisions = dedupe(notes.decisions);
    notes.action_items = dedupe(notes.action_items);
    Ok(notes)
}

/// Split segments into overlapping chunks of roughly [`CHUNK_CHARS`] of
/// rendered content. Each chunk after the first repeats the last
/// [`OVERLAP_SEGMENTS`] segments of the one before it.
fn chunk_ranges(segments: &[WireSegment]) -> Vec<Range<usize>> {
    let mut ranges = Vec::new();
    let mut start = 0usize;
    while start < segments.len() {
        let mut end = start;
        let mut chars = 0usize;
        while end < segments.len() {
            chars += rendered_len(&segments[end]);
            end += 1;
            if chars >= CHUNK_CHARS {
                break;
            }
        }
        ranges.push(start..end);
        if end >= segments.len() {
            break;
        }
        // Next chunk opens with this chunk's overlap tail; always make
        // forward progress.
        start = end.saturating_sub(OVERLAP_SEGMENTS).max(start + 1);
    }
    ranges
}

/// Drop items whose text is a duplicate (case- and whitespace-normalized).
fn dedupe(items: Vec<WireItem>) -> Vec<WireItem> {
    let mut seen = HashSet::new();
    items
        .into_iter()
        .filter(|item| {
            let key = item.text.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase();
            seen.insert(key)
        })
        .collect()
}

// --- llama.cpp -------------------------------------------------------

/// A loaded model. Load once, then [`LlamaSession::run`] as many prompts
/// as the job needs — chunked summarization runs several.
struct LlamaSession {
    backend: LlamaBackend,
    model: LlamaModel,
    threads: i32,
}

impl LlamaSession {
    fn load(model_path: &Path) -> Result<Self, String> {
        let backend =
            LlamaBackend::init().map_err(|e| format!("llama backend init failed: {e}"))?;
        let model_params = LlamaModelParams::default().with_n_gpu_layers(999);
        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
            .map_err(|e| format!("could not load notes model: {e}"))?;
        let threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4);
        Ok(Self { backend, model, threads })
    }

    /// Run one prompt to completion and return the decoded text. `seed`
    /// varies the sampling, so a retry after a parse failure re-rolls.
    fn run(&self, prompt: &str, seed: u32) -> Result<String, String> {
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(CONTEXT_TOKENS))
            .with_n_threads(self.threads)
            .with_n_threads_batch(self.threads);
        let mut ctx = self
            .model
            .new_context(&self.backend, ctx_params)
            .map_err(|e| format!("could not create context: {e}"))?;

        let tokens = self
            .model
            .str_to_token(prompt, AddBos::Always)
            .map_err(|e| format!("could not tokenize prompt: {e}"))?;
        if tokens.len() >= CONTEXT_TOKENS as usize - MAX_OUTPUT_TOKENS {
            return Err("a prompt is too long for the context window".into());
        }

        // Feed the prompt in PROMPT_BATCH-sized windows: llama.cpp
        // asserts that a single decode never exceeds the batch size.
        let mut batch = LlamaBatch::new(PROMPT_BATCH, 1);
        let last = tokens.len() - 1;
        let mut pos: i32 = 0;
        for window in tokens.chunks(PROMPT_BATCH) {
            batch.clear();
            for token in window {
                batch
                    .add(*token, pos, &[0], pos as usize == last)
                    .map_err(|e| format!("could not build batch: {e}"))?;
                pos += 1;
            }
            ctx.decode(&mut batch)
                .map_err(|e| format!("prompt decode failed: {e}"))?;
        }

        // A repeat penalty over a wide window plus a moderate
        // temperature keep structured output coherent and escape the
        // repetition loops pure greedy decoding is prone to. Any JSON
        // the model still slips on is repaired or, failing that,
        // surfaced so the session can be regenerated.
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::penalties(128, 1.15, 0.0, 0.0),
            LlamaSampler::temp(0.4),
            LlamaSampler::top_p(0.95, 1),
            LlamaSampler::dist(seed),
        ]);
        let mut output: Vec<u8> = Vec::new();
        let mut token_pos = pos; // absolute position in the context
        let mut generated = 0usize;

        while generated < MAX_OUTPUT_TOKENS {
            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);
            if self.model.is_eog_token(token) {
                break;
            }
            let bytes = self
                .model
                .token_to_piece_bytes(token, TOKEN_PIECE_BUF, false, None)
                .map_err(|e| format!("could not decode token: {e}"))?;
            output.extend_from_slice(&bytes);

            batch.clear();
            batch
                .add(token, token_pos, &[0], true)
                .map_err(|e| format!("could not extend batch: {e}"))?;
            token_pos += 1;
            generated += 1;
            ctx.decode(&mut batch)
                .map_err(|e| format!("generation decode failed: {e}"))?;
        }

        Ok(String::from_utf8_lossy(&output).into_owned())
    }
}

// --- prompts ---------------------------------------------------------

/// Render the operator's optional context, trimmed and clamped.
fn clamped_context(context: &Option<String>) -> String {
    match context.as_deref().map(str::trim) {
        Some(c) if !c.is_empty() => c.chars().take(MAX_CONTEXT_CHARS).collect(),
        _ => String::new(),
    }
}

/// One rendered transcript line, e.g. `[72s] Me: ...`.
fn rendered_len(segment: &WireSegment) -> usize {
    // `[Ns] Speaker: text\n` — the prefix is small and bounded.
    segment.text.trim().len() + 16
}

/// Render segments as transcript lines. A [`Scenario::VirtualMeeting`]
/// has two channels, so its lines carry a `Me:` / `Other:` speaker
/// label; the single-microphone scenarios have no reliable per-speaker
/// channel, so their lines carry only a timecode.
fn format_segments(segments: &[WireSegment], scenario: Scenario) -> String {
    let mut out = String::new();
    for segment in segments {
        let text = segment.text.trim();
        match scenario {
            Scenario::VirtualMeeting => {
                let speaker = if segment.speaker == "near" { "Me" } else { "Other" };
                out.push_str(&format!("[{}s] {speaker}: {text}\n", segment.start_secs));
            }
            Scenario::SoloMemo | Scenario::InPersonMeeting => {
                out.push_str(&format!("[{}s] {text}\n", segment.start_secs));
            }
        }
    }
    out
}

/// Wrap a system + user message in Gemma's prompt format. Gemma has no
/// system role, so the instructions lead the single user turn.
fn gemma_turn(system: &str, user: &str) -> String {
    format!(
        "<start_of_turn>user\n{system}\n\n{user}<end_of_turn>\n\
         <start_of_turn>model\n"
    )
}

/// How the notes model should read the transcript — one sentence of
/// framing, chosen by the recording scenario.
fn scenario_framing(scenario: Scenario) -> &'static str {
    match scenario {
        Scenario::VirtualMeeting => {
            "This is a transcript of a virtual meeting. \"Me\" is the professional; \
\"Other\" is the client or counterparty."
        }
        Scenario::SoloMemo => {
            "This is a solo voice memo. The professional is speaking alone, with no one \
else present. Treat every line as the professional's own dictation: their notes, \
decisions, and reminders to themselves."
        }
        Scenario::InPersonMeeting => {
            "This transcript was recorded on a single microphone, in a room or on a phone \
call on speaker. More than one person is speaking and the transcript does not identify who \
said what, so do not attribute any statement to a specific person."
        }
    }
}

/// Build the notes prompt for a whole short transcript (`part` is
/// `None`) or one chunk of a long one (`part` is `Some((n, total))`).
fn build_notes_prompt(
    segments: &[WireSegment],
    context: &str,
    scenario: Scenario,
    part: Option<(usize, usize)>,
) -> String {
    let system = "You are a meeting-notes assistant for a busy professional. \
You read a timecoded transcript and return concise, accurate notes. \
Reply with ONLY a single JSON object and nothing else. No prose, no markdown fences.";

    let framing = scenario_framing(scenario);

    let scope = match part {
        Some((n, total)) => format!(
            "This is part {n} of {total} of one longer recording; the parts are in order. \
Write notes covering only this part. Its first lines may repeat the end of the previous \
part for continuity. "
        ),
        None => String::new(),
    };
    let background = if context.is_empty() {
        String::new()
    } else {
        format!("Background the operator gave for this recording: {context}\n\n")
    };

    let user = format!(
        "Write notes for this transcript. {framing} {scope}{background}Record only genuine \
decisions and concrete commitments, not every statement. Each decision and action item must \
set \"at_seconds\" to the [Ns] timecode of the line it came from. Use this exact JSON \
shape:\n{NOTES_SCHEMA}\n\nTranscript:\n{}",
        format_segments(segments, scenario)
    );
    gemma_turn(system, &user)
}

/// Build the prompt that consolidates per-chunk notes into the final
/// notes for the whole meeting.
fn build_reduce_prompt(parts: &[WireNotes], context: &str) -> String {
    let system = "You combine notes from consecutive parts of one meeting into one final set \
of notes. Merge duplicate and near-duplicate items. Reply with ONLY a single JSON object and \
nothing else. No prose, no markdown fences.";
    let background = if context.is_empty() {
        String::new()
    } else {
        format!("Background the operator gave for this meeting: {context}\n\n")
    };

    let user = format!(
        "Below are notes from consecutive parts of one meeting, in order. Produce the final \
combined notes for the whole meeting: one summary of three to five sentences, the key \
decisions, the action items, and a billable-time description. Merge items that repeat across \
parts, and keep an \"at_seconds\" timecode for each. {background}Use this exact JSON shape:\n\
{NOTES_SCHEMA}\n\nPart notes:\n{}",
        format_part_notes(parts)
    );
    gemma_turn(system, &user)
}

/// Render per-chunk notes as the text input to the reduce pass.
fn format_part_notes(parts: &[WireNotes]) -> String {
    let mut out = String::new();
    for (i, part) in parts.iter().enumerate() {
        out.push_str(&format!("Part {}\nSummary: {}\n", i + 1, part.summary.trim()));
        if !part.decisions.is_empty() {
            out.push_str("Decisions:\n");
            for d in &part.decisions {
                out.push_str(&format!("- {} [{}s]\n", d.text, d.source_secs.unwrap_or(0)));
            }
        }
        if !part.action_items.is_empty() {
            out.push_str("Action items:\n");
            for a in &part.action_items {
                out.push_str(&format!("- {} [{}s]\n", a.text, a.source_secs.unwrap_or(0)));
            }
        }
        out.push('\n');
    }
    out
}

// --- parsing the model's response ------------------------------------

/// Parse a notes response into [`WireNotes`], defensively. `segments`
/// supplies the billable duration when the response omits it.
fn parse_notes(raw: &str, segments: &[WireSegment]) -> Result<WireNotes, String> {
    let value = parse_json_object(raw)?;

    let summary = string_field(&value, "summary");
    let decisions = parse_items(value.get("decisions"));
    let action_items = parse_items(value.get("action_items"));
    let description = string_field(&value, "billable_description");

    if summary.is_empty() && decisions.is_empty() && action_items.is_empty() {
        return Err("model did not return usable notes".into());
    }

    let duration_secs = segments.last().map(|s| s.end_secs).unwrap_or_default();
    let billable = (!description.is_empty()).then_some(WireBillable {
        duration_secs,
        description,
    });

    Ok(WireNotes { summary, decisions, action_items, billable })
}

/// Extract and parse the first JSON object from a model response.
fn parse_json_object(raw: &str) -> Result<serde_json::Value, String> {
    let json = extract_json_object(raw).ok_or_else(|| {
        let snippet: String = raw.chars().take(280).collect();
        format!("model response had no JSON object; got: {snippet:?}")
    })?;
    // Small models occasionally leave a trailing comma — strict JSON
    // rejects it, so repair that one common slip before parsing.
    let repaired = strip_trailing_commas(json);
    serde_json::from_str(&repaired).map_err(|e| {
        let snippet: String = repaired.chars().take(400).collect();
        format!("model response was not valid JSON: {e}; got: {snippet:?}")
    })
}

/// Drop trailing commas (`,` before `}` or `]`), the most common way a
/// small model's JSON fails strict parsing. String contents are left
/// untouched.
fn strip_trailing_commas(json: &str) -> String {
    let chars: Vec<char> = json.chars().collect();
    let mut out = String::with_capacity(json.len());
    let mut in_string = false;
    let mut escaped = false;
    for i in 0..chars.len() {
        let c = chars[i];
        if in_string {
            out.push(c);
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        if c == '"' {
            in_string = true;
            out.push(c);
            continue;
        }
        if c == ',' {
            let next = chars[i + 1..]
                .iter()
                .find(|c| !c.is_whitespace());
            if matches!(next, Some('}') | Some(']')) {
                continue; // drop the trailing comma
            }
        }
        out.push(c);
    }
    out
}

fn string_field(value: &serde_json::Value, key: &str) -> String {
    value
        .get(key)
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .trim()
        .to_string()
}

/// Parse a `decisions` / `action_items` array. Tolerates both an array
/// of strings and an array of `{text, at_seconds}` objects.
fn parse_items(value: Option<&serde_json::Value>) -> Vec<WireItem> {
    let Some(array) = value.and_then(|v| v.as_array()) else {
        return Vec::new();
    };
    array
        .iter()
        .filter_map(|entry| {
            if let Some(text) = entry.as_str() {
                return non_empty_item(text, None);
            }
            let text = entry.get("text").and_then(|v| v.as_str()).unwrap_or("");
            non_empty_item(text, parse_seconds(entry.get("at_seconds")))
        })
        .collect()
}

fn non_empty_item(text: &str, source_secs: Option<u64>) -> Option<WireItem> {
    let text = strip_trailing_timecodes(text);
    (!text.is_empty()).then_some(WireItem { text, source_secs })
}

/// Remove trailing `[123s]` timecode tags that the reduce pass can echo
/// into an item's text from its input formatting.
fn strip_trailing_timecodes(text: &str) -> String {
    let mut t = text.trim();
    while t.ends_with(']') {
        let Some(open) = t.rfind('[') else {
            break;
        };
        let inner = &t[open + 1..t.len() - 1];
        let is_timecode = inner.len() >= 2
            && inner.ends_with('s')
            && inner[..inner.len() - 1].chars().all(|c| c.is_ascii_digit());
        if !is_timecode {
            break;
        }
        t = t[..open].trim_end_matches([' ', '/', ',']).trim_end();
    }
    t.to_string()
}

/// Read `at_seconds` whether the model emitted a number or a string.
fn parse_seconds(value: Option<&serde_json::Value>) -> Option<u64> {
    let value = value?;
    value
        .as_u64()
        .or_else(|| value.as_f64().map(|f| f as u64))
        .or_else(|| value.as_str().and_then(|s| s.trim().parse().ok()))
}

/// Extract the first balanced `{...}` block, ignoring braces in strings.
fn extract_json_object(text: &str) -> Option<&str> {
    let bytes = text.as_bytes();
    let start = text.find('{')?;
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for i in start..bytes.len() {
        let c = bytes[i];
        if in_string {
            if escaped {
                escaped = false;
            } else if c == b'\\' {
                escaped = true;
            } else if c == b'"' {
                in_string = false;
            }
            continue;
        }
        match c {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&text[start..=i]);
                }
            }
            _ => {}
        }
    }
    None
}

// --- the JSON protocol shared with scribe-engine ---------------------

/// How to read the transcript — mirrors `scribe-engine`'s
/// `RecordingScenario`, sent over the wire as a snake_case tag. Defaults
/// to a virtual meeting if the field is absent.
#[derive(Deserialize, Clone, Copy, Default)]
#[serde(rename_all = "snake_case")]
enum Scenario {
    SoloMemo,
    #[default]
    VirtualMeeting,
    InPersonMeeting,
}

#[derive(Deserialize)]
struct WireTranscript {
    #[serde(default)]
    scenario: Scenario,
    #[serde(default)]
    context: Option<String>,
    segments: Vec<WireSegment>,
}

#[derive(Deserialize)]
struct WireSegment {
    speaker: String,
    start_secs: u64,
    end_secs: u64,
    text: String,
}

#[derive(Serialize)]
struct WireNotes {
    summary: String,
    decisions: Vec<WireItem>,
    action_items: Vec<WireItem>,
    billable: Option<WireBillable>,
}

#[derive(Serialize, Clone)]
struct WireItem {
    text: String,
    source_secs: Option<u64>,
}

#[derive(Serialize)]
struct WireBillable {
    duration_secs: u64,
    description: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn segment(at: u64, text: &str) -> WireSegment {
        WireSegment {
            speaker: "near".into(),
            start_secs: at,
            end_secs: at + 10,
            text: text.into(),
        }
    }

    #[test]
    fn extracts_json_amid_prose_and_fences() {
        let raw = "Sure! ```json\n{\"summary\": \"a {nested} brace\"}\n``` done";
        assert_eq!(
            extract_json_object(raw).unwrap(),
            "{\"summary\": \"a {nested} brace\"}"
        );
    }

    #[test]
    fn parses_object_and_string_items() {
        let objects = serde_json::json!([
            {"text": "Close by June 30", "at_seconds": 72},
            {"text": "  ", "at_seconds": 5},
            {"text": "Send the estimate", "at_seconds": "570"}
        ]);
        let items = parse_items(Some(&objects));
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].source_secs, Some(72));
        assert_eq!(items[1].source_secs, Some(570));

        let strings = serde_json::json!(["First", "Second"]);
        let items = parse_items(Some(&strings));
        assert_eq!(items.len(), 2);
        assert!(items[0].source_secs.is_none());
    }

    #[test]
    fn long_transcript_splits_into_overlapping_chunks() {
        // ~300 segments of ~100 chars each — well past one chunk.
        let line = "x".repeat(100);
        let segments: Vec<WireSegment> =
            (0..300).map(|i| segment(i as u64 * 10, &line)).collect();

        let ranges = chunk_ranges(&segments);
        assert!(ranges.len() > 1, "a long transcript should chunk");
        // Consecutive chunks overlap, and every segment is covered.
        assert_eq!(ranges[0].start, 0);
        assert_eq!(ranges.last().unwrap().end, segments.len());
        for pair in ranges.windows(2) {
            assert!(pair[1].start < pair[0].end, "chunks should overlap");
            assert!(pair[1].start > pair[0].start, "chunks must make progress");
        }
    }

    #[test]
    fn short_transcript_is_a_single_chunk() {
        let segments = vec![segment(0, "hello"), segment(10, "goodbye")];
        let ranges = chunk_ranges(&segments);
        assert_eq!(ranges, vec![0..2]);
    }

    #[test]
    fn strips_trailing_timecode_tags() {
        assert_eq!(
            strip_trailing_timecodes("Prepare the filing now. [2577s]"),
            "Prepare the filing now."
        );
        assert_eq!(
            strip_trailing_timecodes("Flag nexus issues. [2241s] / [3321s]"),
            "Flag nexus issues."
        );
        assert_eq!(strip_trailing_timecodes("No tag here."), "No tag here.");
        assert_eq!(
            strip_trailing_timecodes("Keep [bracketed] words"),
            "Keep [bracketed] words"
        );
    }

    #[test]
    fn strips_trailing_commas_outside_strings() {
        assert_eq!(strip_trailing_commas(r#"{"a":1,}"#), r#"{"a":1}"#);
        assert_eq!(strip_trailing_commas(r#"{"a":[1,2, ],}"#), r#"{"a":[1,2 ]}"#);
        // A comma inside a string value is left alone.
        assert_eq!(strip_trailing_commas(r#"{"a":"x, y"}"#), r#"{"a":"x, y"}"#);
    }

    #[test]
    fn dedupe_drops_normalized_duplicates() {
        let items = vec![
            WireItem { text: "Send the report".into(), source_secs: Some(1) },
            WireItem { text: "send  the   report".into(), source_secs: Some(2) },
            WireItem { text: "File the motion".into(), source_secs: Some(3) },
        ];
        assert_eq!(dedupe(items).len(), 2);
    }
}
