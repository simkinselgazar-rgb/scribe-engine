//! Note generation — a [`Transcript`] to structured [`Notes`].
//!
//! [`NotesGenerator`] is the pluggable backend seam. [`SidecarNotesGenerator`]
//! is the v0.1 default: it runs the `scribe-notes-llm` sidecar — a local
//! llama.cpp process — and never touches the network.
//!
//! ## Why a sidecar
//!
//! whisper.cpp (the transcription backend) and llama.cpp (the notes
//! backend) each statically bundle their own copy of `ggml`. Linked into
//! one binary, the duplicate symbols collide and corrupt inference. So
//! the notes LLM is a separate process: this crate speaks to it over a
//! small JSON protocol on stdin/stdout. The sidecar is fully on-device —
//! see `notes-llm/`.

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::model::{BillableDraft, NoteItem, Notes, RecordingScenario, Speaker};
use crate::{EngineError, Result, Transcript};

/// A note-generation backend.
pub trait NotesGenerator: Send {
    /// Synthesize the v0.1 fixed-shape notes from a transcript.
    ///
    /// `context` is an optional operator-supplied note about the meeting
    /// (parties, matter, what to focus on) — it grounds the summary.
    /// `scenario` tells the model how to read the transcript: one
    /// speaker, two sides, or an unseparated room.
    fn generate(
        &self,
        transcript: &Transcript,
        context: Option<&str>,
        scenario: RecordingScenario,
    ) -> Result<Notes>;
}

/// Runs note generation in the `scribe-notes-llm` sidecar process.
///
/// Construct it with the path to the sidecar binary (the host app
/// bundles it) and the path to the GGUF notes model. Both are read
/// fresh on every [`NotesGenerator::generate`] call, so one generator
/// can be reused across sessions.
pub struct SidecarNotesGenerator {
    sidecar: PathBuf,
    model: PathBuf,
}

impl SidecarNotesGenerator {
    /// `sidecar` is the `scribe-notes-llm` executable; `model` is a
    /// Gemma GGUF instruction model (the v0.1 default is Gemma 4 E4B).
    pub fn new(sidecar: PathBuf, model: PathBuf) -> Self {
        Self { sidecar, model }
    }
}

impl NotesGenerator for SidecarNotesGenerator {
    fn generate(
        &self,
        transcript: &Transcript,
        context: Option<&str>,
        scenario: RecordingScenario,
    ) -> Result<Notes> {
        if transcript.segments.is_empty() {
            return Err(EngineError::Notes("transcript is empty".into()));
        }

        let request = serde_json::to_vec(&WireTranscript::new(transcript, context, scenario))
            .map_err(|e| EngineError::Notes(format!("could not encode transcript: {e}")))?;

        let mut child = Command::new(&self.sidecar)
            .arg(&self.model)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|e| {
                EngineError::Notes(format!(
                    "could not start notes sidecar {}: {e}",
                    self.sidecar.display()
                ))
            })?;

        // The transcript is small and the sidecar reads stdin to EOF
        // before generating, so a single blocking write cannot deadlock.
        child
            .stdin
            .take()
            .ok_or_else(|| EngineError::Notes("notes sidecar stdin unavailable".into()))?
            .write_all(&request)
            .map_err(|e| EngineError::Notes(format!("could not send transcript: {e}")))?;

        let output = child
            .wait_with_output()
            .map_err(|e| EngineError::Notes(format!("notes sidecar failed: {e}")))?;
        if !output.status.success() {
            return Err(EngineError::Notes(format!(
                "notes sidecar exited with {}",
                output.status
            )));
        }

        let wire: WireNotes = serde_json::from_slice(&output.stdout).map_err(|e| {
            EngineError::Notes(format!("could not parse notes sidecar output: {e}"))
        })?;
        Ok(wire.into())
    }
}

// --- the JSON protocol shared with the sidecar -----------------------
//
// Durations cross the boundary as plain whole seconds — unambiguous and
// readable. The sidecar mirrors these shapes.

#[derive(Serialize)]
struct WireTranscript {
    /// The recording scenario, as a snake_case tag the sidecar reads.
    scenario: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    context: Option<String>,
    segments: Vec<WireSegment>,
}

#[derive(Serialize)]
struct WireSegment {
    speaker: &'static str,
    start_secs: u64,
    end_secs: u64,
    text: String,
}

impl WireTranscript {
    fn new(
        transcript: &Transcript,
        context: Option<&str>,
        scenario: RecordingScenario,
    ) -> Self {
        let segments = transcript
            .segments
            .iter()
            .map(|s| WireSegment {
                speaker: match s.speaker {
                    Speaker::Near => "near",
                    Speaker::Far => "far",
                },
                start_secs: s.start.as_secs(),
                end_secs: s.end.as_secs(),
                text: s.text.clone(),
            })
            .collect();
        Self {
            scenario: match scenario {
                RecordingScenario::SoloMemo => "solo_memo",
                RecordingScenario::VirtualMeeting => "virtual_meeting",
                RecordingScenario::InPersonMeeting => "in_person_meeting",
            },
            context: context
                .map(str::trim)
                .filter(|c| !c.is_empty())
                .map(str::to_string),
            segments,
        }
    }
}

#[derive(Deserialize)]
struct WireNotes {
    summary: String,
    decisions: Vec<WireItem>,
    action_items: Vec<WireItem>,
    billable: Option<WireBillable>,
}

#[derive(Deserialize)]
struct WireItem {
    text: String,
    source_secs: Option<u64>,
}

#[derive(Deserialize)]
struct WireBillable {
    duration_secs: u64,
    description: String,
}

impl From<WireNotes> for Notes {
    fn from(wire: WireNotes) -> Self {
        let item = |i: WireItem| NoteItem {
            text: i.text,
            source: i.source_secs.map(Duration::from_secs),
        };
        Notes {
            summary: wire.summary,
            decisions: wire.decisions.into_iter().map(item).collect(),
            action_items: wire.action_items.into_iter().map(item).collect(),
            billable: wire.billable.map(|b| BillableDraft {
                duration: Duration::from_secs(b.duration_secs),
                description: b.description,
            }),
        }
    }
}
