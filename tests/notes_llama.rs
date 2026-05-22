//! Integration test for the notes backend (the `scribe-notes-llm`
//! sidecar driving llama.cpp).
//!
//! Requires two things, both git-ignored:
//!   - a Gemma GGUF model at `target/test-assets/gemma-4-E4B-it-Q4_K_M.gguf`
//!   - the built sidecar binary (run `cargo build` first)
//! When either is absent the test skips so `cargo test` stays green.

use std::path::PathBuf;
use std::time::Duration;

use scribe_engine::{
    NotesGenerator, RecordingScenario, SidecarNotesGenerator, Speaker, Transcript,
    TranscriptSegment,
};

fn asset(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/test-assets")
        .join(name)
}

/// Locate the built `scribe-notes-llm` sidecar binary, if any.
fn sidecar() -> Option<PathBuf> {
    let target = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target");
    ["release", "debug"]
        .into_iter()
        .map(|profile| target.join(profile).join("scribe-notes-llm"))
        .find(|p| p.exists())
}

fn seg(speaker: Speaker, at: u64, text: &str) -> TranscriptSegment {
    TranscriptSegment {
        speaker,
        start: Duration::from_secs(at),
        end: Duration::from_secs(at + 15),
        text: text.to_string(),
    }
}

/// A realistic professional-services call — a corporate-acquisition
/// kickoff, the same scenario the app's UI was designed against.
fn acme_transcript() -> Transcript {
    use Speaker::{Far, Near};
    Transcript {
        segments: vec![
            seg(Near, 8, "Thanks for making the time. I want to use this first call to scope the engagement."),
            seg(Far, 31, "We are acquiring a smaller competitor and we need the deal papered and closed before the end of the quarter."),
            seg(Near, 72, "Understood. End of quarter is June 30. That is tight but workable if we get the diligence materials quickly."),
            seg(Far, 125, "We can have the data room open to you by Friday — contracts, the cap table, and two years of financials."),
            seg(Near, 220, "Good. I will assign a senior associate to lead diligence and send a document request list by Wednesday."),
            seg(Far, 318, "A couple of customer contracts have change-of-control clauses, and we do not want to spook those customers."),
            seg(Near, 362, "We will review the change-of-control provisions first and build a consent strategy before anyone is contacted."),
            seg(Far, 527, "On fees, we would like a not-to-exceed estimate. The board will ask for one."),
            seg(Near, 570, "I will send a fee estimate with a not-to-exceed figure by early next week, broken out by phase."),
            seg(Far, 862, "The founder of the target wants a two-year consulting agreement as part of the deal."),
            seg(Near, 905, "We will draft the consulting agreement alongside the purchase agreement and tie part of it to a non-compete."),
            seg(Far, 1360, "Who on your side is the day-to-day contact?"),
            seg(Near, 1395, "Sarah Okafor, our senior associate, will run the day-to-day."),
        ],
    }
}

#[test]
fn generates_structured_notes_from_a_transcript() {
    let model = asset("gemma-4-E4B-it-Q4_K_M.gguf");
    let Some(sidecar) = sidecar() else {
        eprintln!("skipping: scribe-notes-llm not built — run `cargo build` first");
        return;
    };
    if !model.exists() {
        eprintln!("skipping: notes model not in target/test-assets");
        return;
    }

    let generator = SidecarNotesGenerator::new(sidecar, model);
    let notes = generator
        .generate(
            &acme_transcript(),
            Some("Kickoff call for the Acme Corp acquisition; focus on the closing timeline."),
            RecordingScenario::VirtualMeeting,
        )
        .expect("generate notes");

    // Eyeball output in `cargo test -- --nocapture`.
    println!("\nSUMMARY: {}", notes.summary);
    for d in &notes.decisions {
        println!("DECISION ({:?}): {}", d.source, d.text);
    }
    for a in &notes.action_items {
        println!("ACTION   ({:?}): {}", a.source, a.text);
    }
    println!("BILLABLE: {:?}\n", notes.billable);

    assert!(
        notes.summary.len() > 20,
        "summary is implausibly short: {:?}",
        notes.summary
    );
    assert!(
        !notes.decisions.is_empty() || !notes.action_items.is_empty(),
        "model produced neither decisions nor action items"
    );

    let billable = notes.billable.expect("a billable draft");
    assert!(!billable.description.is_empty(), "empty billable description");
    // Duration is taken from the transcript, never the model.
    assert_eq!(billable.duration, Duration::from_secs(1395 + 15));

    // Any timecode a note cites must fall within the recording.
    for item in notes.decisions.iter().chain(&notes.action_items) {
        if let Some(source) = item.source {
            assert!(
                source <= Duration::from_secs(1395 + 15),
                "note cites a timecode past the end of the recording"
            );
        }
    }
}
