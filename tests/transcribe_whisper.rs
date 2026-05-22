//! Integration test for the whisper.cpp transcription backend.
//!
//! Requires two assets in `target/test-assets/` (git-ignored): a GGML
//! Whisper model `ggml-base.en.bin` and a stereo sample `jfk-stereo.wav`
//! (the whisper.cpp JFK clip on the left channel, the same clip delayed
//! 3 s on the right). When the assets are absent the test skips so
//! `cargo test` stays green on machines without them.

use std::path::PathBuf;
use std::time::Duration;

use scribe_engine::{RecordingScenario, Speaker, Transcriber, WhisperTranscriber};

fn asset(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/test-assets")
        .join(name)
}

#[test]
fn transcribes_stereo_recording_with_speaker_attribution() {
    let model = asset("ggml-base.en.bin");
    let recording = asset("jfk-stereo.wav");
    if !model.exists() || !recording.exists() {
        eprintln!("skipping: whisper model / sample not in target/test-assets");
        return;
    }

    let transcriber = WhisperTranscriber::new(&model).expect("load whisper model");
    // A two-channel sample — the virtual-meeting scenario, so both ends
    // are read.
    let transcript = transcriber
        .transcribe(&recording, RecordingScenario::VirtualMeeting)
        .expect("transcribe recording");

    assert!(!transcript.segments.is_empty(), "no segments produced");

    // Both channels carry the clip, so both speakers must be attributed.
    let has_near = transcript.segments.iter().any(|s| s.speaker == Speaker::Near);
    let has_far = transcript.segments.iter().any(|s| s.speaker == Speaker::Far);
    assert!(has_near, "expected near-channel (microphone) segments");
    assert!(has_far, "expected far-channel (system audio) segments");

    // The clip's content should come through.
    let joined: String = transcript
        .segments
        .iter()
        .map(|s| s.text.to_lowercase())
        .collect::<Vec<_>>()
        .join(" ");
    assert!(
        joined.contains("country"),
        "transcript missing expected words: {joined}"
    );

    // The merged timeline is ordered by start time.
    let mut prev = Duration::ZERO;
    for segment in &transcript.segments {
        assert!(segment.start >= prev, "segments are not ordered by start time");
        assert!(segment.end >= segment.start, "segment ends before it starts");
        prev = segment.start;
    }
}
