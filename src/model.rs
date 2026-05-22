//! Shared data types: the [`Transcript`] produced by transcription and
//! the [`Notes`] produced from it.

use std::time::Duration;

/// Which side of the conversation a segment came from. The engine keeps
/// the near end (microphone) and the far end (system audio) as separate
/// streams, so every transcript segment is attributed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Speaker {
    /// The operator — captured from the microphone.
    Near,
    /// The other participants — captured from system audio.
    Far,
}

/// One attributed, timecoded span of transcript.
#[derive(Debug, Clone)]
pub struct TranscriptSegment {
    pub speaker: Speaker,
    /// Offset from the start of the recording.
    pub start: Duration,
    pub end: Duration,
    pub text: String,
}

/// A full speaker-attributed, timecoded transcript of one session.
#[derive(Debug, Clone, Default)]
pub struct Transcript {
    pub segments: Vec<TranscriptSegment>,
}

/// One note line. `source` ties it back to the transcript moment it was
/// drawn from — the note-to-transcript trust mechanism from the design
/// brief.
#[derive(Debug, Clone)]
pub struct NoteItem {
    pub text: String,
    /// Offset into the recording this item traces to, when known.
    pub source: Option<Duration>,
}

/// A draft time entry the operator confirms — never asserted as fact.
#[derive(Debug, Clone)]
pub struct BillableDraft {
    pub duration: Duration,
    pub description: String,
}

/// The structured notes generated from a [`Transcript`] — the v0.1
/// fixed shape (see the design brief): summary, key decisions, action
/// items, and a billable-time draft.
#[derive(Debug, Clone, Default)]
pub struct Notes {
    pub summary: String,
    pub decisions: Vec<NoteItem>,
    pub action_items: Vec<NoteItem>,
    pub billable: Option<BillableDraft>,
}
