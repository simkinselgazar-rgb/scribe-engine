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

/// What kind of recording a session is — declared by the operator
/// before recording starts. It decides which audio sources are
/// captured, how the transcript is attributed, and how the notes model
/// is told to read the conversation. Channel-based speaker attribution
/// is only meaningful for a [`VirtualMeeting`](Self::VirtualMeeting),
/// where the far side plays through the computer; the other two
/// scenarios are single-microphone.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RecordingScenario {
    /// Just the operator — a dictated memo or note to self. One speaker;
    /// the microphone is the only source.
    SoloMemo,
    /// A virtual meeting held through the computer (Zoom, Teams, Meet).
    /// The operator is on the microphone and the other party (or
    /// parties) play through the computer's audio, captured as system
    /// audio. The two sides land on separate channels and can be
    /// attributed.
    #[default]
    VirtualMeeting,
    /// People sharing the one microphone — an in-person meeting, or a
    /// phone call on speaker beside the computer. Everyone is recorded
    /// acoustically through the microphone and cannot be separated by
    /// channel.
    InPersonMeeting,
}

impl RecordingScenario {
    /// Whether this scenario captures the far end (system audio). Only a
    /// virtual meeting has a far end on the computer; a solo memo and an
    /// in-person recording are microphone-only, so their far channel is
    /// silent.
    pub fn captures_system_audio(self) -> bool {
        matches!(self, Self::VirtualMeeting)
    }
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
