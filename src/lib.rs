//! # scribe-engine
//!
//! The open-source core of On-Device Scribe — everything that turns a
//! confidential conversation into structured notes, entirely on the
//! operator's machine.
//!
//! **Invariant:** nothing in this crate performs network I/O. Not for
//! telemetry, not for fetching models, not for anything. Audio,
//! transcripts, and notes never leave the machine. A change that adds a
//! network call breaks the product.
//!
//! Three subsystems, each a trait with pluggable backends:
//!
//! - [`capture`] — record system audio (the far end) and the microphone
//!   (the near end) as two streams.
//! - [`transcribe`] — turn captured audio into a speaker-attributed,
//!   timecoded [`Transcript`]. Default backend: whisper.cpp.
//! - [`notes`] — turn a [`Transcript`] into structured [`Notes`].
//!   Default backend: a local LLM via llama.cpp.
//!
//! The host application (the Tauri desktop app) composes the three.

pub mod capture;
pub mod model;
pub mod notes;
pub mod transcribe;

pub use model::{BillableDraft, Notes, NoteItem, Speaker, Transcript, TranscriptSegment};

/// Errors surfaced by the engine. Every variant is local — there is no
/// network failure mode, because the engine never touches a network.
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("audio capture failed: {0}")]
    Capture(String),
    #[error("transcription failed: {0}")]
    Transcribe(String),
    #[error("note generation failed: {0}")]
    Notes(String),
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),
}

/// The engine's result type.
pub type Result<T> = std::result::Result<T, EngineError>;
