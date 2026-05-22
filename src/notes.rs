//! Note generation — a [`Transcript`] to structured [`Notes`].
//!
//! [`NotesGenerator`] is the pluggable backend seam. The v0.1 default
//! backend is a local LLM via llama.cpp, so the transcript never leaves
//! the machine. No backend ships in this scaffold.

use crate::{Notes, Result, Transcript};

/// A note-generation backend.
pub trait NotesGenerator: Send {
    /// Synthesize the v0.1 fixed-shape notes from a transcript.
    fn generate(&self, transcript: &Transcript) -> Result<Notes>;
}
