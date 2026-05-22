//! Transcription — captured audio to a speaker-attributed [`Transcript`].
//!
//! [`Transcriber`] is the pluggable backend seam. The v0.1 default
//! backend is whisper.cpp (Metal on Apple Silicon, CPU/CUDA on Windows);
//! mlx-whisper may be added later as an optional Apple-Silicon
//! accelerator. No backend ships in this scaffold.

use std::path::Path;

use crate::{Result, Transcript};

/// A transcription backend.
pub trait Transcriber: Send {
    /// Transcribe a finished recording into a timecoded transcript.
    fn transcribe(&self, recording: &Path) -> Result<Transcript>;
}
