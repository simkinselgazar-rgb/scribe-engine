//! Transcription — captured audio to a speaker-attributed [`Transcript`].
//!
//! [`Transcriber`] is the pluggable backend seam. [`WhisperTranscriber`]
//! is the v0.1 default: whisper.cpp via `whisper-rs`, running entirely
//! on-device. The model file is supplied by the host app — this module
//! never fetches it.
//!
//! Speaker attribution is structural, not statistical: a recording is a
//! stereo WAV with the near end (microphone) on one channel and the far
//! end (system audio) on the other, so each channel is transcribed
//! independently and every segment is attributed by which channel it
//! came from. See [`crate::audio`].

use std::path::Path;
use std::time::Duration;

use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

use crate::audio::{self, Channels};
use crate::model::{Speaker, TranscriptSegment};
use crate::{EngineError, Result, Transcript};

/// A transcription backend.
pub trait Transcriber: Send {
    /// Transcribe a finished recording into a timecoded transcript.
    fn transcribe(&self, recording: &Path) -> Result<Transcript>;
}

/// The whisper.cpp transcription backend. Load the model once; each
/// [`Transcriber::transcribe`] call spins up its own decoder state, so
/// one transcriber can be reused across sessions.
pub struct WhisperTranscriber {
    ctx: WhisperContext,
    threads: i32,
}

impl WhisperTranscriber {
    /// Load a GGML Whisper model from disk (e.g. `ggml-base.en.bin`).
    /// The host app is responsible for getting the file there; this
    /// constructor only reads it.
    pub fn new(model_path: &Path) -> Result<Self> {
        let model = model_path
            .to_str()
            .ok_or_else(|| EngineError::Transcribe("model path is not valid UTF-8".into()))?;
        let ctx = WhisperContext::new_with_params(model, WhisperContextParameters::default())
            .map_err(|e| EngineError::Transcribe(format!("could not load model: {e}")))?;
        let threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4);
        Ok(Self { ctx, threads })
    }

    /// Transcribe one 16 kHz mono channel and attribute every segment
    /// to `speaker`.
    fn transcribe_channel(
        &self,
        samples: &[f32],
        speaker: Speaker,
    ) -> Result<Vec<TranscriptSegment>> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }
        let mut state = self
            .ctx
            .create_state()
            .map_err(|e| EngineError::Transcribe(format!("could not create decoder: {e}")))?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_n_threads(self.threads);
        params.set_language(Some("en"));
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        state
            .full(params, samples)
            .map_err(|e| EngineError::Transcribe(format!("transcription failed: {e}")))?;

        let segment_count = state
            .full_n_segments()
            .map_err(|e| EngineError::Transcribe(format!("could not read segments: {e}")))?;

        let mut segments = Vec::new();
        for i in 0..segment_count {
            let text = state
                .full_get_segment_text(i)
                .map_err(|e| EngineError::Transcribe(format!("could not read segment text: {e}")))?;
            let text = text.trim();
            if text.is_empty() {
                continue;
            }
            let t0 = state
                .full_get_segment_t0(i)
                .map_err(|e| EngineError::Transcribe(format!("could not read timestamp: {e}")))?;
            let t1 = state
                .full_get_segment_t1(i)
                .map_err(|e| EngineError::Transcribe(format!("could not read timestamp: {e}")))?;
            segments.push(TranscriptSegment {
                speaker,
                start: centiseconds(t0),
                end: centiseconds(t1),
                text: text.to_string(),
            });
        }
        Ok(segments)
    }
}

impl Transcriber for WhisperTranscriber {
    fn transcribe(&self, recording: &Path) -> Result<Transcript> {
        let Channels { near, far } = audio::decode_recording(recording)?;
        // The two ends are transcribed independently, then interleaved
        // back into one timeline ordered by start time.
        let mut segments = self.transcribe_channel(&near, Speaker::Near)?;
        segments.extend(self.transcribe_channel(&far, Speaker::Far)?);
        segments.sort_by_key(|s| s.start);
        Ok(Transcript { segments })
    }
}

/// whisper.cpp reports timestamps in centiseconds (100 = one second).
fn centiseconds(value: i64) -> Duration {
    Duration::from_millis(value.max(0) as u64 * 10)
}
