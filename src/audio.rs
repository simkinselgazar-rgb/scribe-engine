//! Audio decoding for the transcription backend.
//!
//! A *recording* is one stereo WAV file: channel 0 (left) is the near
//! end — the operator's microphone — and channel 1 (right) is the far
//! end — system audio, the other participants. Keeping the two ends on
//! separate channels is what lets the transcriber attribute every
//! segment to a speaker without a diarization model.
//!
//! Whisper wants 16 kHz mono `f32`, so this module decodes, splits, and
//! resamples in one step. All of it is local file I/O.

use std::path::Path;

use crate::{EngineError, Result};

/// Whisper's required input sample rate.
pub const WHISPER_RATE: u32 = 16_000;

/// A decoded recording, split into its two ends. Each channel is 16 kHz
/// mono `f32` in `-1.0..=1.0`. `far` is empty when the source was mono.
pub struct Channels {
    /// The operator's microphone — the near end.
    pub near: Vec<f32>,
    /// System audio — the far end. Empty for a mono recording.
    pub far: Vec<f32>,
}

/// Decode a recording into 16 kHz mono channels ready for Whisper.
pub fn decode_recording(path: &Path) -> Result<Channels> {
    let mut reader = hound::WavReader::open(path)
        .map_err(|e| EngineError::Transcribe(format!("could not open recording: {e}")))?;
    let spec = reader.spec();

    // Read every sample as normalized f32, interleaved by channel.
    let interleaved: Vec<f32> = match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .collect::<std::result::Result<_, _>>()
            .map_err(decode_err)?,
        (hound::SampleFormat::Int, bits) => {
            let scale = 1.0_f32 / (1_i64 << (bits - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 * scale))
                .collect::<std::result::Result<_, _>>()
                .map_err(decode_err)?
        }
        (fmt, bits) => {
            return Err(EngineError::Transcribe(format!(
                "unsupported WAV format: {fmt:?} {bits}-bit"
            )));
        }
    };

    let channels = spec.channels.max(1) as usize;
    let (near_raw, far_raw) = deinterleave(&interleaved, channels);

    Ok(Channels {
        near: resample_to_whisper(&near_raw, spec.sample_rate),
        far: resample_to_whisper(&far_raw, spec.sample_rate),
    })
}

fn decode_err(e: hound::Error) -> EngineError {
    EngineError::Transcribe(format!("could not decode recording: {e}"))
}

/// Split interleaved samples into the near (channel 0) and far
/// (channel 1) ends. A mono source yields an empty far channel.
fn deinterleave(interleaved: &[f32], channels: usize) -> (Vec<f32>, Vec<f32>) {
    if channels <= 1 {
        return (interleaved.to_vec(), Vec::new());
    }
    let frames = interleaved.len() / channels;
    let mut near = Vec::with_capacity(frames);
    let mut far = Vec::with_capacity(frames);
    for frame in interleaved.chunks_exact(channels) {
        near.push(frame[0]);
        far.push(frame[1]);
    }
    (near, far)
}

/// Resample to 16 kHz with box-filter decimation — each output sample
/// is the mean of the input window it spans. Aliasing-safe enough for
/// speech, and exact for integer ratios like 48 kHz → 16 kHz.
///
/// Shared with the recorder, which resamples both captured channels to
/// 16 kHz before writing the stereo WAV.
pub(crate) fn resample_to_whisper(input: &[f32], in_rate: u32) -> Vec<f32> {
    if input.is_empty() || in_rate == WHISPER_RATE {
        return input.to_vec();
    }
    let ratio = in_rate as f64 / WHISPER_RATE as f64;
    let out_len = ((input.len() as f64) / ratio).floor() as usize;
    let mut out = Vec::with_capacity(out_len);
    for j in 0..out_len {
        let start = (j as f64 * ratio) as usize;
        let end = (((j + 1) as f64 * ratio).ceil() as usize)
            .max(start + 1)
            .min(input.len());
        let window = &input[start..end];
        out.push(window.iter().sum::<f32>() / window.len() as f32);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deinterleave_splits_stereo() {
        let (near, far) = deinterleave(&[1.0, 2.0, 3.0, 4.0], 2);
        assert_eq!(near, [1.0, 3.0]);
        assert_eq!(far, [2.0, 4.0]);
    }

    #[test]
    fn deinterleave_mono_has_empty_far() {
        let (near, far) = deinterleave(&[1.0, 2.0, 3.0], 1);
        assert_eq!(near, [1.0, 2.0, 3.0]);
        assert!(far.is_empty());
    }

    #[test]
    fn resample_48k_to_16k_thirds_the_length() {
        let input = vec![0.5_f32; 4800];
        let out = resample_to_whisper(&input, 48_000);
        assert_eq!(out.len(), 1600);
        assert!((out[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn resample_passthrough_at_native_rate() {
        let input = vec![0.1, 0.2, 0.3];
        assert_eq!(resample_to_whisper(&input, WHISPER_RATE), input);
    }
}
