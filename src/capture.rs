//! Audio capture — the far end (system audio) and the near end
//! (microphone) recorded as two streams, finalized into one stereo WAV.
//!
//! [`AudioCapture`] is the platform seam. [`Recorder`] is the v0.1
//! backend: the microphone via `cpal` (the near end / left channel) and
//! macOS system audio via ScreenCaptureKit (the far end / right
//! channel). Keeping the two ends on separate channels is what lets the
//! transcriber attribute speakers without a diarization model — see
//! [`crate::audio`] and [`crate::transcribe`].
//!
//! Each OS audio stream is owned by its own thread, because neither the
//! `cpal` nor the ScreenCaptureKit stream object is `Send`. [`Recorder`]
//! holds only thread handles and shared sample buffers, so it stays
//! `Send` for the trait. All capture is on-device; nothing is sent
//! anywhere.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use crate::audio::resample_to_whisper;
use crate::{EngineError, Result};

/// Default far-end sample rate. ScreenCaptureKit is configured to
/// deliver system audio at this rate on macOS; WASAPI loopback on
/// Windows delivers at whatever the output device's native rate is and
/// reports the real value via [`Recorder::far_rate`].
const SYSTEM_RATE: u32 = 48_000;

/// Window, in samples, over which [`Recorder::levels`] averages.
const LEVEL_WINDOW: usize = 4_800;

/// How often buffered audio is spilled to disk while recording, so a
/// long meeting never holds the whole recording in memory.
const FLUSH_INTERVAL: Duration = Duration::from_secs(3);

/// Gain applied to the RMS level so that a typical speech signal reaches
/// near `1.0` on the capture-panel meters. Speech RMS sits well below
/// full scale; the multiplier compensates so the meters feel responsive.
const LEVEL_GAIN: f32 = 4.0;

/// A live capture session. Backends are platform-specific; the host app
/// holds a `Box<dyn AudioCapture>` and does not know which one it has.
pub trait AudioCapture: Send {
    /// Begin capturing both streams.
    fn start(&mut self) -> Result<()>;

    /// Stop capturing and finalize the recording to its WAV path.
    fn stop(&mut self) -> Result<()>;

    /// Current near-end / far-end input levels, for the capture panel's
    /// level meters.
    fn levels(&self) -> CaptureLevels;
}

/// Instantaneous input levels (each `0.0..=1.0`) for the live capture
/// panel's two meters.
#[derive(Debug, Clone, Copy, Default)]
pub struct CaptureLevels {
    pub near: f32,
    pub far: f32,
}

/// A shared, growable buffer of mono `f32` samples for one channel.
type Samples = Arc<Mutex<Vec<f32>>>;

/// A running capture source: a thread that owns one OS audio stream and
/// shuts it down when told.
struct Source {
    stop: Sender<()>,
    thread: JoinHandle<()>,
}

impl Source {
    /// Signal the source thread to stop and wait for it to finish.
    fn shutdown(self) {
        let _ = self.stop.send(());
        let _ = self.thread.join();
    }
}

/// Records the microphone and macOS system audio into one stereo WAV.
///
/// While recording, captured samples are spilled to raw scratch files
/// every few seconds, so memory stays flat no matter how long the
/// meeting runs; the scratch files are resampled, interleaved into the
/// stereo WAV, and deleted on stop.
///
/// System audio is captured only when `capture_system_audio` is set —
/// a remote call has a far end, a solo memo or an in-person meeting
/// does not. When it is unset the recording is microphone-only and the
/// far channel of the WAV is silence.
pub struct Recorder {
    out_path: PathBuf,
    /// Raw f32 scratch files, one per channel, written while recording.
    near_raw: PathBuf,
    far_raw: PathBuf,
    near: Samples,
    far: Samples,
    /// The microphone's true sample rate, learned when its stream opens.
    near_rate: Arc<AtomicU32>,
    /// The far end's true sample rate, learned when the system-audio
    /// stream opens. macOS pins it to [`SYSTEM_RATE`] via SCK; Windows
    /// reports the WASAPI device's native mix rate (typically 48 kHz,
    /// sometimes 44.1 kHz).
    far_rate: Arc<AtomicU32>,
    /// Whether to capture the far end (system audio) — true only for a
    /// remote call. See [`crate::RecordingScenario`].
    capture_system_audio: bool,
    mic: Option<Source>,
    system: Option<Source>,
    flusher: Option<Source>,
}

impl Recorder {
    /// Create a recorder that will write its stereo WAV to `out_path`
    /// when [`AudioCapture::stop`] is called. `capture_system_audio`
    /// selects whether the far end is recorded — set it for a remote
    /// call, clear it for a microphone-only memo or in-person meeting.
    pub fn new(out_path: PathBuf, capture_system_audio: bool) -> Self {
        Self {
            near_raw: out_path.with_extension("near.f32"),
            far_raw: out_path.with_extension("far.f32"),
            out_path,
            near: Samples::default(),
            far: Samples::default(),
            // Overwritten by the microphone thread once its stream opens.
            // The fallback value doesn't matter — stop() only reads this
            // after the mic thread has joined and set the real rate.
            near_rate: Arc::new(AtomicU32::new(SYSTEM_RATE)),
            // Overwritten by the system-audio thread (when present) once
            // its stream opens. Left at [`SYSTEM_RATE`] when system audio
            // is not captured — there is no far signal to resample so the
            // rate is unused.
            far_rate: Arc::new(AtomicU32::new(SYSTEM_RATE)),
            capture_system_audio,
            mic: None,
            system: None,
            flusher: None,
        }
    }
}

impl AudioCapture for Recorder {
    fn start(&mut self) -> Result<()> {
        if self.mic.is_some() {
            return Err(EngineError::Capture("already recording".into()));
        }
        // Fresh buffers for a new recording.
        self.near = Samples::default();
        self.far = Samples::default();

        // The microphone is required — without it there is no recording.
        let mic = spawn_source("microphone", {
            let near = self.near.clone();
            let rate = self.near_rate.clone();
            move |ready, stop| microphone(near, rate, ready, stop)
        })?;
        // The flusher keeps memory flat by spilling to disk; required.
        let flusher = match spawn_source("recording buffer", {
            let near = self.near.clone();
            let far = self.far.clone();
            let near_raw = self.near_raw.clone();
            let far_raw = self.far_raw.clone();
            move |ready, stop| spill_loop(near, far, near_raw, far_raw, ready, stop)
        }) {
            Ok(source) => source,
            Err(e) => {
                mic.shutdown();
                return Err(e);
            }
        };
        // System audio is captured only for a remote call, and even then
        // best-effort: Screen-Recording permission may not be granted
        // yet. If it is not wanted, or fails, the recording continues
        // with just the near end.
        let system = if self.capture_system_audio {
            match spawn_source("system audio", {
                let far = self.far.clone();
                let rate = self.far_rate.clone();
                move |ready, stop| system_audio(far, rate, ready, stop)
            }) {
                Ok(source) => Some(source),
                Err(e) => {
                    eprintln!("scribe: system audio unavailable, recording microphone only — {e}");
                    None
                }
            }
        } else {
            None
        };

        self.mic = Some(mic);
        self.flusher = Some(flusher);
        self.system = system;
        Ok(())
    }

    fn stop(&mut self) -> Result<()> {
        let mic = self
            .mic
            .take()
            .ok_or_else(|| EngineError::Capture("not recording".into()))?;
        mic.shutdown();
        if let Some(system) = self.system.take() {
            system.shutdown();
        }
        // The flusher stops last: with the sources joined there are no
        // more appends, so its final spill captures every sample.
        if let Some(flusher) = self.flusher.take() {
            flusher.shutdown();
        }

        let near = read_raw_f32(&self.near_raw)?;
        let far = read_raw_f32(&self.far_raw)?;
        let result = write_stereo_wav(
            &near,
            self.near_rate.load(Ordering::Relaxed),
            &far,
            self.far_rate.load(Ordering::Relaxed),
            &self.out_path,
        );
        let _ = std::fs::remove_file(&self.near_raw);
        let _ = std::fs::remove_file(&self.far_raw);
        result
    }

    fn levels(&self) -> CaptureLevels {
        CaptureLevels {
            near: rms_level(&self.near),
            far: rms_level(&self.far),
        }
    }
}

/// Spawn a capture source on its own thread, waiting for it to confirm
/// the OS stream opened before returning.
fn spawn_source(
    label: &'static str,
    body: impl FnOnce(&Sender<std::result::Result<(), String>>, Receiver<()>) + Send + 'static,
) -> Result<Source> {
    let (ready_tx, ready_rx) = mpsc::channel::<std::result::Result<(), String>>();
    let (stop_tx, stop_rx) = mpsc::channel::<()>();
    let thread = std::thread::spawn(move || body(&ready_tx, stop_rx));

    match ready_rx.recv() {
        Ok(Ok(())) => Ok(Source { stop: stop_tx, thread }),
        Ok(Err(e)) => {
            let _ = thread.join();
            Err(EngineError::Capture(format!("{label}: {e}")))
        }
        Err(_) => Err(EngineError::Capture(format!(
            "{label}: capture thread exited before it was ready"
        ))),
    }
}

/// Spill buffered audio to the raw scratch files while recording, so the
/// whole recording never sits in memory. On each tick it writes all but
/// the recent [`LEVEL_WINDOW`] (kept for the meters); on stop it drains
/// everything that remains.
fn spill_loop(
    near: Samples,
    far: Samples,
    near_path: PathBuf,
    far_path: PathBuf,
    ready: &Sender<std::result::Result<(), String>>,
    stop: Receiver<()>,
) {
    let near_file = match File::create(&near_path) {
        Ok(f) => f,
        Err(e) => {
            let _ = ready.send(Err(format!("could not open recording buffer: {e}")));
            return;
        }
    };
    let far_file = match File::create(&far_path) {
        Ok(f) => f,
        Err(e) => {
            let _ = ready.send(Err(format!("could not open recording buffer: {e}")));
            return;
        }
    };
    let mut near_w = BufWriter::new(near_file);
    let mut far_w = BufWriter::new(far_file);
    let _ = ready.send(Ok(()));

    loop {
        match stop.recv_timeout(FLUSH_INTERVAL) {
            Err(RecvTimeoutError::Timeout) => {
                spill(&near, &mut near_w, LEVEL_WINDOW);
                spill(&far, &mut far_w, LEVEL_WINDOW);
            }
            // Stop signalled (or the recorder dropped): take everything.
            _ => {
                spill(&near, &mut near_w, 0);
                spill(&far, &mut far_w, 0);
                let _ = near_w.flush();
                let _ = far_w.flush();
                return;
            }
        }
    }
}

/// Drain all but the last `keep` samples from `samples` and append them
/// to `writer` as little-endian `f32`.
fn spill(samples: &Samples, writer: &mut impl Write, keep: usize) {
    let chunk: Vec<f32> = {
        let Ok(mut buf) = samples.lock() else {
            return;
        };
        if buf.len() <= keep {
            return;
        }
        let take = buf.len() - keep;
        buf.drain(..take).collect()
    };
    let mut bytes = Vec::with_capacity(chunk.len() * 4);
    for sample in &chunk {
        bytes.extend_from_slice(&sample.to_le_bytes());
    }
    let _ = writer.write_all(&bytes);
}

/// Read a raw little-endian `f32` scratch file back into samples.
fn read_raw_f32(path: &Path) -> Result<Vec<f32>> {
    let bytes = std::fs::read(path)
        .map_err(|e| EngineError::Capture(format!("could not read recording buffer: {e}")))?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// RMS of the most recent [`LEVEL_WINDOW`] samples, scaled to `0.0..=1.0`.
fn rms_level(samples: &Samples) -> f32 {
    let Ok(buf) = samples.lock() else {
        return 0.0;
    };
    let window = LEVEL_WINDOW.min(buf.len());
    if window == 0 {
        return 0.0;
    }
    let tail = &buf[buf.len() - window..];
    let mean_sq = tail.iter().map(|s| s * s).sum::<f32>() / window as f32;
    (mean_sq.sqrt() * LEVEL_GAIN).min(1.0)
}

/// Resample both channels to 16 kHz, interleave them, and write a stereo
/// 16-bit WAV — channel 0 the near end, channel 1 the far end. The
/// shorter channel is padded with silence.
fn write_stereo_wav(
    near: &[f32],
    near_rate: u32,
    far: &[f32],
    far_rate: u32,
    path: &Path,
) -> Result<()> {
    let near = resample_to_whisper(near, near_rate);
    let far = resample_to_whisper(far, far_rate);
    let frames = near.len().max(far.len());

    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: crate::audio::WHISPER_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .map_err(|e| EngineError::Capture(format!("could not create recording: {e}")))?;
    for i in 0..frames {
        let n = to_i16(near.get(i).copied().unwrap_or(0.0));
        let f = to_i16(far.get(i).copied().unwrap_or(0.0));
        writer
            .write_sample(n)
            .and_then(|()| writer.write_sample(f))
            .map_err(|e| EngineError::Capture(format!("could not write recording: {e}")))?;
    }
    writer
        .finalize()
        .map_err(|e| EngineError::Capture(format!("could not finalize recording: {e}")))
}

fn to_i16(sample: f32) -> i16 {
    (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16
}

// --- microphone (cpal, cross-platform) -------------------------------

/// Open the default microphone and stream mono samples into `near`
/// until `stop` fires.
fn microphone(
    near: Samples,
    near_rate: Arc<AtomicU32>,
    ready: &Sender<std::result::Result<(), String>>,
    stop: Receiver<()>,
) {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

    let open = || -> std::result::Result<cpal::Stream, String> {
        let device = cpal::default_host()
            .default_input_device()
            .ok_or("no microphone found")?;
        let supported = device
            .default_input_config()
            .map_err(|e| format!("no usable microphone format: {e}"))?;
        let channels = supported.channels() as usize;
        let format = supported.sample_format();
        near_rate.store(supported.sample_rate(), Ordering::Relaxed);
        let config: cpal::StreamConfig = supported.into();

        let on_error = |e| eprintln!("scribe: microphone stream error: {e}");
        let stream = match format {
            cpal::SampleFormat::F32 => {
                let near = near.clone();
                device.build_input_stream(
                    &config,
                    move |data: &[f32], _: &_| append_mono(&near, data, channels, |s| s),
                    on_error,
                    None,
                )
            }
            cpal::SampleFormat::I16 => {
                let near = near.clone();
                device.build_input_stream(
                    &config,
                    move |data: &[i16], _: &_| {
                        append_mono(&near, data, channels, |s| s as f32 / 32768.0)
                    },
                    on_error,
                    None,
                )
            }
            other => return Err(format!("unsupported microphone sample format: {other:?}")),
        }
        .map_err(|e| format!("could not open microphone: {e}"))?;
        stream.play().map_err(|e| format!("could not start microphone: {e}"))?;
        Ok(stream)
    };

    match open() {
        Ok(_stream) => {
            let _ = ready.send(Ok(()));
            // Hold the stream alive — cpal delivers samples on its own
            // thread — until the recorder calls stop.
            let _ = stop.recv();
        }
        Err(e) => {
            let _ = ready.send(Err(e));
        }
    }
}

/// Average each interleaved frame to mono and append it to `buf`.
fn append_mono<T: Copy>(buf: &Samples, data: &[T], channels: usize, to_f32: impl Fn(T) -> f32) {
    let channels = channels.max(1);
    let Ok(mut buf) = buf.lock() else {
        return;
    };
    for frame in data.chunks_exact(channels) {
        let sum: f32 = frame.iter().map(|&s| to_f32(s)).sum();
        buf.push(sum / channels as f32);
    }
}

// --- system audio (ScreenCaptureKit, macOS) --------------------------

#[cfg(target_os = "macos")]
fn system_audio(
    far: Samples,
    far_rate: Arc<AtomicU32>,
    ready: &Sender<std::result::Result<(), String>>,
    stop: Receiver<()>,
) {
    use screencapturekit::prelude::*;

    let open = || -> std::result::Result<SCStream, String> {
        let content = SCShareableContent::get().map_err(|e| format!("{e}"))?;
        let display = content
            .displays()
            .into_iter()
            .next()
            .ok_or("no display available for audio capture")?;
        let filter = SCContentFilter::create()
            .with_display(&display)
            .with_excluding_windows(&[])
            .build();
        // Video is captured but never handled — keep its frames tiny.
        let config = SCStreamConfiguration::new()
            .with_width(16)
            .with_height(16)
            .with_captures_audio(true)
            .with_sample_rate(SYSTEM_RATE as i32)
            .with_channel_count(2);

        let mut stream = SCStream::new(&filter, &config);
        stream.add_output_handler(SystemAudioHandler { far: far.clone() }, SCStreamOutputType::Audio);
        stream
            .start_capture()
            .map_err(|e| format!("could not start system-audio capture: {e}"))?;
        Ok(stream)
    };

    match open() {
        Ok(stream) => {
            // SCK is configured for SYSTEM_RATE above — record that.
            far_rate.store(SYSTEM_RATE, Ordering::Relaxed);
            let _ = ready.send(Ok(()));
            let _ = stop.recv();
            if let Err(e) = stream.stop_capture() {
                eprintln!("scribe: system-audio stop failed: {e}");
            }
        }
        Err(e) => {
            let _ = ready.send(Err(e));
        }
    }
}

/// ScreenCaptureKit output handler — averages each system-audio sample
/// buffer to mono and appends it to the far channel.
#[cfg(target_os = "macos")]
struct SystemAudioHandler {
    far: Samples,
}

#[cfg(target_os = "macos")]
impl screencapturekit::prelude::SCStreamOutputTrait for SystemAudioHandler {
    fn did_output_sample_buffer(
        &self,
        sample: screencapturekit::prelude::CMSampleBuffer,
        of_type: screencapturekit::prelude::SCStreamOutputType,
    ) {
        use screencapturekit::prelude::{CMSampleBufferExt, SCStreamOutputType};
        if !matches!(of_type, SCStreamOutputType::Audio) {
            return;
        }
        let Some(list) = sample.audio_buffer_list() else {
            return;
        };
        // ScreenCaptureKit delivers non-interleaved float audio: one
        // buffer per channel. Average the channels into mono.
        let channels: Vec<Vec<f32>> = (0..list.num_buffers())
            .filter_map(|i| list.buffer(i))
            .map(|b| bytes_to_f32(b.data()))
            .filter(|c| !c.is_empty())
            .collect();
        if channels.is_empty() {
            return;
        }
        let frames = channels.iter().map(Vec::len).min().unwrap_or(0);
        if let Ok(mut far) = self.far.lock() {
            for f in 0..frames {
                let sum: f32 = channels.iter().map(|c| c[f]).sum();
                far.push(sum / channels.len() as f32);
            }
        }
    }
}

/// Reinterpret a little-endian `f32` PCM byte slice as samples.
#[cfg(target_os = "macos")]
fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

// --- system audio (WASAPI loopback, Windows) -------------------------
//
// On Windows, cpal's WASAPI backend reuses the same `build_input_stream`
// path for output devices: when the device's data-flow is `eRender`,
// cpal automatically sets `AUDCLNT_STREAMFLAGS_LOOPBACK`, so opening the
// default *output* device as input gives us the same mix the user is
// hearing (Zoom remote audio, browser audio, the system mixer). No
// permission prompt — WASAPI loopback is not gated like macOS Screen
// Recording, only the microphone needs consent.

#[cfg(target_os = "windows")]
fn system_audio(
    far: Samples,
    far_rate: Arc<AtomicU32>,
    ready: &Sender<std::result::Result<(), String>>,
    stop: Receiver<()>,
) {
    use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

    let open = || -> std::result::Result<cpal::Stream, String> {
        let device = cpal::default_host()
            .default_output_device()
            .ok_or("no default output device for system-audio loopback")?;
        // Use the device's preferred *output* config — that is what
        // WASAPI loopback will deliver to us as input. Querying the
        // input-side configs on a render endpoint is not meaningful.
        let supported = device
            .default_output_config()
            .map_err(|e| format!("no usable system-audio format: {e}"))?;
        let channels = supported.channels() as usize;
        let format = supported.sample_format();
        far_rate.store(supported.sample_rate(), Ordering::Relaxed);
        let config: cpal::StreamConfig = supported.into();

        let on_error = |e| eprintln!("scribe: system-audio stream error: {e}");
        let stream = match format {
            cpal::SampleFormat::F32 => {
                let far = far.clone();
                device.build_input_stream(
                    &config,
                    move |data: &[f32], _: &_| append_mono(&far, data, channels, |s| s),
                    on_error,
                    None,
                )
            }
            cpal::SampleFormat::I16 => {
                let far = far.clone();
                device.build_input_stream(
                    &config,
                    move |data: &[i16], _: &_| {
                        append_mono(&far, data, channels, |s| s as f32 / 32768.0)
                    },
                    on_error,
                    None,
                )
            }
            cpal::SampleFormat::U16 => {
                let far = far.clone();
                device.build_input_stream(
                    &config,
                    move |data: &[u16], _: &_| {
                        // U16 PCM is unsigned with mid-scale 32768.
                        append_mono(&far, data, channels, |s| {
                            (s as f32 - 32768.0) / 32768.0
                        })
                    },
                    on_error,
                    None,
                )
            }
            other => {
                return Err(format!("unsupported system-audio sample format: {other:?}"))
            }
        }
        .map_err(|e| format!("could not open WASAPI loopback: {e}"))?;
        stream
            .play()
            .map_err(|e| format!("could not start WASAPI loopback: {e}"))?;
        Ok(stream)
    };

    match open() {
        Ok(_stream) => {
            let _ = ready.send(Ok(()));
            // Hold the stream alive — cpal delivers samples on its own
            // thread — until the recorder calls stop. Dropping the
            // stream value at the end of this scope stops loopback.
            let _ = stop.recv();
        }
        Err(e) => {
            let _ = ready.send(Err(e));
        }
    }
}

#[cfg(not(any(target_os = "macos", target_os = "windows")))]
fn system_audio(
    _far: Samples,
    _far_rate: Arc<AtomicU32>,
    ready: &Sender<std::result::Result<(), String>>,
    _stop: Receiver<()>,
) {
    // Linux / BSD / other: no first-party system-audio capture path
    // ships in v0.1. The host app should clear `capture_system_audio`
    // on these platforms — falling through to here just records mic.
    let _ = ready.send(Err(
        "system-audio capture is not supported on this platform".into(),
    ));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decode a WAV back into its two channels.
    fn read_channels(path: &Path) -> (Vec<i16>, Vec<i16>) {
        let mut reader = hound::WavReader::open(path).expect("open wav");
        let all: Vec<i16> = reader.samples::<i16>().map(|s| s.expect("sample")).collect();
        let near = all.iter().step_by(2).copied().collect();
        let far = all.iter().skip(1).step_by(2).copied().collect();
        (near, far)
    }

    #[test]
    fn stereo_wav_keeps_channels_separate() {
        let path = std::env::temp_dir().join("scribe-capture-test.wav");
        // Near at full scale, far silent — both already at 16 kHz.
        write_stereo_wav(&[1.0, 1.0, 1.0], 16_000, &[0.0, 0.0, 0.0], 16_000, &path)
            .expect("write wav");

        let (near, far) = read_channels(&path);
        assert_eq!(near.len(), 3);
        assert!(near.iter().all(|&s| s > 30_000), "near channel lost its signal");
        assert!(far.iter().all(|&s| s == 0), "far channel should be silent");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn spill_drains_all_but_the_kept_tail_and_round_trips() {
        let path = std::env::temp_dir().join("scribe-spill-test.f32");
        let samples: Samples = Arc::new(Mutex::new(vec![0.1, 0.2, 0.3, 0.4, 0.5]));
        {
            let mut writer = BufWriter::new(File::create(&path).expect("create scratch"));
            spill(&samples, &mut writer, 2); // keep the last two
            writer.flush().expect("flush");
        }
        // Three spilled to disk, two kept in memory for the meters.
        assert_eq!(samples.lock().unwrap().len(), 2);
        let read = read_raw_f32(&path).expect("read scratch");
        assert_eq!(read.len(), 3);
        assert!((read[0] - 0.1).abs() < 1e-6);
        assert!((read[2] - 0.3).abs() < 1e-6);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn shorter_channel_is_padded_with_silence() {
        let path = std::env::temp_dir().join("scribe-capture-pad-test.wav");
        write_stereo_wav(&[0.5, 0.5, 0.5, 0.5], 16_000, &[0.5], 16_000, &path)
            .expect("write wav");

        let (near, far) = read_channels(&path);
        assert_eq!(near.len(), 4, "frame count follows the longer channel");
        assert_eq!(far.len(), 4);
        assert!(far[0] > 10_000, "far's real sample survived");
        assert_eq!(&far[1..], &[0, 0, 0], "far was padded with silence");
        std::fs::remove_file(&path).ok();
    }
}
