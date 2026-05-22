//! Audio capture — the far end (system audio) and the near end
//! (microphone) as two independent streams.
//!
//! Platform backends implement [`AudioCapture`]: ScreenCaptureKit on
//! macOS, WASAPI loopback on Windows. No backend ships in this scaffold;
//! they land during build.

use crate::Result;

/// A live capture session. Backends are platform-specific; the host app
/// holds a `Box<dyn AudioCapture>` and does not know which one it has.
pub trait AudioCapture: Send {
    /// Begin capturing both streams.
    fn start(&mut self) -> Result<()>;

    /// Stop capturing and finalize the recording.
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
