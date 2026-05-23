# CLAUDE.md — scribe-engine

A short orientation for AI assistants (and human contributors) working
on this crate. For an audience-facing introduction, read `README.md`
first.

## Project Overview

The open-source on-device transcription and notes core of a closed-source
paid desktop app for confidential client meetings. A Rust workspace that
captures conversation audio, transcribes it, and turns the transcript
into structured notes. Apache-2.0.

## The invariant — read first

**Nothing in this crate performs network I/O** — not for telemetry, not
for fetching models, not for anything. Audio, transcripts, and notes
never leave the machine. A change that adds a network call breaks the
product. If a feature seems to need the network, stop and surface it.

This is the load-bearing claim that makes the engine auditable. Don't
weaken it.

## Tech Stack

- **Language:** Rust, edition 2021 — a Cargo workspace (a `lib` crate
  plus the `notes-llm` sidecar binary)
- **Transcription backend:** whisper.cpp via `whisper-rs`, in-process.
  Metal on Apple Silicon. Model file supplied by the host app.
- **Notes backend:** a local LLM via `llama-cpp-2`, in the
  **`scribe-notes-llm` sidecar process** (see below). Gemma 4 GGUF is
  the v0.1 default; the sidecar uses the Gemma prompt format. Gemma is
  governed by the Gemma Terms of Use (a custom Google license, *not*
  Apache-2.0); the consuming app is responsible for confirming the
  terms are acceptable for its deployment.
- **Audio capture:** microphone via `cpal` (cross-platform); macOS
  system audio via `screencapturekit`. Recordings are stereo WAV —
  channel 0 near (mic), channel 1 far (system audio).
- **Tests:** `cargo test`
- **License:** Apache 2.0

## The sidecar — why notes generation is a separate process

whisper.cpp and llama.cpp each statically bundle their own copy of
`ggml`. Linked into one binary, the duplicate symbols collide and
corrupt inference (we observed load-time crashes and silent corruption).
So the notes LLM lives in its own crate, `notes-llm/`
(`scribe-notes-llm`), built as a standalone binary.
`SidecarNotesGenerator` spawns it and talks over a small JSON protocol
on stdin/stdout. **Never add `llama-cpp-2` to the `scribe-engine` lib**
— that reintroduces the collision. The lib (and any app that links it)
carries only whisper.cpp's ggml.

The sidecar's stderr is reserved for status messages to its parent
process. **Do not include transcript content or model output snippets
in error strings** — the parent app may log stderr, and that's how
confidential content ends up in places it shouldn't.

## Conventions

- **Three subsystems, each a trait with pluggable backends:** `capture`
  → `AudioCapture` (`Recorder`), `transcribe` → `Transcriber`
  (`WhisperTranscriber`), `notes` → `NotesGenerator`
  (`SidecarNotesGenerator`). The host app composes them; it never
  depends on a concrete backend.
- **`RecordingScenario`** (`model.rs`: `SoloMemo` / `VirtualMeeting` /
  `InPersonMeeting`) is threaded through all three subsystems: it
  selects whether `Recorder` captures system audio, whether `transcribe`
  reads the far channel (silence there only invites Whisper
  hallucinations), and how the sidecar frames the notes prompt. Only a
  `VirtualMeeting` (a call held through the computer) is two-channel;
  the other two are single-microphone.
- Shared data types live in `model.rs` and are re-exported from
  `lib.rs`.
- Every fallible engine call returns `crate::Result<T>` (`EngineError`).
  No `unwrap`/`expect` in library code.
- `snake_case` files and modules, one subsystem per file.
- Keep the trait surface synchronous until a backend genuinely needs
  `async`.

## File Structure

| What | Where |
|------|-------|
| Crate root, `EngineError`, re-exports | `src/lib.rs` |
| Shared data types (`Transcript`, `Notes`, …) | `src/model.rs` |
| WAV decode + resample for transcription | `src/audio.rs` |
| Audio-capture trait + `Recorder` backend | `src/capture.rs` |
| Transcription trait + `WhisperTranscriber` | `src/transcribe.rs` |
| Notes trait + `SidecarNotesGenerator` (spawns the sidecar) | `src/notes.rs` |
| The notes LLM sidecar binary | `notes-llm/src/main.rs` |
| Integration tests | `tests/` |
| Test assets (git-ignored, under `target/`) | `target/test-assets/` |

## Testing

- Run all: `cargo test` · check / lint: `cargo check` · `cargo clippy`
- The transcription and notes integration tests need assets in
  `target/test-assets/` (a Whisper GGML model, a Gemma GGUF model, a
  stereo sample WAV) and the built sidecar binary; they skip cleanly
  when those are absent so `cargo test` stays green on machines without
  the assets.

## Build environment

- `cmake` is required (whisper.cpp and llama.cpp build from source).
- `screencapturekit` pulls Swift bridge crates. With only the macOS
  Command Line Tools installed (no full Xcode), `.cargo/config.toml`
  adds the Swift static-archive search path and the OS Swift-runtime
  rpath. Keep that file. Contributors with the full Xcode installed
  may need to adjust the path.

## Status

v0.1: the three engine subsystems are implemented, tested, and used in
production by the consuming desktop app.

- **Transcription:** `WhisperTranscriber` (whisper.cpp). Verified
  end-to-end against a real model + stereo sample.
- **Notes:** `SidecarNotesGenerator` + the `scribe-notes-llm` sidecar
  (llama.cpp). Verified end-to-end producing structured notes from a
  transcript.
- **Capture:** `Recorder` — `cpal` microphone + `screencapturekit`
  system audio → stereo WAV with disk-streaming for long recordings.
  Compiles and links; WAV/mix/resample logic unit-tested. Live OS
  capture requires Screen-Recording + Microphone permissions and is
  exercised by the consuming app.

## Known Issues

- **Live capture is unverified in CI.** `Recorder::start` needs OS
  permissions; only the offline logic is unit-tested.
- **Long meetings** are handled by chunked map-reduce in the notes
  sidecar (overlapping ~10-minute chunks, summarize each, one reduce
  pass consolidates). `Recorder` streams audio to disk so memory stays
  flat for any recording length.
- **Windows system audio (WASAPI loopback) is not implemented** —
  `system_audio` returns an error off macOS.
