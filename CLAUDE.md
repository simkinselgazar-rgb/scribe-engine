---
date: 2026-05-21
tags: [project, code, rust, on-device-scribe, scribe-engine, p0-3, open-source]
type: project
---

# CLAUDE.md — Scribe Engine

## Project Overview
The open-source core of On-Device Scribe ([[Business Function AI Templates|P0 #3]]) — a Rust crate that captures conversation audio, transcribes it, and turns the transcript into structured notes, **entirely on-device**. It is consumed by the closed-source desktop app (`../on-device-scribe`). Apache 2.0; destined for a public repo on the `simkinselgazar-rgb` org.

## The invariant — read first
**Nothing in this crate performs network I/O** — not for telemetry, not for fetching models, not for anything. Audio, transcripts, and notes never leave the machine. A change that adds a network call breaks the product (see [[on-device-scribe-diagnosis|the diagnosis]]). If a feature seems to need the network, stop and surface it.

## Tech Stack
- **Language:** Rust, edition 2021 — a Cargo workspace (a `lib` crate + the `notes-llm` sidecar bin)
- **Transcription backend:** whisper.cpp via `whisper-rs`, in-process. Metal on Apple Silicon. Model file supplied by the host app.
- **Notes backend:** a local LLM via `llama-cpp-2`, in the **`scribe-notes-llm` sidecar process** (see below). Gemma GGUF model (v0.1 default: Gemma 4 E4B, Apache-2.0); the sidecar uses the Gemma prompt format.
- **Audio capture:** microphone via `cpal` (cross-platform); macOS system audio via `screencapturekit`. Recordings are stereo WAV — channel 0 near (mic), channel 1 far (system audio).
- **Tests:** `cargo test`
- **License:** Apache 2.0

## The sidecar — why notes generation is a separate process
whisper.cpp and llama.cpp each statically bundle their own copy of `ggml`. Linked into one binary, the duplicate symbols collide and corrupt inference (observed: a load-time crash). So the notes LLM lives in its own crate, `notes-llm/` (`scribe-notes-llm`), built as a standalone binary. `SidecarNotesGenerator` spawns it and talks over a small JSON protocol on stdin/stdout. **Never add `llama-cpp-2` to the `scribe-engine` lib** — that reintroduces the collision. The lib (and the desktop app that links it) carries only whisper.cpp's ggml.

## Conventions
- **Three subsystems, each a trait with pluggable backends:** `capture` → `AudioCapture` (`Recorder`), `transcribe` → `Transcriber` (`WhisperTranscriber`), `notes` → `NotesGenerator` (`SidecarNotesGenerator`). The host app composes them; it never depends on a concrete backend.
- Shared data types live in `model.rs` and are re-exported from `lib.rs`.
- Every fallible engine call returns `crate::Result<T>` (`EngineError`). No `unwrap`/`expect` in library code.
- `snake_case` files and modules, one subsystem per file.
- Keep the trait surface synchronous until a backend genuinely needs `async`.

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
- The transcription and notes integration tests need assets in `target/test-assets/` (a Whisper GGML model, a notes GGUF model, sample audio) and the built sidecar; they skip cleanly when absent.

## Build environment
- `cmake` is required (whisper.cpp and llama.cpp build from source).
- `screencapturekit` pulls Swift bridge crates. With only the Command Line Tools installed (no full Xcode), `.cargo/config.toml` adds the Swift static-archive search path and the OS Swift-runtime rpath. Keep that file.

## Current State
**Phases B, C, D complete and tested (2026-05-22).**
- **B — transcription:** `WhisperTranscriber` (whisper.cpp). Verified end-to-end against a real model + stereo sample.
- **C — notes:** `SidecarNotesGenerator` + the `scribe-notes-llm` sidecar (llama.cpp). Verified end-to-end producing structured notes from a transcript.
- **D — capture:** `Recorder` — `cpal` microphone + `screencapturekit` system audio → stereo WAV. Compiles and links; WAV/mix/resample logic unit-tested. Live OS capture needs Screen-Recording + Microphone permissions and a real call — a manual QA step.

Not yet pushed to GitHub.

## Known Issues
- **Live capture is unverified in CI.** `Recorder::start` needs OS permissions; only the offline logic is unit-tested.
- **Long meetings** are handled by chunked map-reduce in the notes sidecar (split into overlapping ~10-min chunks, summarize each, one reduce pass consolidates). `Recorder` streams audio to disk so memory stays flat for any recording length.
- **Windows system audio (WASAPI loopback) is not implemented** — `system_audio` returns an error off macOS.

## Vault Integration
This project lives inside The Vault. It inherits vault conventions from the parent `CLAUDE.md`:
- Follow Obsidian frontmatter and `[[link]]` conventions for any markdown notes
- When you create significant new content in this project, consider whether it affects any vault-level indexes (e.g., `Projects/index.md`)
- Cross-link related project notes back to the vault's `Research/`, `People/`, or `Projects/` areas where relevant
- Action items from this project belong in the vault-root `Action Items.md`, not here

## Related
- [[on-device-scribe-diagnosis|Step 3 Diagnosis]] · [[On-Device Scribe - Design Brief|Design Brief]] · [[on-device-scribe-design-context|Design Context]]
- `../on-device-scribe` — the closed-source desktop app that consumes this crate
- [[Local Whisper Engine]] — the prior Python transcription tool; this crate reuses the approach, not the code
