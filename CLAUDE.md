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
- **Language:** Rust, edition 2021 — a `lib` crate
- **Transcription backend:** whisper.cpp via `whisper-rs` (Metal on Apple Silicon, CPU/CUDA on Windows). mlx-whisper is a possible later optional accelerator.
- **Notes backend:** a local LLM via a `llama.cpp` binding
- **Audio capture:** platform FFI — ScreenCaptureKit (macOS), WASAPI loopback (Windows)
- **Tests:** `cargo test`
- **License:** Apache 2.0

## Conventions
- **Three subsystems, each a trait with pluggable backends:** `capture` → `AudioCapture`, `transcribe` → `Transcriber`, `notes` → `NotesGenerator`. The host app composes them; it never depends on a concrete backend.
- Shared data types live in `model.rs` and are re-exported from `lib.rs`.
- Every fallible engine call returns `crate::Result<T>` (`EngineError`). No `unwrap`/`expect` in library code.
- `snake_case` files and modules, one subsystem per file.
- Keep the trait surface synchronous until a backend genuinely needs `async`.

## File Structure
| What | Where |
|------|-------|
| Crate root, `EngineError`, re-exports | `src/lib.rs` |
| Shared data types (`Transcript`, `Notes`, …) | `src/model.rs` |
| Audio-capture trait + platform backends | `src/capture.rs` |
| Transcription trait + backends | `src/transcribe.rs` |
| Note-generation trait + backends | `src/notes.rs` |
| Tests | `tests/` |

## Testing
- Run all: `cargo test`
- Check / lint: `cargo check` · `cargo clippy`

## Current State
Scaffolded 2026-05-21 (Dalio step 5). The public API is defined — three traits plus the shared `model` — and the crate compiles. **No backends are implemented yet**; they land during build, alongside `/impeccable craft` on the app. Not yet pushed to GitHub.

## Known Issues
*(none yet)*

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
