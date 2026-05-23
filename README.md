# scribe-engine

The on-device transcription and notes core of **On-Device Scribe** — a
Rust workspace that records conversation audio, transcribes it, and
turns the transcript into structured notes. Everything runs locally; no
audio, transcript, or notes ever leave the machine.

Apache-2.0. macOS today; Windows is a fast-follow on the same code.

## The invariant — read first

**Nothing in this crate performs network I/O.** Not for telemetry, not
for fetching models, not for anything. Audio, transcripts, and notes
never leave the machine. A change that adds a network call breaks the
product. The host application is responsible for supplying model files
at known local paths; this crate only reads them.

This is what makes the engine auditable: the dependency graph contains
no HTTP, TLS, or socket libraries, and the source contains no network
syscalls. If you're evaluating it for a confidentiality-sensitive
deployment (legal work, healthcare, accounting, anything covered by
contractual or regulatory privacy), the entire engine is open for that
audit.

## What's in here

A Cargo workspace with two crates:

| Crate | What it is |
|---|---|
| `scribe-engine` (lib) | Three trait subsystems and their default backends: audio capture, transcription, and notes generation. Plus the shared data types (`Transcript`, `Notes`, `RecordingScenario`, `Speaker`). |
| `scribe-notes-llm` (bin) | A standalone sidecar process that runs the notes LLM. Talks to the lib over a small JSON protocol on stdin/stdout. See below for why this is a separate process. |

### The three subsystems

Each is a trait with a pluggable backend:

- **`capture`** — `AudioCapture` / `Recorder`: microphone via `cpal`
  (cross-platform); macOS system audio via `screencapturekit`. Recordings
  are stereo WAV — channel 0 near (microphone), channel 1 far (system
  audio). Long meetings stream to disk so memory stays flat.
- **`transcribe`** — `Transcriber` / `WhisperTranscriber`: whisper.cpp
  via `whisper-rs`. Each channel transcribed independently and merged
  back into one timeline; speaker attribution is structural, not
  statistical.
- **`notes`** — `NotesGenerator` / `SidecarNotesGenerator`: spawns the
  `scribe-notes-llm` sidecar and exchanges JSON over its standard
  streams.

A `RecordingScenario` enum (solo memo / virtual meeting / in-person
meeting) is threaded through all three subsystems and shapes their
behavior: whether system audio is captured, whether the far channel is
read, and how the notes prompt is framed.

### Why the sidecar

whisper.cpp and llama.cpp each statically bundle their own copy of
`ggml`. Linked into one binary the duplicate symbols collide and corrupt
inference (we observed load-time crashes and silent corruption). So the
notes LLM lives in its own crate, `notes-llm/`, built as a standalone
binary. The lib spawns it per session and talks to it over JSON on
stdin/stdout. **Don't add `llama-cpp-2` as a direct dependency of the
lib** — that reintroduces the collision.

## Models

The host app supplies model files; the engine only reads them. Default
choices (and what the consumer app downloads on first run):

| Slot | Default model | License |
|---|---|---|
| Transcription | Whisper `large-v3-turbo` Q5_0 (~547 MB) | Whisper itself is MIT (OpenAI). Use is governed by Whisper's license. |
| Notes | Gemma 4 (Google) — Effective-4B or Effective-2B depending on host RAM | **Gemma Terms of Use** (Google's custom license — *not* Apache-2.0). Read those terms before deploying. |

The engine does not bundle either model. The host app is responsible
for downloading, verifying, and storing them; the engine is given a
path on disk and reads from it.

## Build

```bash
# Library + sidecar
cargo build
cargo build -p scribe-notes-llm --release

# Tests (some integration tests are skipped when assets are absent;
# see tests/notes_llama.rs and tests/transcribe_whisper.rs for what
# they look for under target/test-assets/)
cargo test
```

Requires `cmake` (whisper.cpp and llama.cpp build from source). On
macOS with Command Line Tools but not full Xcode, the `.cargo/config.toml`
adds the Swift static-archive search path and the OS Swift-runtime
rpath that `screencapturekit` needs.

## Status

This is the open-source core of a closed-source paid macOS desktop
app, **On-Device Scribe**, currently in v0.1 development. The engine
itself is feature-complete for v0.1 and used in production by that app.
This repository serves both as the source of truth for the engine and
as an audit-grade reference for anyone evaluating "on-device" claims
about the consuming product.

## Contributing

Issues and PRs welcome. A few rules of the road:

- The on-device invariant is load-bearing. Any change that adds a
  network call, telemetry, or background fetch is a non-starter — open
  an issue first to discuss.
- No `unwrap`/`expect` in library code; return `Result<T, EngineError>`.
- Keep the trait surface synchronous until a backend genuinely needs
  `async`.
- The sidecar's stderr is for status to a parent process; don't include
  transcript content or model output snippets in error strings.

See `CLAUDE.md` for the project conventions in more detail.

## Security

See `SECURITY.md` for vulnerability disclosure.

## License

Apache-2.0. See `LICENSE` and `NOTICE` for the full text and
attributions.
