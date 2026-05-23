# Security

`scribe-engine` is the on-device core of a privacy-sensitive product;
its security posture matters to its users.

## The invariant

The engine performs **zero network I/O**. If you find a code path that
opens a socket, makes an HTTP request, resolves DNS, or otherwise
communicates beyond the local machine, that is a security defect and we
want to know.

## Reporting a vulnerability

Please report security issues privately rather than opening a public
GitHub issue.

**Email:** aelgazar@simkinselgazar.com

Include:

- A description of the issue and its potential impact
- Steps to reproduce (a minimal proof of concept if you have one)
- The version (commit hash) you tested against
- Your contact information for follow-up

We aim to acknowledge reports within three business days and to
coordinate disclosure timelines with reporters. If you do not hear back
within a week, please follow up.

## Scope

In scope:

- The `scribe-engine` library crate
- The `scribe-notes-llm` sidecar binary
- The build environment (`.cargo/config.toml`, `Cargo.toml`,
  `Cargo.lock`)

Out of scope (audit but report to the upstream maintainers):

- Vulnerabilities in `whisper.cpp`, `llama.cpp`, `cpal`,
  `screencapturekit`, or other upstream dependencies — report directly
  to those projects. Cross-reference us once they have an advisory so
  we can pin to a fixed release.
- The desktop application that consumes this engine — that is a
  separate codebase with its own disclosure channel.

## What gets a CVE

Anything that:

- Breaks the on-device invariant (the engine reaches the network)
- Allows an attacker who can write to one input (an audio file, a
  model file, the sidecar's stdin/stdout) to read or write outside
  the engine's intended scope
- Causes the engine to load or execute code from an
  attacker-controllable source
- Leaks transcript content, model output, or other potentially
  confidential data through unexpected channels (stderr, log files,
  external services)

## What does not

- A consumer app misconfiguration that exposes engine outputs (report
  to that app's project)
- A model file producing inaccurate transcription or notes (report to
  the model's maintainers)
- Performance, memory use, or stability bugs that do not have a
  security consequence (open a regular issue)
