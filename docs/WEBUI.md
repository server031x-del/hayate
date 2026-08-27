# HAYATE Studio architecture and operations

HAYATE Studio is a local control surface. It does not implement a second model
pipeline. Every submitted request is converted to `GenerationRequest`, checked
by `ExternalH3GenerationBackend.plan()`, and launched through the audited
maybleMyers/h3 entrypoint.

The interface provides readable operational typography and persistent Dark /
Clear display modes. The selected theme is stored only in the local browser.

The generation screen keeps the established `最速` profile unchanged and adds
`高速・画質優先`. The latter preserves the same 20 scheduler points,
SageAttention, 256-pixel VAE tile, and EasyCache threshold while ending cache
reuse at 85% instead of 95%. This forces one additional late Transformer
evaluation where small edges and textures are finalized. Video and audio VAE
attention remain on SDPA; SageAttention is scoped to the denoising Transformer.

The H3 prompt assistant is opt-in. It keeps the manual three-field editor and
adds an OpenAI Responses API authoring path. The model ID is configured in
Settings and requests use the official OpenAI endpoint; the API key is entered through the password field
and stored in Windows Credential Manager through `keyring` when available (or
only for the current server process when no secure backend is available). The
key is never returned by `/api/bootstrap` or `/api/settings`, written to the
SQLite job store, or inherited by a MiniMax H3 child process. The assistant
returns Structured Outputs for subject, action, environment, camera, lighting,
style, separate soundscape/music fields, negative review, and a copy-ready
`final_prompt`. It changes the
generation prompt only after the operator presses **この案を適用**. The job
record retains the original prompt, effective prompt, and template version when
this transformation is used.

To use the AI authoring button, open **設定 → AI prompt director**, enter an
OpenAI API key, choose a model (the default is `gpt-5.6-terra`), and save. Leave
the password field blank on later saves to preserve the existing key. Use the
explicit消去 checkbox when the credential must be removed. The runtime package
includes the official `openai` Python SDK and `keyring`; no key is created by
HAYATE itself. The feature can also use an existing `OPENAI_API_KEY` environment
variable when no UI credential is configured. Each request sends the brief plus
the selected task, duration, canvas, audio preference, and (when improving an
existing prompt) the current prompt text.
If Credential Manager reports a deletion failure, the API returns an error and
the configured status remains visible instead of claiming that the key was
removed.

Library cards and the video detail dialog expose an explicit delete action. A
confirmation dialog names the selected output and explains that the SQLite job
record, MP4, matching `.hayate.log`, and matching `.hayate.json` manifest are
permanently removed. The API accepts deletion only for final job states and
derives the two sidecar paths from the persisted MP4 path; queued or active jobs
and unrelated neighboring files are never deletion targets.

## Runtime layout

```text
Browser (localhost)
  -> FastAPI REST + SSE
  -> SQLite job/history store
  -> single FIFO JobManager
  -> cross-process GPU 0 lease
  -> HAYATE MiniMax H3 entrypoint
  -> maybleMyers/h3 generation pipeline
```

The server stores structured job state in `data/webui/hayate-webui.sqlite3`.
Raw process output remains in the normal `<video>.hayate.log`; it is not copied
into SQLite. Completed CLI outputs are imported from `<video>.hayate.json` on
startup.

## Progress contract

The HAYATE entrypoint emits one-line, flushed JSON events:

```text
HAYATE_EVENT {"schema":1,"type":"progress",...}
```

Load phases and denoise steps use this contract. Text parsing remains only as a
fallback for upstream messages without a structured event. Runtime metrics are
also emitted as an event and retained in the generation manifest.

## Stop behavior

- **Stop and save** creates upstream's `<output>.stop_decode` marker. The engine
  stops at the next denoise callback, decodes its current latent, and records a
  `partial` result.
- **Cancel now** sends a Windows process-group break, waits briefly, then
  terminates only the owned process tree if required. It may not produce a
  playable output.

WebUI, `hayate generate`, `kernel-check`, and `load-check` share a file-backed
GPU lease. This prevents a CLI diagnostic from entering CUDA while a queued
WebUI generation owns GPU 0.

## Local security boundary

- Default bind: `127.0.0.1:7860`; non-loopback binds require
  `--allow-network` and remain unauthenticated.
- OpenAI credential mutation and `/api/prompt-assistant` are loopback-only when
  the server is launched with `--allow-network`; use a real authenticated,
  encrypted reverse proxy before exposing a paid API credential to a network.
- No CORS is enabled. Mutating API calls require the HAYATE UI header and a
  same-origin request. Trusted hosts and a restrictive Content Security Policy
  are applied.
- Browser inputs use managed uploads with random identifiers, image signatures,
  a 25 MiB limit, decoder verification, and a 64-megapixel limit.
- Media and logs are served only through the exact artifact paths persisted by
  the trusted local JobStore. Requests cannot provide a path.
- Model, checkpoint, Python, and upstream paths come from server settings; a
  generation request cannot inject a command or executable path.

## Recovery

SQLite uses WAL and short transactions. Jobs left queued/running/stopping by an
unclean WebUI shutdown are marked `interrupted` on restart rather than silently
resumed. Existing completed artifacts remain visible in the library.
