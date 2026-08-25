# HAYATE Studio architecture and operations

HAYATE Studio is a local control surface. It does not implement a second model
pipeline. Every submitted request is converted to `GenerationRequest`, checked
by `ExternalH3GenerationBackend.plan()`, and launched through the audited
maybleMyers/h3 entrypoint.

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
