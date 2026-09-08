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
When using the included launcher, the detected OpenVPN network
`10.8.0.0/24` is allowed for these two operations. If your VPN uses another
client CIDR, replace the `--trusted-client-network` value in
`START_HAYATE_WEBUI.cmd`; the value must match the source address seen by this
server. Additional networks can be supplied by repeating the option.

## Model setup

Settings includes a **MiniMax H3モデル** panel. **標準フォルダを準備** creates
`models/minimax-h3-snapshot`, `models/text_encoders`, `models/vae`,
`models/lora`, and the optional `models/fastvideo` directory without touching
existing files. **標準パスを適用** is a separate,
explicit action that points the model registry, support snapshot, and PDD paths
at those folders while preserving the configured upstream checkout, output, and
Python paths.

The catalog is an allowlist of the audited W4A8 transformer, NVFP4/AWQ text
encoder, INT8 ConvRot Video VAE, FP32 Audio VAE, PDD Acc LoRA, AdaLN affine map,
the small upstream support-file set, and the optional Kijai FastH3 VSA artifact.
Each entry pins a Hugging Face commit, expected size, and SHA-256. The operator
must acknowledge the model terms before a download is accepted. Downloads run
outside the generation queue, use an in-volume temporary directory, verify
before atomic placement, and never overwrite a non-matching file. An interrupted
or failed transfer can be safely re-run after its temporary directory is cleaned
on the next startup; byte-range resume is not promised.

The Kijai FastH3 file is marked **実験** in the catalog. Its header is checked
for the `to_gate_compress` VSA gate and INT8/ComfyUI fused layout, but it is not
sent to the normal mayble H3 W4A8 loader and is not a HAYATE generation input.
It is retained for provenance, integrity checks, and an external ComfyUI/VSA
workflow. Use **利用条件を診断** after placing the file to inspect the optional
runtime and the separate FastVideo snapshot. HAYATE generation requires the
official FastVideo directory (the pinned v1 snapshot is
[`FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree`](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree);
`model_index.json`, `modular_model_index.json`, or `fastvideo_inference.json`,
plus `transformer/config.json` and transformer weight files); the Kijai single
file cannot be passed to that directory loader.
HAYATE therefore fails closed rather than silently ignoring the VSA gate. The
current validated FastH3 preview contract is T2VA; I2V/FL2VA is not enabled by
this catalog entry. The WebUI exposes a strict Triton profile for supported
CUDA GPUs and a separate **FastH3 Blackwell最速** profile. The latter is enabled
only when the diagnostic sees a Blackwell compute capability (10.0/10.3), the
sm100a VSA kernel, and `flash_attn.cute` (FA4); it enables the official
`profile=all` fusions, tile-64 sm100a VSA, regional DiT compile, and parallel
VAE decode.

For Windows, the reproducible setup helper is
[`scripts/setup_fasth3_wsl.ps1`](../scripts/setup_fasth3_wsl.ps1). It installs an
isolated FastVideo 0.2.1 / PyTorch CUDA 13.0 environment and pins the official
snapshot revision `5ea076f35b84da4c3c82217112fa733d8eea2ae1`. The approximately
148 GB (about 138 GiB) model download is opt-in (`-DownloadModel`) and is
refused when the target volume has less than 160 GiB free. See [`FASTH3_WSL.md`](FASTH3_WSL.md)
for the complete command and WSL path mapping.

The generation screen groups resolution presets by light formats and
higher-detail RAM/VRAM tiers. Custom width and height remain limited to
32-pixel multiples and the server validates the same constraint.

Settings also exposes optional **FastVideo model directory** and **FastVideo
Python** fields. They are independent of the standard H3 settings and do not
change the running process. The default directory is `models/fastvideo`; save
the path only when an official FastVideo snapshot and its isolated Python
environment are available. No FastVideo package or 22.9GB Kijai single-file
weight is installed automatically.

For the Kijai VSA single-file path, the separate `M:\Project\HAYATE-ComfyUI`
runtime can be used without duplicating HAYATE's model files. Its
`extra_model_paths.yaml` points to `M:\Project\HAYATE\models`, and
`scripts/start_comfyui_hayate.ps1` keeps ComfyUI output in HAYATE's shared
`outputs` directory. See [`COMFYUI_HAYATE.md`](COMFYUI_HAYATE.md).

## Multi-GPU scheduling

The H3 engine remains one process on one CUDA device. HAYATE discovers physical
NVIDIA UUIDs, exposes them in **実行GPU**, and launches the selected child with
`CUDA_VISIBLE_DEVICES=<UUID>` while passing upstream `--device cuda:0`. This
avoids confusing physical `nvidia-smi` indices with PyTorch visible ordinals and
does not pretend to pool VRAM across adapters.

`Auto` tries every allowed SM 8.0+ adapter with a stable UUID and uses a
UUID-scoped scheduler lease. A driver report without UUID can still be targeted
by explicit physical index, but is excluded from automatic multi-GPU routing.
The Settings screen can start up to eight WebUI workers, one per GPU; the default is
one worker because H3 CPU offload shares host RAM, PCIe, and storage bandwidth.
The worker-count setting is persisted immediately and takes effect on the next
WebUI start; the current process is never resized underneath an active job.
Explicit GPU jobs that find their adapter busy are requeued so a later job can
use another free adapter. The chosen GPU and UUID are retained in runtime
metrics and lease owner metadata. Existing `CUDA_VISIBLE_DEVICES` values are
treated as an allow-list and never widened.

The parent keeps a scheduler reservation and the H3 child takes a separate
runtime lease. If the parent exits unexpectedly while the child is still using
CUDA, the child lock continues to protect that physical adapter.

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
  -> GPU-aware JobManager (safe default: one worker)
  -> UUID-scoped scheduler/runtime leases
  -> HAYATE MiniMax H3 entrypoint
  -> maybleMyers/h3 generation pipeline
```

When either FastH3 profile is explicitly selected and its preflight passes, the
last two stages instead become:

```text
GPU-aware JobManager -> HAYATE FastH3 launcher -> operator-provided FastVideo VSA runtime
```

This alternate branch uses the official FastVideo directory snapshot (not the
Kijai ComfyUI single file), is T2VA-only in the current adapter, and is never
chosen implicitly by the normal H3 profiles. The profiles stay disabled until
the WebUI diagnostic confirms the external runtime and directory contract. The
Blackwell profile is additionally blocked unless its sm100a VSA and FA4 gates
pass; otherwise select the strict profile.

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

WebUI, `hayate generate`, `kernel-check`, and `load-check` use the same
UUID-scoped file-lock namespace. This prevents a CLI diagnostic from entering
CUDA while a queued WebUI generation owns the selected physical adapter.

## Local security boundary

- Default bind: `0.0.0.0:7860` for trusted-LAN access; the interface remains
  unauthenticated. Use `--local-only` or `--host 127.0.0.1` for loopback-only
  operation.
- OpenAI credential mutation and `/api/prompt-assistant` accept loopback and
  only the explicitly configured `--trusted-client-network` CIDRs when the
  server uses a non-loopback bind. The included launcher permits the private
  OpenVPN range `10.8.0.0/24`; use a real authenticated, encrypted reverse
  proxy before exposing a paid API credential to any other network.
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
