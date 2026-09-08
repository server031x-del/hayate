[HAYATE StudioをColabで開く](https://colab.research.google.com/github/server031x-del/hayate/blob/codex/colab-fast-h3/colab/HAYATE.ipynb) — GitHubから直接導入できます。手順は[colab/README.md](colab/README.md)。

# HAYATE

**High-speed AI Yield & Acceleration Technology Engine**

Goal: run large generative AI models efficiently on consumer hardware.

Initial target: MiniMax H3.

Reference validation host (the runtime is not limited to this hardware):

- NVIDIA GeForce RTX 3060 12 GB (v0.1 validation GPU)
- NVIDIA GeForce GTX 1660 SUPER 6 GB (detected; native H3 W4A8 is SM-gated)
- 32 GB system RAM
- Intel Core i7-9700

Status: **Experimental, end-to-end generation enabled**. v0.1 includes safe
model inspection, a pinned external upstream launcher, direct optimized loaders,
and consumer-GPU / host-RAM packed-weight streaming. A fixed-seed 50-step
video/audio generation has passed; broader resolutions and long clips remain
benchmark work.

## Relationship to maybleMyers/h3

HAYATE uses the MiniMax-H3 engine structure from
[`maybleMyers/h3`](https://github.com/maybleMyers/h3) as its principal technical
base. It does not reimplement the working MiniMax pipeline, Scheduler,
Transformer, Video VAE, or Audio VAE. HAYATE adds model-header validation,
consumer-GPU loading policy, stage ownership, memory telemetry, and benchmarks
around that engine.

The exact audited commit, `REUSE / MODIFY / REPLACE / ADD` decisions, and
license boundary are recorded in:

- [`docs/UPSTREAM_COMPONENT_CLASSIFICATION.md`](docs/UPSTREAM_COMPONENT_CLASSIFICATION.md)
- [`docs/UPSTREAM_LICENSE_AUDIT.md`](docs/UPSTREAM_LICENSE_AUDIT.md)
- [`docs/MODEL_HEADER_AUDIT.md`](docs/MODEL_HEADER_AUDIT.md)
- [`upstream/h3.lock.json`](upstream/h3.lock.json)

The audited h3 commit has no repository-level license. HAYATE v0.1 therefore
does not copy its license-unclear files. An external-checkout adapter validates
and imports the pinned engine at runtime without vendoring it. Files with explicit
Apache-2.0 headers can be vendored later with their notices retained.

## Quick start

Install [uv](https://docs.astral.sh/uv/), then from this directory:

On Windows, the one-click bootstrap creates the standard directories and
installs the generation/WebUI extras. It does not download model weights:

```powershell
.\SETUP_HAYATE.cmd
```

The same setup can be run from PowerShell with `.\scripts\setup_hayate.ps1`.
It also obtains the audited `maybleMyers/h3` commit into `upstream/h3` without
modifying an existing checkout. Use `-SkipSync` to skip dependency installation
or `-SkipUpstream` when an external h3 checkout will be configured. See
[`docs/SETUP.md`](docs/SETUP.md) for Linux/WSL and first-run details.

```powershell
uv sync
uv run hayate inspect
```

Equivalent module invocation:

```powershell
uv run python -m hayate inspect
```

### Google Colab WebUI

The hosted WebUI path is documented in [`docs/COLAB_WEBUI.md`](docs/COLAB_WEBUI.md).
Build the private-source upload bundle with:

```powershell
python scripts/create_colab_webui_bundle.py
```

Then run the WebUI cells in [`colab/HAYATE.ipynb`](colab/HAYATE.ipynb). The
notebook binds HAYATE Studio to the Colab VM and publishes it through a
temporary HTTPS tunnel; it does not start a service on the local PC.

For a CUDA-enabled full runtime environment, install the optional PyTorch
dependency. The uv configuration pins that extra to PyTorch's CUDA 12.8 wheel
index, which is compatible with this machine's newer NVIDIA driver:

```powershell
uv sync --extra cuda
```

Verify that the distributed W4A8 extension really executes on an eligible GPU
(an eager fallback is deliberately rejected). Use `--gpu auto`, a physical
index, or an NVIDIA UUID to choose the adapter:

```powershell
uv run hayate kernel-check --gpu auto
```

The inspection CLI itself does not require PyTorch and does not load model
payloads.

On Windows, HAYATE uses safetensors 0.8 `pread` loading by default for the
large direct-loader checkpoints. This avoids intermittent native access
violations observed at the mmap/Torch boundary and lowers committed virtual
memory. `HAYATE_SAFETENSORS_BACKEND=mmap` restores the faster legacy path for
controlled comparison runs; it is not recommended for normal Windows use.

## Configure models without copying them

Edit `configs/models.yaml`, or copy it to the ignored
`configs/models.local.yaml`. Paths are resolved relative to the YAML file and
may point directly at an existing ComfyUI model directory. ComfyUI is not
imported or required.

```yaml
models:
  minimax_h3:
    transformer:
      path: "D:/ComfyUI/models/diffusion_models/minimax_h3_fl2va_pruned_w4a8_mixed.safetensors"
      expected_quantization: W4A8_MIXED
    text_encoder:
      path: "D:/ComfyUI/models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
      expected_quantization: NVFP4_AWQ
    video_vae:
      path: "D:/ComfyUI/models/vae/minimax_h3_video_vae_int8_convrot.safetensors"
      expected_quantization: INT8_CONVROT
    audio_vae:
      path: "D:/ComfyUI/models/vae/minimax_h3_audio_vae_fp32.safetensors"
      expected_quantization: FP32
```

Run with an alternate registry:

```powershell
uv run hayate inspect --config configs/models.local.yaml --verbose
```

The optional Kijai FastH3/VSA artifact can be checked without loading its
22.9GB payload:

```powershell
uv run hayate fasth3-check `
  --checkpoint models/minimax_h3_fastvideo_vsa_datafree_1300step_4step_int8_convrot.safetensors `
  --model-dir models/fastvideo
```

This command reports the VSA gate/layout, FastVideo runtime availability, and
the exact blockers. The Kijai file is a ComfyUI single-file conversion and is
checked for provenance/integrity only; HAYATE execution uses the separate
official [FastVideo FastH3 Preview v1 directory snapshot](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree)
(the older v0.2 snapshot remains compatible with the directory validator). This is a compatibility guard,
not a failed normal H3 setup.

After the optional FastVideo runtime and directory snapshot pass preflight, the
same route can be selected explicitly from the CLI:

```powershell
uv run hayate generate `
  --fasth3 `
  --python wsl://Ubuntu/home/<WSLユーザー>/hayate-fasth3-venv/bin/python `
  --ckpt-dir models/fastvideo `
  --prompt "A cinematic car commercial" `
  --output outputs/fasth3.mp4
```

Blackwellで最速プロファイルを明示する場合は、先に条件を診断してから
`fasth3-check --performance-profile fast`を通し、WebUIの
**FastH3 Blackwell最速**を選択します。CLIの`--fasth3`は安全なstrict経路です。

The adapter currently targets T2VA only (five sigma points / four DiT
forwards), uses the Triton VSA path on consumer GPUs, and never falls back to
dense attention when a VSA requirement is missing. The WebUI also exposes a
separate Blackwell-only speed profile that requires sm100a VSA and FA4 and
enables the official `profile=all` fusions, regional compile, and parallel VAE
decode; it remains disabled on other GPUs.

For the Kijai single-file VSA route, use the separate shared-model ComfyUI
runtime described in [`docs/COMFYUI_HAYATE.md`](docs/COMFYUI_HAYATE.md). It is
installed on M:, points at the existing HAYATE model folders, and keeps
generated output in `M:/Project/HAYATE/outputs`; no model copy is made under the
ComfyUI directory.

速度を優先する場合のSageAttention/Triton構成（モデルを共有し、重みを複製しない）は
[`docs/COMFYUI_HAYATE.md`](docs/COMFYUI_HAYATE.md) の **Turbo構成** を参照してください。

For a reproducible Windows/WSL installation, use
[`docs/FASTH3_WSL.md`](docs/FASTH3_WSL.md). It creates an isolated FastVideo
environment and provides an explicit, resumable download command for the
official approximately 148 GB (about 138 GiB) snapshot. The normal HAYATE bootstrap never
starts that download automatically.

Preflight the actual upstream generation command without loading tensors:

```powershell
uv run hayate generate `
  --upstream M:/path/to/maybleMyers-h3 `
  --ckpt-dir M:/path/to/MiniMax-H3-snapshot `
  --config configs/models.local.yaml `
  --prompt "A cinematic scene" `
  --output outputs/hayate.mp4 `
  --dry-run
```

HAYATE passes the four single-file overrides to upstream
`minimax_generate_video.py`. Its conservative fidelity profile keeps SDPA and
the upstream 50-point sigma grid, swaps 49 transformer blocks, streams the text
encoder with zero resident decoder layers, chunks activations, and enables VAE
tiling. Remove `--dry-run` only after preflight reports `READY`.

For a faster 20-point run, HAYATE can wrap the upstream Transformer with an
Apache-2.0 EasyCache-derived runtime-adaptive residual cache while leaving the
upstream pipeline, schedulers, and VAE stages intact:

```powershell
uv run hayate generate `
  --upstream M:/path/to/maybleMyers-h3 `
  --ckpt-dir M:/path/to/MiniMax-H3-snapshot `
  --config configs/models.local.yaml `
  --prompt "A cinematic scene" `
  --output outputs/hayate-fast.mp4 `
  --fast
```

`--fast`, `--fast-sage`, `--fast-sage-detail`, `--pdd`, and `--pdd-sage` are
the hardware-neutral profile names. The older `--rtx3060-*` spellings remain
accepted as compatibility aliases for existing scripts.

The conservative defaults match the common `0.2` threshold and `0.15`–`0.95`
sampling window. The fast profile above raises the HAYATE integration
threshold to `0.4`, but forces a real Transformer evaluation after at most two
cached calls. EasyCache is opt-in because skipped Transformer evaluations trade
a small amount of numerical fidelity for speed.

For the separately released PDD Acc 8-Step trajectory, place the FL2VA PDD
checkpoint and AdaLN affine map under `models/lora/`, then select `PDD 8-Step`
in Studio or use the dedicated CLI profile:

```powershell
uv run --no-sync hayate generate `
  --upstream M:/path/to/maybleMyers-h3 `
  --ckpt-dir M:/path/to/MiniMax-H3-snapshot `
  --config configs/models.local.yaml `
  --prompt "A cinematic scene" `
  --output outputs/hayate-pdd.mp4 `
  --pdd
```

PDD is an alternative to EasyCache, not an additional cache layer. HAYATE
rejects the combination. The validated PDD profile uses SDPA; SageAttention
remains available as an explicit experimental switch, but is fail-closed below
243 frames on the validated consumer-GPU path after short-clip non-finite latent
detection.
The released 2688-wide AdaLN adapters are projected onto the validated Comfy-Org
pruned 8-wide coordinates without modifying the W4A8 base. See
[`docs/PDD_ACCELERATION.md`](docs/PDD_ACCELERATION.md).
PDD adapter pages remain pageable by default; set `HAYATE_PDD_PIN_LORA=1`
explicitly only after measuring a host.

On the reference Windows consumer GPU, the validated approximate-attention profile
cut the same fixed-seed 512x512, 243-frame run from 12m04s to 6m59s while
retaining the safe 256-pixel VAE tiling geometry:

```powershell
uv run --no-sync hayate generate `
  --upstream M:/path/to/maybleMyers-h3 `
  --ckpt-dir M:/path/to/MiniMax-H3-snapshot `
  --config configs/models.local.yaml `
  --prompt "A cinematic scene" `
  --output outputs/hayate-fast-sage.mp4 `
  --fast-sage
```

This profile requires a Windows-compatible SageAttention 2.2 build and is
deliberately separate from `--fast`: SageAttention quantizes attention
internals and is therefore not numerically identical to SDPA. Use SDPA for the
fidelity reference. VAE tile sizes above the released 256-pixel geometry remain
experimental; a 512-pixel tile was faster but failed the fixed-seed visual gate.
`--no-sync` preserves the separately installed platform-specific wheel; see
[`docs/SAGEATTENTION_WINDOWS.md`](docs/SAGEATTENTION_WINDOWS.md).

`--verbose` prints every tensor name, shape, dtype, and physical storage size.
`--json` emits a machine-readable report. Benchmark JSON is saved under
`benchmarks/` unless `--no-save-benchmark` is used.

## HAYATE Studio WebUI

HAYATE Studio is a local, ComfyUI-independent generation interface. It keeps
the audited MiniMax H3 execution path and the four single-file model overrides;
the browser is an operator surface over the same `GenerationRequest` preflight.

Install the optional server dependencies once:

```powershell
uv sync --extra generation --extra webui
```

An exact `uv sync` removes platform-specific packages that are not in the lock
file. If the Fast Sage profile is required, reinstall the validated
SageAttention/Triton Windows wheels as described in
[`docs/SAGEATTENTION_WINDOWS.md`](docs/SAGEATTENTION_WINDOWS.md), then start
without syncing again:

```powershell
.\START_HAYATE_WEBUI.cmd
```

Or start it directly:

```powershell
.venv\Scripts\hayate.exe webui --open-browser
```

The default bind is `0.0.0.0:7860`, so open `http://127.0.0.1:7860` on the
same PC or use the machine's LAN address from another trusted device. Use
`--local-only` (or `--host 127.0.0.1`) when the interface must stay loopback-only.
The first launch discovers existing MP4 outputs and their HAYATE manifests.
Runtime paths can then be reviewed and saved from the Settings screen.

The Settings screen also has a model setup panel. It creates the standard
`models/` directories, scans the four H3 single-file targets plus optional
acceleration assets, and exposes download buttons only for pinned, verified
Hugging Face sources. A license acknowledgement is required for each download;
weights are never copied into Git or silently overwritten. The catalog includes
Kijai's `minimax_h3_fastvideo_vsa_datafree_1300step_4step_int8_convrot.safetensors`
as an explicitly experimental FastH3/VSA artifact. It is kept separate from
the mayble H3 W4A8 path: the single-file ComfyUI conversion cannot be passed to
FastVideo's directory loader, so Studio provides a header-only **FastH3利用条件
診断** and fails closed when the optional FastVideo runtime/model directory is
not configured. The normal H3 profiles remain unchanged.

Studio includes:

- unchanged Fast Sage (`最速`), a Fast Sage Detail (`高速・画質優先`)
  profile with one extra late refinement, Fast SDPA, Quality, and fully custom
  generation profiles; plus an explicitly experimental FastH3 VSA 4-step
  profile and a Blackwell-only FastH3 v1 speed profile that are disabled until
  the official FastVideo directory/runtime preflight passes (the Kijai single
  file remains external-ComfyUI-only);
- readable resolution presets from lightweight 512px social formats through
  16:9 HD, grouped by light and high-detail memory tiers, plus a 32-pixel
  custom mode;
- T2V/I2V image upload, duration snapping to MiniMax H3 frame geometry, prompt
  cache, EasyCache, block swap, activation chunking, and VAE controls;
- an explicit MiniMax H3 prompt assistant that previews and applies structured
  visual/action/camera, sound, and music fields without silently rewriting the
  user's prompt, plus optional OpenAI Responses API authoring with a
  configurable model;
- physical-GPU selection by stable NVIDIA UUID, UUID-scoped leases shared with
  the CLI, and optional one-process-per-GPU WebUI workers (safe default: 1);
- structured `HAYATE_EVENT` progress, stage timeline, ETA, VRAM/RAM status,
  persisted job logs, safe stop-and-save, and immediate cancellation;
- persistent SQLite history, existing-output import, search, video previews,
  exact settings, duration, and peak VRAM statistics;
- a readable responsive desktop/mobile interface with persistent Dark/Clear
  display modes and no Node.js requirement at runtime.

The default binds to `0.0.0.0` for trusted-LAN access and there is no
authentication layer. The included launcher permits the detected private
OpenVPN range `10.8.0.0/24` to register the OpenAI key and use AI prompt
authoring; change its `--trusted-client-network` value if your VPN CIDR differs.
Use `--local-only` on an untrusted network; do not expose the interface to the
public internet. Uploaded images, prompts,
job history, settings, and the SQLite database remain under `data/webui/`; model
weights are referenced in place and are not copied. The OpenAI API key is not
stored in that directory: the Settings password field uses Windows Credential
Manager through `keyring` when available, with an in-process-only fallback.

## v0.1 capabilities

- Windows/Linux/WSL-friendly hardware profiling with all detected NVIDIA GPUs
  visible in the UI; H3 W4A8 auto/manual assignment is gated by SM 8.0+ while
  GPU 0 remains the primary display marker for backward compatibility.
- Bounded safetensors JSON-header parsing without mapping tensor payloads.
- Evidence-based detection of W4A8 mixed, NVFP4+AWQ, INT8 ConvRot, FP32, FP16,
  BF16, and FP8. Ambiguous layouts remain `UNKNOWN`.
- YAML model registry and direct external model paths.
- `BaseModelLoader` plus dedicated W4A8, NVFP4/AWQ, INT8 ConvRot, standard,
  and unknown-safe loaders.
- RAM and sampled total-VRAM snapshots/peaks.
- Four-stage resource ownership matching upstream conditioner, transformer,
  VAE decode, and output encode ordering.
- Stage timing and JSON benchmark reports.
- A non-importing adapter for the pinned upstream MiniMax-H3 engine contract.
- Prompt-cache fingerprints covering the upstream commit, text-encoder file,
  and image/reference contents, with crash-safe atomic cache replacement.
- Opt-in MiniMax-H3 EasyCache with video/audio residuals, caller-owned tensor
  preservation, configurable threshold/window, and runtime skip telemetry.
- MiniMax-H3 PDD Acc 8-Step as a switchable EasyCache-exclusive mode, including
  strict 728-tensor validation, pruned-AdaLN affine projection, and runtime NFE
  telemetry; SDPA is the validated profile and SageAttention is experimental.
- Windows-native `pread` loading for large safetensors checkpoints, avoiding
  intermittent `torch_cpu.dll` access violations at the mmap boundary.
- Header-only FastH3/VSA inspection (`hayate fasth3-check`) with explicit
  `to_gate_compress`/INT8/ComfyUI-layout evidence; no silent fallback to the
  standard W4A8 loader.

## Current execution support

| Format | Header inspection | Loader routing | Direct optimized execution |
|---|---:|---:|---:|
| W4A8 mixed | Yes, with explicit evidence | Yes | Yes; 200 packed layers, staged strict load, packed 49-block streaming |
| NVFP4 + AWQ | Yes, with explicit evidence | Yes | Yes; mmap staged load, 350 NVFP4 layers + INT8 embedding, Ampere BF16 fallback |
| INT8 ConvRot | Yes, marker convention | Yes | Yes; 144 Video VAE groups, strict load and CUDA decode |
| FP32 / FP16 / BF16 | Yes | Yes | Raw safetensors materialization when PyTorch is installed |
| FP8 | Yes | Yes | Hardware/kernel-specific integration pending |
| Unknown | Yes | Safe blocked route | No |
| FastH3/VSA single-file | Yes; VSA gate and fused ComfyUI layout | Diagnostic only; external ComfyUI/VSA | Not a HAYATE input. HAYATE can launch the separate official FastVideo directory with its optional runtime; no dense fallback |

The exact public W4A8 header contract is now audited: 200 packed
`asym_w4a8_int8` layers, group size 16, ConvRot group size 256, FP8 relative
scales, FP32 channel scales, and 16-value codebooks. RTX 3060 `sm_86` meets the
maintained layout's SM80 floor. `hayate kernel-check` verifies the actual
Windows wheel and GPU operation before the loader can be enabled. See
[`docs/W4A8_RESEARCH.md`](docs/W4A8_RESEARCH.md).

## Reference-host end-to-end result

On 2026-08-25, the four target single files generated a 256x256 T2VA sample on
the RTX 3060 with seed `20260825`, SDPA, 49 swapped blocks, and 50 scheduler
points. Total wall time was `233.1 s`; 49 denoise calls stabilized around
`2.62-2.68 s/step`. The packed offloader kept one block resident and streamed
49 blocks through a `0.411 GiB` GPU ring. A repeated smoke run measured `20.20
GiB` peak process working set and `3.58 GiB` peak CUDA allocation. The output
contains four distinct H.264 frames plus finite AAC stereo audio and is not black. See
[`docs/EXECUTION_VALIDATION.md`](docs/EXECUTION_VALIDATION.md).

## Tests

```powershell
uv run pytest
```

Tests use tiny generated safetensors fixtures; CI does not need large model
files or a CUDA GPU.

## Optimization rule

An optimization is accepted only with recorded before/after execution time,
VRAM, RAM, and quality impact under identical seed and settings. The v0.1
inspection benchmark marks quality as not measured because no generation is
performed.

## License

HAYATE source code is licensed under Apache-2.0. Model weights and referenced
upstream projects keep their own terms; see `THIRD_PARTY_NOTICES.md`.
