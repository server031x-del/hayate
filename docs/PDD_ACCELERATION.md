# MiniMax-H3 PDD Acc 8-Step integration

HAYATE supports the FL2VA checkpoint from
[`alibaba-pai/MiniMax-H3-Acc-LoRAs`](https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs)
as a separate acceleration mode. It does not merge the adapter into the W4A8
backbone. The 362 low-rank branches remain explicit, and the video/audio output
heads select four of the released 32 directions for each of eight Transformer
evaluations.

## Runtime contract

- checkpoint metadata: grid 32, block size 4, rank 64, alpha 64;
- 362 paired LoRA modules: 50 AdaLN and 312 attention/feed-forward/refiner;
- two parallel output heads, each with 32 directions;
- 8 model evaluations, represented as 9 scheduler points because the upstream
  MiniMax scheduler includes the terminal zero;
- EasyCache is rejected when PDD is active. The validated PDD profile uses
  SDPA; SageAttention remains selectable as an experimental switch. On the
  RTX 3060 path HAYATE fails closed for PDD+SageAttention below 243 frames
  because short-clip runs produced non-finite video and audio latents;
- the existing non-PDD Quality, Fast SDPA, Fast Sage, and Fast Sage Detail
  paths are unchanged.

The released PDD file targets the original 2688-wide AdaLN input. HAYATE's
current W4A8 checkpoint uses the Comfy-Org pruned 8-wide coordinate table.
HAYATE therefore applies the affine identity documented by
[`multimodalart/MiniMax-H3-Pruned`](https://huggingface.co/multimodalart/MiniMax-H3-Pruned):

```text
x = mean + c @ basis
A' = A @ basis.T
offset = B @ (A @ mean)
```

`A'` and the constant offset are computed in float64, then retained in float32.
The base W4A8 weight is neither dequantized nor modified. The affine map is
accepted only when the transformer's raw `adaln_t_table` SHA-256 is
`ac8727cdec52137c73878d004de5bd2a0e19227e8311e29ab3b68f328310e34e`.
Unknown coordinate gauges fail closed.

## Files

The local WebUI defaults are:

```text
models/lora/MiniMax-H3-FL2VA-Acc-8Step.safetensors
models/lora/adaln_affine.safetensors
```

PDD LoRA tensors stay pageable by default. Set `HAYATE_PDD_PIN_LORA=1` only
after measuring the host; it is an optional transfer optimization and is not
required for correctness.

The validated FL2VA PDD file is 1,372,450,680 bytes with SHA-256
`0b29be7042d883970eb0c20774a9ba03d95669ed80a721bb4d21be8ea0d0a196`.
The affine map is 96,960 bytes with SHA-256
`34f285e7aeae741666868bf5912506ab4793098357a01a9d8358f6ce7704d532`.

## CLI

```powershell
uv run --no-sync hayate generate `
  --upstream M:/path/to/maybleMyers-h3 `
  --ckpt-dir M:/path/to/MiniMax-H3-snapshot `
  --config configs/models.local.yaml `
  --prompt "A cinematic scene" `
  --output outputs/hayate-pdd.mp4 `
  --rtx3060-pdd
```

For a custom invocation, pass `--pdd-checkpoint` and, when the transformer is
pruned, `--pdd-adaln-affine`. Do not combine `--easycache` with PDD. The
experimental `--rtx3060-pdd-sage` profile is intended for validated long clips.

## Provenance and licenses

The PDD repository declares Apache-2.0 in its model-card metadata, although a
top-level `LICENSE` file was not present when inspected on 2026-08-27. HAYATE
implements the published schedule and checkpoint contract independently and
does not copy its helper module. The affine tensor and all MiniMax-derived model
weights remain governed by their source repositories and the MiniMax-H3
Community License Agreement; HAYATE does not redistribute them.
