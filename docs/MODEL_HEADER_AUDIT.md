# Target model header audit

Audit date: 2026-08-25. HAYATE first audited the public files with bounded HTTP
Range header reads, then downloaded the four named payloads for strict loading,
real-layer kernels, decode, and end-to-end generation.

| Role | Public source used for header audit | Header bytes | Tensors | Dtype counts |
|---|---|---:|---:|---|
| FL2VA W4A8 transformer | `starsfriday/MiniMax-H3-w4a8` and `Kijai/MiniMax-H3-experimental` | 143,408 | 1,132 | F32 410, BF16 220, F16 102, F8_E4M3 200, I8 200 |
| Qwen3-VL NVFP4/AWQ | `Comfy-Org/MiniMax-H3/text_encoders` | 231,400 | 2,054 | F32 351, BF16 651, F8_E4M3 350, I8 1, U8 701 |
| Video VAE INT8 ConvRot | `Kijai/MiniMax-H3-experimental` | 99,672 | 850 | F32 562, I8 144, U8 144 |
| Audio VAE FP32 | `Comfy-Org/MiniMax-H3/vae` | 105,520 | 917 | F32 917 |
| FastH3 VSA DataFree 4-Step INT8 ConvRot | `Kijai/MiniMax-H3-experimental` | 115,120 | 1,082 | INT8/FP32/U8 mixed; 50 VSA gate groups |

## Qwen3-VL NVFP4/AWQ evidence

The file has no top-level metadata. HAYATE therefore uses a strict compound
header signature rather than its filename:

- 350 packed `U8` module weights;
- 350 `F8_E4M3` `.weight_scale` tensors;
- 350 scalar `F32` `.weight_scale_2` tensors;
- 351 small `.comfy_quant` descriptor tensors;
- 100 BF16 `.pre_quant_scale` tensors identifying the AWQ pre-scaling path;
- one separately quantized INT8 embedding plus its scale/descriptor.

Range-reading the descriptor payloads confirms that the packed layers declare:

```json
{"format": "nvfp4", "full_precision_matrix_mult": true}
```

The embedding declares `{"format": "int8_tensorwise"}`. The first public
NVFP4 K-projection bundle was materialized directly and executed on RTX 3060.
Because Ampere has no native NVFP4 tensor cores, the audited path dequantizes
that layer for BF16 matmul; it produced finite output exactly matching explicit
dequantization. HAYATE retains packed storage using a conditioner-lifetime
Windows mmap and streams one layer at a time. The full override strict-loaded
in `4.37 s` with `5.31 GB` RSS, 351 quantized text tensors, 352 vision
parameters/buffers, and no meta tensors.

At least eight complete packed-weight/primary-scale/secondary-scale/descriptor
groups plus an AWQ pre-quant scale are required for HAYATE to select
`NVFP4AWQLoader`. A filename match cannot trigger it.

## Video VAE INT8 ConvRot evidence

The header contains 144 groups of:

```text
<prefix>.weight       I8
<prefix>.weight_scale F32
<prefix>.comfy_quant  U8
```

Their descriptor payload is exactly:

```json
{"format": "int8_tensorwise", "convrot": true, "convrot_groupsize": 256}
```

All 144 quantized weights are 2D transformer `Linear` weights inside the VAE;
the convolution weights remain FP32. The remaining 562 tensors are FP32. The model header also carries the MiniMax
Video VAE latent mean/std/config metadata. HAYATE enables execution through its
exact 144-module Linear contract and binds these layers directly to
comfy-kitchen's native INT8 Linear path;
a public `decoder.transformer_blocks.0.attn.to_out` tensor executed on RTX 3060
with finite output and relative L2 `0.008678` versus explicit dequantization
(seed `20260825`, 50 timed calls, `0.860 ms` mean for one input row).
The full model strict-loaded with 216 post-conversion quantized wrappers and no
meta tensors. A fixed random latent decoded on CUDA to `[1,3,5,128,128]` in
`1.042 s`, with finite output, standard deviation `0.7503`, and peak allocated
VRAM `3.05 GiB`; the result was not black.

## Audio VAE FP32 evidence

All 917 tensors use F32. The public single file contains merged `.weight`
parameters, while upstream constructs 172 legacy weight-normalized modules.
HAYATE removes those parametrizations before strict assignment; the full model
loaded in `3.67 s`, at `1.29 GB` RSS, with no meta tensors.

## FastH3 VSA evidence and boundary

The current Kijai artifact is pinned in the WebUI catalog at revision
`f4cac997f880e93cf6940af61ee8d58ef31ff7f7`, size `22,898,594,920` bytes, and
SHA-256
`7221ae65d78780354d51e5048d29728d9f1f8fb9baf50b1dd3df85f5101413d`. Its
header has the ComfyUI fused `blocks.*.attn.qkv_proj` / `out_proj` layout,
`to_gate_compress` VSA-gate weights, INT8 weights, and `.comfy_quant` markers.
There is no top-level metadata that would make it a normal HAYATE W4A8 file.

HAYATE therefore exposes header-only inspection and an explicit FastH3
preflight, but never routes this file through `W4A8Loader` or silently disables
the gate. The linked file is a single-file ComfyUI conversion; the official
FastVideo component loader expects a directory with model/config files. HAYATE
execution uses the separate
[`FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree`](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree)
snapshot (the older `FastVideo-Minimax-FastH3-Preview-v0.2` directory remains
compatible) when its external runtime is configured. The current preview
contract is T2VA; no I2V/FL2VA quality claim is made.

## Sources and model terms

- https://huggingface.co/starsfriday/MiniMax-H3-w4a8
- https://huggingface.co/Kijai/MiniMax-H3-experimental
- https://huggingface.co/Comfy-Org/MiniMax-H3

The model repositories identify the weight license as the MiniMax-H3 Community
License Agreement. HAYATE does not redistribute these weights.
