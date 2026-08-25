# RTX 3060 / 32 GB execution validation

Validation date: 2026-08-25.

## Reference environment

- GPU: NVIDIA GeForce RTX 3060 12 GB, `sm_86` (GPU 0 only)
- Secondary GPU: GTX 1660 SUPER detected, not scheduled
- RAM: 31.90 GiB
- PyTorch: 2.11.0+cu128
- comfy-kitchen: 0.2.31
- upstream h3: `94220c1fdf14d6d9d40be06fb99f55b27c0d9024`

## Full component loads

| Component | Wall time | Process RSS | Result |
|---|---:|---:|---|
| W4A8 Transformer | 19.41 s | 13.45 GB | 635 tensors, 300 quantized wrappers, meta 0 |
| NVFP4/AWQ Conditioner | 4.37 s | 5.31 GB | text 351 quantized, vision 352 tensors, meta 0 |
| INT8 ConvRot Video VAE | 5.65 s | 4.26 GB | strict key match; 216 post-conversion wrappers, meta 0 |
| FP32 Audio VAE | 3.67 s | 1.29 GB | 172 weight norms merged, strict key match, meta 0 |

The W4 loader assigns tensors directly into a meta-initialized upstream model;
it does not retain a second 12.54 GB state dictionary. The Qwen loader keeps the
safetensors mmap owner alive with the conditioner, avoiding a second 15.69 GB
clone on Windows.

## End-to-end fixed-seed generation

Shared settings:

```text
prompt: A silver paper airplane glides through a quiet sunlit library,
        cinematic, stable camera, detailed natural light.
seed: 20260825
task: T2VA
size: 256x256
requested frames: 5 (encoded output: 4 frames at 24 fps)
attention: SDPA
block swap: 49 of 50
text encoder: 0 resident decoder layers, CUDA streaming
VAE tiling: enabled
```

| Run | Scheduler points | Denoise calls | Denoise time | Total wall time | Result |
|---|---:|---:|---:|---:|---|
| integration smoke | 3 | 2 | 6.2 s | 108.7 s | valid H.264 + AAC, visibly under-denoised |
| quality | 50 | 49 | about 129 s | 233.1 s | prompt-aligned paper airplane/library scene |

A repeated 3-point integration run with wrapper telemetry recorded peak process
working set `21,689,942,016` bytes (`20.20 GiB`), CUDA peak allocated
`3,841,542,144` bytes (`3.58 GiB`), and CUDA peak reserved `3,942,645,760`
bytes (`3.67 GiB`). Windows peak private/pagefile accounting reached about
`46.72 GiB`, so this result fits 32 GB physical RAM but still assumes an enabled
system page file. Future NVMe/pagefile-independent staging remains open work.

The quality output has four distinct decoded frame MD5 values. Audio is AAC
stereo at 32 kHz with overall RMS `-23.73 dB`, peak `-14.15 dB`, and zero
NaN/Inf samples. Visual inspection confirms a non-black result with a stable
paper airplane and library background.

The packed block streamer uses one resident block, 49 streamed blocks, about
`0.205 GiB` physical storage per block, and a `0.411 GiB` double-buffer ring.
This replaces upstream's plain-tensor staging only for packed tensor subclasses;
the transformer forward order and denoise pipeline remain upstream-owned.

These two runs do not establish quantized-versus-unquantized quality
equivalence because matching unquantized target weights were not available on
the reference host. Any claimed optimization comparison must keep seed, prompt,
resolution, frame count, scheduler, and model weights identical.
