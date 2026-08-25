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

## 512x512 ten-second EasyCache generation

A text-to-video-and-audio automotive commercial was generated with 243 frames
at 24 fps (10.125 seconds), 512x512 resolution, seed `20260825`, 20 scheduler
points, and EasyCache defaults (`0.2`, active from `0.15` through `0.95`). The
prompt embedding was read from a validated HAYATE prompt cache.

The upstream denoise loop made 19 cache-controller calls. Six Transformer
evaluations were skipped and 13 were computed, a theoretical DiT-only speedup
of `19 / 13 = 1.46x`. Denoising took 11 minutes 22 seconds and total generation,
including model loading, VAE decode, and MP4 encoding, took 874.75 seconds
(14 minutes 34.75 seconds).

Peak process working set was 16.49 GiB, peak private bytes 27.50 GiB, peak CUDA
allocation 5.54 GiB, and peak CUDA reservation 7.48 GiB. The output contains
H.264 512x512 video with exactly 243 frames and AAC 32 kHz stereo audio; both
streams are 10.125 seconds. FFmpeg reported no black/freeze interval and the
audio contains zero NaN/Inf samples. Decoded AAC peak is +0.67 dBFS, so a
production delivery should add a small true-peak limiter or attenuation pass.

Contact-sheet inspection shows a consistent silver sports car moving from a
coastal highway into a modern city setting. This run validates the optimized
execution path, not numerical equivalence to an uncached 20-point baseline.

The same fixed prompt and seed were repeated after correcting EasyCache's
full-to-full transformation-rate calibration, with threshold `0.4` and a hard
limit of two consecutive skips:

| EasyCache profile | Full DiT calls | Skipped | Denoise | Total |
|---|---:|---:|---:|---:|
| conservative (`0.2`) | 13 | 6 | 11m 22s | 874.75 s |
| RTX 3060 fast (`0.4`, max 2) | 10 | 9 | 8m 47s | 724.14 s |

The fast profile reduced total wall time by 150.62 seconds (17.2%) without
changing peak CUDA allocation/reservation. It produced the same 243-frame,
10.125-second video/audio contract. FFmpeg found no black/freeze interval or
NaN/Inf audio samples; contact-sheet inspection retained the silver car,
coastal highway, and city sequence. Framewise SSIM against the conservative
cached run was `0.863170`; this is a speed/quality comparison between two cached
runs, not an uncached-fidelity score.

## Windows non-mmap checkpoint loading

An intermittent Windows native access violation was observed in
`torch_cpu.dll` while constructing the large W4A8 checkpoint through
`safetensors.safe_open`. HAYATE now selects the safetensors 0.8 `pread` backend
on Windows and drains each owned tensor into its destination model. Linux keeps
the mmap path, and Windows comparison runs can opt back into it with
`HAYATE_SAFETENSORS_BACKEND=mmap`.

The same cached-prompt 256x256, 5-frame, 3-point generation was run with both
backends:

| Backend | Wall time | Peak process working set | Peak private bytes | Peak CUDA allocation |
|---|---:|---:|---:|---:|
| mmap | 55.3 s | 20.45 GB | 49.41 GB | 3.84 GB |
| pread | 66.4 s | 17.65 GB | 25.49 GB | 3.84 GB |

Both MP4 files have the identical SHA-256
`1c6030e75321df8982330847dcf603167717cd06ed15e5fd5be6972e427e101b`
and contain H.264 256x256 video plus AAC 32 kHz stereo audio. The stable path
cost 11.1 seconds in this short load-dominated run while reducing peak private
accounting by 23.92 GB. A standalone pread load also strict-loaded all 635 W4A8
Transformer tensors with zero meta tensors and all NVFP4/AWQ conditioner
weights with zero meta tensors.
