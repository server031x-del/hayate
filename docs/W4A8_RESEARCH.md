# W4A8 direct-loader research record

The target filename alone is never treated as proof of a quantization layout.
The layout findings began with HTTP Range retrieval of the safetensors header.
The 12.54 GB payload was subsequently downloaded for full strict-load and
end-to-end execution tests.

## Inspected artifacts

- `starsfriday/MiniMax-H3-w4a8` FL2VA file
- `Kijai/MiniMax-H3-experimental` FL2VA file
- Filename: `minimax_h3_fl2va_pruned_w4a8_mixed.safetensors`
- Exact file size: `12,540,858,008` bytes
- Header length: `143,408` bytes
- Tensor descriptors: `1,132`

Both inspected headers have the same tensor keys, dtypes, shapes, offsets, and
semantic quantization metadata. Their nested JSON property order differs, so
the raw header bytes are not identical. HAYATE validates the contract rather
than accepting a filename or hard-coded file hash.

Sources:

- https://huggingface.co/starsfriday/MiniMax-H3-w4a8
- https://huggingface.co/Kijai/MiniMax-H3-experimental/blob/main/minimax_h3_fl2va_pruned_w4a8_mixed.safetensors
- https://github.com/starsFriday/ComfyUI-W4A8-Loader

The weights use the MiniMax-H3 Community License. The reference loader has no
top-level license in the inspected repository view, so HAYATE does not copy it.

## Confirmed safetensors contract

The top-level `_quantization_metadata` string contains `200` layer entries:
four weights in each of `50` transformer blocks.

```text
blocks.N.attn.qkv_proj
blocks.N.attn.out_proj
blocks.N.mlp.fc1
blocks.N.mlp.fc2
```

Every entry declares:

```json
{
  "format": "asym_w4a8_int8",
  "group_size": 16,
  "convrot": true,
  "convrot_groupsize": 256
}
```

Each quantized prefix has this header layout:

| Tensor suffix | Storage dtype | Shape contract | Meaning |
|---|---|---|---|
| `.weight` | `I8` | `[N, padded_K / 2]` | two packed 4-bit codebook indices per byte after ConvRot padding |
| `.weight_s_rel` | `F8_E4M3` | `[N, padded_K / 16]` | relative scale per group of 16 weights |
| `.weight_s_channel` | `F32` | `[N]` | per-output-channel scale |
| `.weight_codebook` | `F32` | `[16]` | learned asymmetric 16-value codebook |

No `.zero`, `.zero_point`, or `.qzeros` tensor exists. The asymmetric mapping
is represented by the learned 16-value codebook rather than a conventional
integer zero-point tensor.

Example, block 0 `qkv_proj`:

```text
weight           I8       [21504, 2688]
weight_s_rel     F8_E4M3  [21504, 336]
weight_s_channel F32      [21504]
weight_codebook  F32      [16]
```

The header's complete dtype distribution is:

```text
F32       410 tensors
BF16      220 tensors
F16       102 tensors
F8_E4M3   200 tensors
I8        200 tensors
```

The 200 target weights are W4. Non-targeted norms, projections, biases, and the
pruned AdaLN curve representation stay at their source floating dtype, which
is why this is a mixed checkpoint.

## Execution contract confirmed from the maintained kernel package

The original converter/loader documents:

- calibration-free learned codebooks;
- runtime INT8 activation quantization;
- an optimized INT8 GEMM path;
- `comfy-kitchen` `AsymW4A8Int8Layout`;
- CUDA Toolkit 12.8 or newer;
- NVIDIA SM80 or newer;
- tested RTX 4090 / SM89;

As of `comfy-kitchen` 0.2.31, the layout is part of the Apache-2.0 licensed
`Comfy-Org/comfy-kitchen` package. PyPI publishes a Windows x86-64 wheel. The
layout declares `MIN_SM_VERSION = (8, 0)` and routes `linear`, `mm`, and
`addmm` through registered eager, Triton, or native CUDA implementations.

Sources:

- https://github.com/Comfy-Org/comfy-kitchen/pull/90
- https://github.com/Comfy-Org/comfy-kitchen/blob/main/comfy_kitchen/tensor/w4a8_int8.py
- https://pypi.org/project/comfy-kitchen/0.2.31/

RTX 3060 / SM86 satisfies the layout's architecture floor. That is not enough
to claim HAYATE support: the distributed Windows extension must execute on the
actual card. HAYATE therefore forces a quantize plus linear operation through
the native `cuda` backend with `hayate kernel-check`; an eager fallback cannot
turn the check green. The reference host passed both checks, so the loader
reports:

```text
FULL_STRICT_LOAD_PASS / NATIVE_REAL_LAYER_PASS
```

## Still unknown or not yet accepted

| Question | State |
|---|---|
| Exact nibble ordering and codebook interpretation | Delegated to the audited `comfy-kitchen` layout; HAYATE does not reimplement it |
| Exact runtime activation quantization granularity/rounding | Delegated to the registered kernel contract; validate by fixed-seed output tests |
| Windows RTX 3060 native CUDA operation | **PASS** on the reference host: PyTorch 2.11.0+cu128, comfy-kitchen 0.2.31, sm_86 |
| End-to-end RTX 3060 execution | **PASS**: fixed-seed 50-step 256x256 short sample |
| Peak VRAM/RAM across broader target sizes | **PENDING** |
| Same-seed unquantized quality equivalence | **PENDING**: matching baseline weights unavailable locally |
| LoRA merge behavior on packed weights | **UNKNOWN / not implemented** |

## HAYATE v0.1 validation

`W4A8Loader` now checks, before allocation:

1. explicit `asym_w4a8_int8` metadata;
2. group size 16 and ConvRot group size 256;
3. all four companion tensors for every declared layer;
4. I8/F8/F32 dtype contract;
5. packed K, relative-scale K, output-channel, and codebook shapes;
6. absence/presence of explicit zero-point tensors;
7. hardware report and native kernel-probe state.

`comfy-kitchen` is imported directly and does not require a ComfyUI runtime.
On the reference RTX 3060, the forced native CUDA smoke test passed with
relative L2 `0.071987` and a `0.434 ms` mean for the tiny `[8,256] x
[128,256]` test linear (20 timed calls). A real public
`blocks.0.attn.out_proj` layer also passed with relative L2 `0.011431` and
`0.359 ms` mean. The complete transformer strict-loaded all 635 native tensors
in `19.41 s` at `13.45 GB` RSS. During generation, the packed-aware offloader
kept one block resident and streamed 49 through a `0.411 GiB` GPU ring without
expanding W4 storage to BF16.
