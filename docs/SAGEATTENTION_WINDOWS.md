# SageAttention on the RTX 3060 Windows reference host

The validated environment is Python 3.12.11, PyTorch 2.11.0+cu128, RTX 3060
SM86, Triton Windows 3.6.0.post26, and SageAttention 2.2.0 post4. PyTorch 2.11
is paired with Triton 3.6 rather than 3.7 by the Triton Windows compatibility
table.

The SageAttention wheel was downloaded from the `woct0rdho/SageAttention`
release `v2.2.0-windows.post4`:

```text
sageattention-2.2.0+cu128torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl
SHA-256 B8B3134D00DFBDAE5C10CC34CC8508891D9420ADAA182502FA30A496428531ED
```

It declares Apache-2.0, supports Python's ABI3 from 3.9 onward, and bundles
SM86 CUDA kernels. The local installation was confined to HAYATE's `.venv`:

```powershell
uv pip install --python .venv/Scripts/python.exe `
  "triton-windows==3.6.0.post26" `
  M:/path/to/sageattention-2.2.0+cu128torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl
```

At 56 heads, head dimension 128, BF16, and batch one, the installed `sageattn`
entry point was compared against PyTorch SDPA:

| Tokens | SDPA | SageAttention | Kernel speedup | Relative L2 |
|---:|---:|---:|---:|---:|
| 4,096 | 34.99 ms | 14.88 ms | 2.35x | 0.01162 |
| 16,384 | 509.79 ms | 198.09 ms | 2.57x | 0.01170 |

The results were finite. SageAttention is approximate, so the relative error is
an expected quality-gate input rather than an equivalence pass. The fixed-seed
end-to-end result is recorded in `docs/EXECUTION_VALIDATION.md`. Use
`--rtx3060-fast-sage` only after the package import succeeds; keep
`--rtx3060-fast`/SDPA as the fidelity reference and fallback.
Invoke the profile through `uv run --no-sync hayate ...` or the venv's
`hayate.exe`; a normal exact environment sync does not know about this
platform-specific wheel unless it is added to the project dependency set.

## ComfyUI H3 VSA Turbo profile

The separate Kijai ComfyUI runtime uses a CUDA 13 build and a newer Windows
wheel than the HAYATE-native profile above:

- Runtime: `M:\Project\HAYATE-ComfyUI\.venv`
- PyTorch: `2.11.0+cu130`
- Triton: `triton-windows==3.6.0.post26`
- SageAttention: 2.2.0 Windows `post5`, `cu130torch2.10.0andhigher`
- ComfyUI: Kijai `vsa` branch, commit `10febb01d7be73d1491cf5e5347b5ab8b6c2c09e`

The packages were imported and exercised on the RTX 3060. Start the shared
model runtime with:

```powershell
.\scripts\start_comfyui_hayate.ps1 -EnableTurbo
```

This is an external-ComfyUI path and does not change the native HAYATE
`--fast-sage` profile or its SDPA fidelity reference. See
[`COMFYUI_HAYATE.md`](COMFYUI_HAYATE.md) for the shared model paths and the
end-to-end VSA timings.
