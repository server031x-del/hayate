# Third-party notices

HAYATE is designed around the MiniMax-H3 engine architecture in
[`maybleMyers/h3`](https://github.com/maybleMyers/h3), audited at commit
`94220c1fdf14d6d9d40be06fb99f55b27c0d9024`.

No source code from that repository is redistributed in HAYATE v0.1. The exact
component classification and licensing review are in:

- `docs/UPSTREAM_COMPONENT_CLASSIFICATION.md`
- `docs/UPSTREAM_LICENSE_AUDIT.md`

Several upstream model-definition files state Apache-2.0 provenance from the
MiniMax and Hugging Face teams. If those files are vendored in a later release,
their copyright/license headers and a modification notice must be preserved.

HAYATE's W4A8 format research references
[`starsFriday/ComfyUI-W4A8-Loader`](https://github.com/starsFriday/ComfyUI-W4A8-Loader)
and uses `Comfy-Org/comfy-kitchen` 0.2.31 directly as an optional dependency.
`comfy-kitchen` is licensed under Apache-2.0; HAYATE does not copy or modify its
sources. The inspected starsFriday loader repository has no top-level license,
so its code is not copied without a license grant.

MiniMax-H3 model weights are separately licensed by MiniMax. HAYATE does not
ship model weights and does not change their license terms.

HAYATE's runtime-adaptive transformer cache is an independent MiniMax-H3
integration based on the algorithm published by
[`H-EmbodVis/EasyCache`](https://github.com/H-EmbodVis/EasyCache), licensed
under Apache-2.0. HAYATE does not copy ComfyUI's GPL EasyCache integration.

The optional Windows SageAttention profile uses
[`woct0rdho/SageAttention`](https://github.com/woct0rdho/SageAttention) 2.2.0
post4, derived from [`thu-ml/SageAttention`](https://github.com/thu-ml/SageAttention),
under Apache-2.0. Its optional Triton runtime is
[`woct0rdho/triton-windows`](https://github.com/woct0rdho/triton-windows),
distributed under the MIT license. HAYATE calls these installed packages and
does not copy their sources.

The optional local WebUI calls FastAPI (MIT), Starlette (BSD-3-Clause), Uvicorn
(BSD-3-Clause), python-multipart (Apache-2.0), and Pillow (HPND) as installed
Python dependencies. HAYATE's WebUI HTML, CSS, and JavaScript are original
project sources and do not vendor a third-party UI framework.
