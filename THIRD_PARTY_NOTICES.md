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

HAYATE's optional PDD acceleration mode interoperates with
[`alibaba-pai/MiniMax-H3-Acc-LoRAs`](https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs).
That repository declares Apache-2.0 in its model-card metadata; a top-level
license file was not present when inspected on 2026-08-27. HAYATE does not copy
its Python helper and independently implements the published PDD schedule and
checkpoint contract. Projection of released 2688-wide AdaLN adapters onto the
pruned 8-wide coordinates follows the mathematical identity and external data
described by
[`multimodalart/MiniMax-H3-Pruned`](https://huggingface.co/multimodalart/MiniMax-H3-Pruned).
The affine map and PDD/model weights are downloaded separately and retain their
source terms, including the MiniMax-H3 Community License Agreement where
applicable.

The optional Windows SageAttention profile uses the 2.2.0 Windows post5
cu130/torch2.10+ wheel from
[`woct0rdho/SageAttention`](https://github.com/woct0rdho/SageAttention), derived
from [`thu-ml/SageAttention`](https://github.com/thu-ml/SageAttention), under
Apache-2.0. Its optional Triton runtime is
[`woct0rdho/triton-windows`](https://github.com/woct0rdho/triton-windows),
distributed under the MIT license. HAYATE calls these installed packages and
does not copy their sources.

The optional local WebUI calls FastAPI (MIT), Starlette (BSD-3-Clause), Uvicorn
(BSD-3-Clause), python-multipart (Apache-2.0), and Pillow (HPND) as installed
Python dependencies. HAYATE's WebUI HTML, CSS, and JavaScript are original
project sources and do not vendor a third-party UI framework.

The optional FastH3/VSA catalog entry references
[`Kijai/MiniMax-H3-experimental`](https://huggingface.co/Kijai/MiniMax-H3-experimental).
The 4-step file is a ComfyUI single-file conversion and is not redistributed by
HAYATE; its MiniMax-H3 model terms remain applicable. If the optional FastVideo
backend is installed by an operator, its source and Apache-2.0 notices come from
[`hao-ai-lab/FastVideo`](https://github.com/hao-ai-lab/FastVideo). HAYATE does
not vendor FastVideo or the VSA kernels, and the experimental path is disabled
when those external runtime requirements are absent.

The optional ComfyUI FastH3 VSA node is derived from
[`barelymining/ComfyUI-MiniMax-H3-FastVideo`](https://github.com/barelymining/ComfyUI-MiniMax-H3-FastVideo)
at commit `be8f1ef72e3dc430c005e10923c2a0863aa6a9e8` and is distributed under
the MIT license. It calls the separately installed `vsa` 0.0.3 package from
[`hao-ai-lab/FastVideo`](https://github.com/hao-ai-lab/FastVideo), distributed
under Apache-2.0. The vendored node files retain their upstream license and
README; HAYATE-specific changes are documented in the source history. Model
weights remain external and keep the MiniMax-H3 Community License terms.

The HAYATE FastH3 setup helper pins the external model snapshot
`FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree` at revision
`5ea076f35b84da4c3c82217112fa733d8eea2ae1`. The snapshot is a 35B BF16
Diffusers-format model distributed under the MiniMax H3 Community License;
the model's license and acceptable-use terms remain the operator's
responsibility. The pinned revision and its required directory contract are
documented in `docs/FASTH3_WSL.md`.
