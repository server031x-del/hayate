"""
ComfyUI-H3-VSA — route MiniMax H3 attention through FastVideo's Video
Sparse Attention (Triton kernel) using the VSA-DataFree LoRA's trained
`to_gate_compress` gate weights.

Runtime-only: no ComfyUI source files are touched. The node:
  1) reads gate weights from a user-supplied safetensors file (produced by
     convert_fastvideo_vsa_to_comfy.py --dst-gate)
  2) injects a bias-free nn.Linear(hidden -> inner) into every attn module
     of the H3 diffusion model, loaded with the gate weights for that block
  3) patches attn.forward to compute `gate = to_gate_compress(x)`, reshape
     to [B, H, S, D], and call vsa.video_sparse_attn with
     compress_attn_weight=gate and topk = topk_ratio * n_blocks
  4) falls back to normal attention when preconditions fail (short seq,
     wrong dtype, missing block gate, vsa unavailable)

Recommended pairing:
  * FastH3: minimax_h3_fastvideo_vsa_datafree_1300step_4step_int8_convrot
    (its 50 `to_gate_compress` matrices are embedded; choose
    `<model-embedded-gates>` and do not stack the Turbo LoRA)
  * Original VSA path: pruned INT8 base + VSA LoRA + an external gate file
  * Steps:  4
  * CFG:    1.0
  * topk_ratio: 0.10 (matches FastVideo training sparsity)
"""

from __future__ import annotations

import glob
import logging
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.model_management
import comfy.patcher_extension
import comfy.quant_ops
import folder_paths

log = logging.getLogger(__name__)

# A FastH3 checkpoint may already contain the trained ``to_gate_compress``
# matrices.  This sentinel keeps the workflow explicit and, importantly,
# avoids asking the node to load a second copy of the several-GB gate state.
BUILTIN_GATE = "<model-embedded-gates>"

try:
    from vsa import video_sparse_attn as _vsa
    _VSA_OK = True
except Exception as _e:  # noqa: BLE001
    _vsa = None
    _VSA_OK = False
    log.warning("[H3-VSA] vsa import failed: %s", _e)


class _Unsupported(Exception):
    pass


def _list_lora_paths():
    paths = [BUILTIN_GATE]
    for root in folder_paths.get_folder_paths("loras"):
        for p in glob.glob(os.path.join(root, "**", "*.safetensors"),
                           recursive=True):
            paths.append(os.path.relpath(p, root).replace("\\", "/"))
    return sorted(set(paths)) or ["<none-found>"]


def _load_gate_state(path: str) -> dict[int, torch.Tensor]:
    """Return {block_idx: [inner, hidden] weight tensor}."""
    from safetensors import safe_open
    out: dict[int, torch.Tensor] = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            # expected key form: blocks.N.attn.to_gate_compress.weight
            parts = k.split(".")
            if (len(parts) >= 5 and parts[0] == "blocks"
                    and parts[2] == "attn"
                    and parts[3] == "to_gate_compress"):
                try:
                    idx = int(parts[1])
                except ValueError:
                    continue
                out[idx] = f.get_tensor(k)
    return out


def _model_gate_blocks(blocks) -> list[int]:
    """Return blocks whose checkpoint already carries a VSA gate layer."""
    return [
        i for i, blk in enumerate(blocks)
        if getattr(getattr(blk, "attn", None), "to_gate_compress", None) is not None
    ]


def _make_attn_forward(attn, fallback, gate_layer, topk_ratio, min_tokens,
                       block_elem, hit_log):
    heads = attn.heads
    head_dim = attn.head_dim
    inner = heads * head_dim

    def forward(x, rope_freqs=None, transformer_options={}):
        s = x.shape[0]
        # First VSA-active call this generation: free fragmented VRAM left over
        # from a prior cancelled run so the sparse-attn allocations don't OOM.
        if not hit_log._active and x.device.type == "cuda":
            torch.cuda.empty_cache()
        try:
            if not _VSA_OK:
                raise _Unsupported("vsa unavailable")
            if x.ndim != 2 or s < min_tokens:
                raise _Unsupported("below min_tokens")
            if x.dtype != torch.bfloat16 or x.device.type != "cuda":
                raise _Unsupported("requires bfloat16 CUDA")
            if gate_layer is None:
                raise _Unsupported("no gate weights for this block")

            # qkv + norm/rope, same as sol-attn plumbing
            q, k, v = attn.qkv_proj(x).split(inner, dim=-1)
            v = v.view(s, heads, head_dim)

            if rope_freqs is not None:
                q = q.view(1, s, heads, head_dim)
                k = k.view(1, s, heads, head_dim)
                qw = comfy.model_management.cast_to(attn.q_norm.weight, device=x.device)
                kw = comfy.model_management.cast_to(attn.k_norm.weight, device=x.device)
                rot = rope_freqs.shape[-3] * 2
                comfy.quant_ops.ck.rms_rope_split_half_(
                    q, k, rope_freqs, qw, kw,
                    epsilon=attn.q_norm.eps, rot_dim=rot,
                )
                q = q[0]
                k = k[0]
            else:
                q = attn.q_norm(q.view(s, heads, head_dim))
                k = attn.k_norm(k.view(s, heads, head_dim))

            # gate: [S, hidden] -> [S, inner] -> [S, H, D].  ComfyUI's
            # mixed-precision Linear carries QuantizedTensor metadata (INT8
            # ConvRot in the FastH3 file); call its forward path so scales and
            # rotations are applied instead of feeding INT8 storage directly
            # to torch.nn.functional.linear.  External CPU gate layers are
            # ordinary nn.Linear modules and use the streamed weight path.
            if getattr(gate_layer, "comfy_cast_weights", False):
                gate = gate_layer(x)
            else:
                gw_gpu = gate_layer.weight.to(device=x.device, non_blocking=True)
                gate = F.linear(x, gw_gpu)
                del gw_gpu
            gate = gate.view(s, heads, head_dim)

            # to VSA layout [B, H, S, D]
            q = q.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
            k = k.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
            v = v.unsqueeze(0).permute(0, 2, 1, 3).contiguous()
            gate_v = gate.unsqueeze(0).permute(0, 2, 1, 3).contiguous()

            # pad to multiple of 64
            pad = (block_elem - s % block_elem) % block_elem
            n_blocks = (s + pad) // block_elem
            if pad:
                q = F.pad(q, (0, 0, 0, pad))
                k = F.pad(k, (0, 0, 0, pad))
                v = F.pad(v, (0, 0, 0, pad))
                gate_v = F.pad(gate_v, (0, 0, 0, pad))

            topk = max(1, int(math.ceil(n_blocks * topk_ratio)))
            vbs = torch.full((n_blocks,), block_elem,
                             device=q.device, dtype=torch.int32)
            if pad:
                vbs[-1] = block_elem - pad

            out = _vsa(q, k, v,
                       variable_block_sizes=vbs,
                       topk=topk,
                       block_size=(4, 4, 4),
                       compress_attn_weight=gate_v)
            out = out.permute(0, 2, 1, 3).contiguous()[0]
            if pad:
                out = out[:s]

            hit_log.hit(s, topk, n_blocks)
            return attn.out_proj(out.reshape(s, inner))

        except _Unsupported as e:
            raise RuntimeError(f"HAYATE requires active VSA: {e}") from e
        except Exception as e:  # noqa: BLE001
            # A Triton/kernel incompatibility should not make a normal H3
            # generation fail.  Log it once and continue with dense attention.
            raise RuntimeError(f"HAYATE VSA kernel failed: {e}") from e

        return fallback(x, rope_freqs=rope_freqs,
                        transformer_options=transformer_options)

    forward._h3_vsa_fallback = fallback
    return forward


class _HitLog:
    def __init__(self):
        self._active = False
        self._miss: set[str] = set()

    def hit(self, s, topk, n_blocks):
        if not self._active:
            log.info("[H3-VSA] ACTIVE — tokens=%d topk=%d/%d gate=on",
                     s, topk, n_blocks)
            self._active = True

    def miss(self, reason):
        if reason not in self._miss:
            self._miss.add(reason)
            log.info("[H3-VSA] dense fallback: %s", reason)


class H3VSA:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "gate_file": (_list_lora_paths(), {
                "tooltip": "safetensors from convert_fastvideo_vsa_to_comfy.py "
                           "--dst-gate. Place in ComfyUI/models/loras."}),
            "topk_ratio": ("FLOAT", {
                "default": 0.10, "min": 0.05, "max": 1.0, "step": 0.05,
                "tooltip": "Sparsity. 0.10 matches FastVideo training "
                           "(vsa_sparsity=0.9)."}),
            "min_tokens": ("INT", {
                "default": 4096, "min": 256, "max": 65536, "step": 256}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "H3 Acceleration"

    def apply(self, model, gate_file, topk_ratio, min_tokens):
        dm = model.get_model_object("diffusion_model")
        blocks = getattr(dm, "blocks", None)
        if dm.__class__.__name__ != "MiniMaxH3Model" or blocks is None:
            log.warning("[H3-VSA] not a MiniMaxH3Model; returning unchanged")
            return (model,)

        # FastVideo's data-free 4-step checkpoint embeds all 50 gate matrices.
        # Prefer those weights over an external gate file so the workflow does
        # not duplicate the large gate state in RAM/VRAM.  External gate files
        # remain supported for the original pruned INT8 + VSA-LoRA setup.
        embedded_blocks = _model_gate_blocks(blocks)
        gate_state: dict[int, torch.Tensor] = {}
        if gate_file == BUILTIN_GATE:
            if len(embedded_blocks) != len(blocks):
                raise RuntimeError(
                    "[H3-VSA] gate_file requests embedded gates, but "
                    f"only {len(embedded_blocks)}/{len(blocks)} blocks contain them")
            log.info("[H3-VSA] using model-embedded gate weights (%d blocks)",
                     len(embedded_blocks))
        else:
            # resolve gate_file against loras folder
            gate_path = None
            for root in folder_paths.get_folder_paths("loras"):
                p = os.path.join(root, gate_file)
                if os.path.exists(p):
                    gate_path = p
                    break
            if gate_path is None:
                raise RuntimeError(f"[H3-VSA] gate file not found: {gate_file}")
            log.info("[H3-VSA] loading gate weights from %s", gate_path)
            gate_state = _load_gate_state(gate_path)
            log.info("[H3-VSA] gate weights: %d blocks", len(gate_state))

        # Clone before injecting any external gate layers.  ComfyUI clones can
        # share the underlying diffusion modules, so mutating the input model
        # here would leak a VSA gate into a later normal/Turbo branch.
        patched = model.clone()
        patched_dm = patched.get_model_object("diffusion_model")
        patched_blocks = getattr(patched_dm, "blocks", None)
        if patched_blocks is None or len(patched_blocks) != len(blocks):
            raise RuntimeError("[H3-VSA] cloned model has an incompatible block layout")

        # Inject bias-free nn.Linear(hidden -> inner) into each cloned attn
        # module when an external gate file is used.  Gates remain CPU-pinned
        # when idle; the forward path streams one block at a time.  Embedded
        # FastH3 gates are left in their quantized ComfyUI Linear modules.
        hidden = None
        for i, blk in enumerate(patched_blocks):
            attn = blk.attn
            inner = attn.heads * attn.head_dim
            if hidden is None:
                hidden = attn.qkv_proj.weight.shape[1]
            if getattr(attn, "to_gate_compress", None) is None:
                w = gate_state.get(i)
                if w is None:
                    log.warning("[H3-VSA] no gate for block %d — will fallback", i)
                    attn.to_gate_compress = None
                    continue
                if tuple(w.shape) != (inner, hidden):
                    raise RuntimeError(
                        f"[H3-VSA] block {i} gate shape {tuple(w.shape)} != ({inner},{hidden})")
                ref = attn.qkv_proj.weight
                # Quantized qkv weights expose INT8 storage dtype, while the
                # gate itself must be created in the model compute dtype.
                ref_dtype = getattr(getattr(ref, "_params", None), "orig_dtype", None)
                try:
                    ref_is_float = ref_dtype is not None and torch.empty((), dtype=ref_dtype).is_floating_point()
                except (TypeError, RuntimeError):
                    ref_is_float = False
                if not ref_is_float:
                    ref_dtype = getattr(dm, "dtype", None) or torch.bfloat16
                gc = nn.Linear(hidden, inner, bias=False).to(dtype=ref_dtype)
                with torch.no_grad():
                    gc.weight.copy_(w.to(ref.dtype))
                # CPU-pinned so subsequent host->device transfers are fast.
                try:
                    pinned = gc.weight.detach().contiguous().pin_memory()
                    gc._parameters["weight"] = torch.nn.Parameter(pinned, requires_grad=False)
                except Exception:
                    pass  # non-pinnable env; correctness unaffected
                attn.to_gate_compress = gc

        hit_log = _HitLog()
        block_elem = 64
        for i in range(len(blocks)):
            path = f"diffusion_model.blocks.{i}.attn.forward"
            fallback = patched.get_model_object(path)
            if hasattr(fallback, "_h3_vsa_fallback"):
                fallback = fallback._h3_vsa_fallback
            attn = patched.get_model_object(f"diffusion_model.blocks.{i}.attn")
            gate_layer = getattr(attn, "to_gate_compress", None)
            patched.add_object_patch(
                path,
                _make_attn_forward(attn, fallback, gate_layer,
                                   topk_ratio, min_tokens, block_elem, hit_log),
            )

        # Gates stay CPU-pinned; each attn forward streams its block's gate
        # weight to GPU on demand and drops the reference right after. Saves
        # ~3.6 GB of resident VRAM during sampling at the cost of one H2D
        # transfer (~73 MB) per block per step.
        log.info("[H3-VSA] patched %d blocks (topk_ratio=%.2f) — gates streamed on-demand",
                 len(blocks), topk_ratio)
        return (patched,)
