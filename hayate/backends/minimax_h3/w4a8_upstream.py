from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from hayate.loaders.int8_convrot_binding import INT8ConvRotCheckpointBinding
from hayate.loaders.w4a8_binding import W4A8CheckpointBinding

logger = logging.getLogger(__name__)


def _tensor_storage_bytes(tensor) -> int:
    qdata = getattr(tensor, "_qdata", None)
    params = getattr(tensor, "_params", None)
    if qdata is None:
        return int(tensor.nbytes)
    total = int(qdata.nbytes)
    if params is not None:
        for value in vars(params).values():
            if hasattr(value, "nbytes"):
                total += int(value.nbytes)
    return total


def _install_quantized_block_swap_patch() -> None:
    """Teach upstream's chunked block swap to retain packed tensor subclasses."""

    import torch
    from modules.custom_offloading_utils import (
        ChunkedStreamingOffloader,
        clean_memory_on_device,
        synchronize_device,
    )

    if getattr(ChunkedStreamingOffloader, "_hayate_quantized_swap", False):
        return

    def prepare_block(offloader, block, index):
        if offloader.blocks_to_swap <= 0:
            return
        if index < offloader.num_resident:
            old = torch.__future__.get_swap_module_params_on_conversion()
            torch.__future__.set_swap_module_params_on_conversion(True)
            try:
                block.to(offloader.device)
            finally:
                torch.__future__.set_swap_module_params_on_conversion(old)
            return

        covered = set()
        signature = []
        for chunk_index, group in enumerate(offloader.chunk_groups):
            owners = []
            for child_name in group:
                module = block.get_submodule(child_name)
                for param_name, param in module.named_parameters(prefix=child_name):
                    if param.device.type != "cpu":
                        raise RuntimeError(f"packed block master must stay on CPU: {param_name}")
                    covered.add(param_name)
                    owners.append((param, None))
                    signature.append((param_name, tuple(param.shape), param.dtype, type(param)))
            offloader.chunk_params[(index, chunk_index)] = owners
        all_names = {name for name, _ in block.named_parameters()}
        if covered != all_names:
            missing = sorted(all_names - covered)
            raise RuntimeError(f"packed chunk groups miss block params, e.g. {missing[:5]}")
        if offloader._template is None:
            offloader._template = signature
        elif signature != offloader._template:
            raise RuntimeError("packed streamed blocks are not uniform")
        if index == offloader.num_resident:
            offloader._ensure_staging()
        for module in block.modules():
            for name, value in module._buffers.items():
                if value is not None and value.device.type != offloader.device.type:
                    module._buffers[name] = value.to(offloader.device)

    def ensure_staging(offloader):
        if offloader.staging is not None:
            return
        first = offloader.num_resident
        offloader.staging = []
        offloader.set_release_evt = []
        for chunk_index in range(offloader.chunks_per_block):
            owners = [param for param, _ in offloader.chunk_params[(first, chunk_index)]]
            sets = []
            for _ in range(offloader.NUM_SETS):
                values = []
                for owner in owners:
                    value = torch.empty_like(owner, device=offloader.device)
                    if isinstance(owner, torch.nn.Parameter):
                        value = torch.nn.Parameter(value, requires_grad=False)
                    values.append(value)
                sets.append(values)
                offloader._ring_bytes += sum(_tensor_storage_bytes(t) for t in values)
            offloader.staging.append(sets)
            offloader.set_release_evt.append([None] * offloader.NUM_SETS)

    def finalize(offloader):
        if offloader.blocks_to_swap <= 0 or offloader._prepared:
            return
        first = offloader.num_resident
        block_bytes = sum(
            _tensor_storage_bytes(param)
            for chunk_index in range(offloader.chunks_per_block)
            for param, _ in offloader.chunk_params[(first, chunk_index)]
        )
        synchronize_device(offloader.device)
        clean_memory_on_device(offloader.device)
        offloader._prepared = True
        logger.info(
            "HAYATE packed block streaming: %d resident, %d streamed, %.3f GiB/block, %.3f GiB ring",
            offloader.num_resident,
            offloader.blocks_to_swap,
            block_bytes / 1024**3,
            offloader._ring_bytes / 1024**3,
        )

    def submit(offloader, pos):
        block_index, chunk_index = offloader.chunk_order[pos]
        if offloader.gate is not None and not offloader.gate.is_ready(block_index):
            return False
        set_index = offloader._set_for_pos(pos)
        release_event = offloader.set_release_evt[chunk_index][set_index]
        offloader.set_release_evt[chunk_index][set_index] = None
        owners = [param for param, _ in offloader.chunk_params[(block_index, chunk_index)]]
        holders = offloader.staging[chunk_index][set_index]
        with torch.cuda.stream(offloader.copy_stream):
            if release_event is not None:
                offloader.copy_stream.wait_event(release_event)
            for owner, gpu_holder in zip(owners, holders):
                gpu_holder.copy_(owner, non_blocking=True)
                torch.utils.swap_tensors(owner, gpu_holder)
            event = torch.cuda.Event()
            event.record(offloader.copy_stream)
        offloader.upload_evt[pos] = event
        offloader.pos_staging[pos] = set_index
        offloader._held.append(pos)
        offloader._submitted = pos
        return True

    def release(offloader, pos, record_release=True):
        block_index, chunk_index = offloader.chunk_order[pos]
        set_index = offloader.pos_staging.pop(pos)
        owners = [param for param, _ in offloader.chunk_params[(block_index, chunk_index)]]
        holders = offloader.staging[chunk_index][set_index]
        for owner, cpu_holder in zip(owners, holders):
            torch.utils.swap_tensors(owner, cpu_holder)
        if record_release:
            event = torch.cuda.Event()
            event.record()
            offloader.set_release_evt[chunk_index][set_index] = event
        offloader.upload_evt.pop(pos, None)

    ChunkedStreamingOffloader.prepare_block = prepare_block
    ChunkedStreamingOffloader._ensure_staging = ensure_staging
    ChunkedStreamingOffloader.finalize = finalize
    ChunkedStreamingOffloader._submit = submit
    ChunkedStreamingOffloader._release = release
    ChunkedStreamingOffloader._hayate_quantized_swap = True


def _assign_model_tensor(model, key: str, value) -> None:
    """Assign one tensor into a meta-initialized model without a full state-dict peak."""

    import torch

    module_path, _, name = key.rpartition(".")
    module = model.get_submodule(module_path) if module_path else model
    if name in module._parameters:
        expected = module._parameters[name]
        if expected is None or tuple(expected.shape) != tuple(value.shape):
            raise ValueError(
                f"parameter shape mismatch for {key}: "
                f"expected {None if expected is None else tuple(expected.shape)}, got {tuple(value.shape)}"
            )
        setattr(module, name, torch.nn.Parameter(value, requires_grad=False))
        return
    if name in module._buffers:
        expected = module._buffers[name]
        if expected is not None and tuple(expected.shape) != tuple(value.shape):
            raise ValueError(
                f"buffer shape mismatch for {key}: expected {tuple(expected.shape)}, got {tuple(value.shape)}"
            )
        setattr(module, name, value)
        return
    raise KeyError(f"checkpoint key does not exist in upstream model: {key}")


def is_w4a8_checkpoint(path: str | os.PathLike | None) -> bool:
    if path is None or not str(path).endswith(".safetensors") or not Path(path).is_file():
        return False
    try:
        binding = W4A8CheckpointBinding(path)
    except Exception:
        return False
    return bool(binding.layer_prefixes)


def load_w4a8_transformer(
    *,
    checkpoint_dir: str,
    checkpoint_path: str,
    device,
    dit_dtype,
    lora_weights_list=None,
    lora_multipliers=None,
):
    """Build upstream's model and assign W4A8 wrappers without dequantizing weights."""

    if lora_weights_list:
        raise NotImplementedError(
            "LoRA-on-packed-W4A8 needs dequantize/merge/requantize validation and is disabled"
        )
    import torch
    from accelerate import init_empty_weights
    from safetensors import safe_open

    from minimax_video.int8_quant import convert_int8_dit_tensor
    from minimax_video.model_loader import _is_fp32_key
    from minimax_video.transformer import MiniMaxH3Transformer3DModel

    config_path = Path(checkpoint_dir) / "transformer" / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.pop("_class_name", None)
    config.pop("_diffusers_version", None)
    binding = W4A8CheckpointBinding(checkpoint_path)
    header_names = {tensor.name: tensor for tensor in binding.header.tensors}
    curve = header_names.get("adaln_t_table")
    if curve is not None:
        grid, curve_dim = curve.shape
        config["adaln_curve_grid"] = grid
        config["time_embed_dim"] = curve_dim
    with init_empty_weights():
        model = MiniMaxH3Transformer3DModel.from_config(config)
    model.requires_grad_(False)

    quantized_names = {
        prefix + suffix
        for prefix in binding.layer_prefixes
        for suffix in (".weight", ".weight_s_rel", ".weight_s_channel", ".weight_codebook")
    }
    expected_keys = set(model.state_dict())
    loaded_keys: set[str] = set()
    with safe_open(checkpoint_path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key in quantized_names:
                continue
            value = handle.get_tensor(key).clone()
            for native_key, converted in convert_int8_dit_tensor(key, value, qkv_head_dim=0):
                if _is_fp32_key(native_key) or native_key == "adaln_t_table":
                    target_dtype = torch.float32
                elif curve is not None and (
                    ".adaln_proj.linear." in native_key or native_key.startswith("norm_out.linear.")
                ):
                    target_dtype = torch.float32
                else:
                    target_dtype = dit_dtype
                assigned = (
                    converted.to(target_dtype)
                    if converted.is_floating_point() and converted.dtype != target_dtype
                    else converted
                )
                _assign_model_tensor(model, native_key, assigned)
                loaded_keys.add(native_key)

    for index, prefix in enumerate(binding.layer_prefixes, start=1):
        converted = binding.materialize_converted(prefix, convert_int8_dit_tensor)
        for native_key, value in converted.items():
            _assign_model_tensor(model, native_key, value)
            loaded_keys.add(native_key)
        if index % 20 == 0:
            logger.info("HAYATE W4A8 binding: %d/%d layers", index, len(binding.layer_prefixes))

    missing = expected_keys - loaded_keys
    unexpected = loaded_keys - expected_keys
    if missing or unexpected:
        raise RuntimeError(
            f"HAYATE W4A8 strict key check failed: {len(missing)} missing, "
            f"{len(unexpected)} unexpected; examples={sorted(missing)[:5] + sorted(unexpected)[:5]}"
        )
    logger.info(
        "HAYATE W4A8 transformer staged load: all %d tensors assigned",
        len(loaded_keys),
    )
    model.eval().requires_grad_(False)
    return model


def install_w4a8_override(generation_module) -> None:
    """Patch only the generation module's loader reference; all other engine code is reused."""

    from minimax_video import model_loader

    _install_quantized_block_swap_patch()

    original = model_loader.load_transformer
    original_vae = model_loader.load_vae
    original_audio_vae = model_loader.load_audio_vae

    def dispatch(
        ckpt_dir,
        device,
        task="t2va",
        dit_dtype=None,
        fp8=False,
        fp8_scaled=False,
        fp8_fast=False,
        fp8_exclude_adaln=False,
        lora_weights_list=None,
        lora_multipliers=None,
        dit_path=None,
        int8_use_int_mm=False,
    ):
        if not is_w4a8_checkpoint(dit_path):
            return original(
                ckpt_dir,
                device,
                task=task,
                dit_dtype=dit_dtype,
                fp8=fp8,
                fp8_scaled=fp8_scaled,
                fp8_fast=fp8_fast,
                fp8_exclude_adaln=fp8_exclude_adaln,
                lora_weights_list=lora_weights_list,
                lora_multipliers=lora_multipliers,
                dit_path=dit_path,
                int8_use_int_mm=int8_use_int_mm,
            )
        if task == "ref2va":
            raise ValueError("the initial W4A8 checkpoint is FL2VA/T2VA, not transformer_ref")
        if fp8 or fp8_scaled or fp8_fast:
            logger.warning("W4A8 checkpoint is already quantized; FP8 flags are ignored")
        return load_w4a8_transformer(
            checkpoint_dir=ckpt_dir,
            checkpoint_path=dit_path,
            device=device,
            dit_dtype=dit_dtype,
            lora_weights_list=lora_weights_list,
            lora_multipliers=lora_multipliers,
        )

    model_loader.load_transformer = dispatch

    def load_vae_dispatch(
        ckpt_dir,
        device,
        vae_dtype=None,
        vae_path=None,
    ):
        if not _is_int8_convrot_video_vae(vae_path):
            return original_vae(
                ckpt_dir,
                device,
                vae_dtype=vae_dtype,
                vae_path=vae_path,
            )
        return _load_int8_convrot_video_vae(
            ckpt_dir=ckpt_dir,
            checkpoint_path=vae_path,
            device=device,
            vae_dtype=vae_dtype,
        )

    model_loader.load_vae = load_vae_dispatch

    def load_audio_vae_dispatch(
        ckpt_dir,
        device,
        dtype=None,
        audio_vae_path=None,
    ):
        if not _is_single_weight_audio_vae(audio_vae_path):
            return original_audio_vae(
                ckpt_dir,
                device,
                dtype=dtype,
                audio_vae_path=audio_vae_path,
            )
        return _load_single_weight_audio_vae(
            ckpt_dir=ckpt_dir,
            checkpoint_path=audio_vae_path,
            device=device,
            dtype=dtype,
        )

    model_loader.load_audio_vae = load_audio_vae_dispatch


def _is_int8_convrot_video_vae(path) -> bool:
    if path is None or not str(path).endswith(".safetensors") or not Path(path).is_file():
        return False
    from hayate.models.safetensors_header import read_safetensors_header

    try:
        header = read_safetensors_header(path)
    except Exception:
        return False
    names = {tensor.name: tensor for tensor in header.tensors}
    return any(
        name.endswith(".weight")
        and tensor.dtype == "I8"
        and name[: -len(".weight")] + ".weight_scale" in names
        and name[: -len(".weight")] + ".comfy_quant" in names
        for name, tensor in names.items()
    )


def _is_single_weight_audio_vae(path) -> bool:
    """Detect the public single-file audio VAE with merged weight normalization."""

    if path is None or not str(path).endswith(".safetensors") or not Path(path).is_file():
        return False
    from hayate.models.safetensors_header import read_safetensors_header

    try:
        names = {tensor.name for tensor in read_safetensors_header(path).tensors}
    except Exception:
        return False
    return (
        "decoder.conv_pre.weight" in names
        and "decoder.conv_pre.weight_g" not in names
        and "decoder.conv_pre.weight_v" not in names
    )


def remove_weight_norm_tree(model) -> int:
    """Turn legacy weight_g/weight_v modules into ordinary weight modules."""

    from torch.nn.utils import remove_weight_norm

    removed = 0
    for module in model.modules():
        if "weight_g" not in module._parameters or "weight_v" not in module._parameters:
            continue
        remove_weight_norm(module)
        removed += 1
    return removed


def _load_single_weight_audio_vae(
    *,
    ckpt_dir: str,
    checkpoint_path: str,
    device,
    dtype,
):
    """Load the public FP32 audio VAE after matching its merged-weight layout."""

    import torch
    from safetensors import safe_open

    from minimax_video.model_loader import _from_config
    from minimax_video.vae_audio import AutoencoderKLMiniMaxH3Audio

    dtype = dtype or torch.float32
    config_path = Path(ckpt_dir) / "audio_vae" / "config.json"
    model = _from_config(AutoencoderKLMiniMaxH3Audio, str(config_path), dtype)
    removed = remove_weight_norm_tree(model)
    state: dict[str, torch.Tensor] = {}
    with safe_open(checkpoint_path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if key in {"latents_mean", "latents_std"}:
                continue
            value = handle.get_tensor(key).clone()
            state[key] = value.to(dtype) if value.is_floating_point() else value
    info = model.load_state_dict(state, strict=True, assign=True)
    logger.info(
        "HAYATE single-weight Audio VAE load (%d weight norms removed): %s",
        removed,
        info,
    )
    model.eval().requires_grad_(False)
    return model.to(device)


def _load_int8_convrot_video_vae(
    *,
    ckpt_dir: str,
    checkpoint_path: str,
    device,
    vae_dtype,
):
    import torch
    from safetensors import safe_open

    from _convert_minimax_h3_upstream import convert_video_vae_key
    from minimax_video.model_loader import _from_config
    from minimax_video.vae_video import AutoencoderKLMiniMaxH3

    vae_dtype = vae_dtype or torch.float32
    config_path = Path(ckpt_dir) / "vae" / "config.json"
    model = _from_config(AutoencoderKLMiniMaxH3, str(config_path), vae_dtype)
    converter_config = dict(model.config)
    model.requires_grad_(False)
    binding = INT8ConvRotCheckpointBinding(checkpoint_path)
    state: dict[str, torch.Tensor] = {}
    with safe_open(checkpoint_path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        prefixes = sorted(
            key[: -len(".weight")]
            for key in keys
            if key.endswith(".weight")
            and key[: -len(".weight")] + ".weight_scale" in keys
            and key[: -len(".weight")] + ".comfy_quant" in keys
        )
        skipped = {
            prefix + suffix
            for prefix in prefixes
            for suffix in (".weight", ".weight_scale", ".comfy_quant")
        }
        for key in handle.keys():
            if key in skipped:
                continue
            if key in {"latents_mean", "latents_std"}:
                continue
            value = handle.get_tensor(key).clone()
            for native_key, converted in convert_video_vae_key(key, value, converter_config):
                state[native_key] = (
                    converted.to(vae_dtype) if converted.is_floating_point() else converted
                )
    for index, prefix in enumerate(prefixes, start=1):
        state.update(binding.materialize_converted(
            prefix,
            convert_video_vae_key,
            converter_config,
            device="cpu",
            orig_dtype=str(vae_dtype).removeprefix("torch."),
        ))
        if index % 24 == 0:
            logger.info("HAYATE INT8 Video VAE binding: %d/%d layers", index, len(prefixes))
    info = model.load_state_dict(state, strict=True, assign=True)
    logger.info("HAYATE INT8 Video VAE load: %s", info)
    model.eval().requires_grad_(False)
    return model.to(device)
