from __future__ import annotations

import json
import logging
import types
from pathlib import Path

from hayate.loaders.nvfp4_awq_binding import NVFP4AWQCheckpointBinding
from hayate.models.quantization import detect_quantization
from hayate.models.safetensors_header import read_safetensors_header
from hayate.models.types import QuantizationFormat

logger = logging.getLogger(__name__)


def _install_quantized_streamer_patch() -> None:
    """Make upstream's double-buffer streamer safe for tensor subclasses."""

    import torch
    from minimax_video.qwen3vl_text import _LayerStreamer

    if getattr(_LayerStreamer, "_hayate_quantized_swap", False):
        return

    def begin(streamer, layers, streamed):
        for idx in streamed:
            if idx in streamer._owners or idx in streamer._skip:
                continue
            owners = [p for _, p in layers[idx].named_parameters()] + [
                b for _, b in layers[idx].named_buffers()
            ]
            signature = tuple((tuple(t.shape), t.dtype, type(t)) for t in owners)
            if streamer._signature is None:
                streamer._signature = signature
            if signature != streamer._signature:
                streamer._skip.add(idx)
                continue
            streamer._owners[idx] = owners
        streamer._pending = [i for i in streamed if i in streamer._owners]
        for slot in streamer.slots:
            if streamer._pending:
                streamer._upload(streamer._pending.pop(0), slot)

    def upload(streamer, idx, slot):
        owners = streamer._owners[idx]
        if slot.tensors is None:
            slot.tensors = []
            for owner in owners:
                value = torch.empty_like(owner, device=streamer.device)
                if isinstance(owner, torch.nn.Parameter):
                    value = torch.nn.Parameter(value, requires_grad=False)
                slot.tensors.append(value)
        streamer.copy_stream.wait_event(slot.free)
        with torch.cuda.stream(streamer.copy_stream):
            for dst, src in zip(slot.tensors, owners):
                dst.copy_(src, non_blocking=True)
        slot.ready.record(streamer.copy_stream)
        streamer._active[idx] = slot

    def attach(streamer, idx):
        slot = streamer._active[idx]
        torch.cuda.current_stream().wait_event(slot.ready)
        for owner, gpu_holder in zip(streamer._owners[idx], slot.tensors):
            torch.utils.swap_tensors(owner, gpu_holder)

    def detach(streamer, idx):
        slot = streamer._active.pop(idx)
        slot.free.record(torch.cuda.current_stream())
        for owner, cpu_holder in zip(streamer._owners[idx], slot.tensors):
            torch.utils.swap_tensors(owner, cpu_holder)
        if streamer._pending:
            streamer._upload(streamer._pending.pop(0), slot)

    def end(streamer):
        torch.cuda.current_stream().synchronize()
        for idx, slot in list(streamer._active.items()):
            for owner, cpu_holder in zip(streamer._owners[idx], slot.tensors):
                if owner.device.type != "cpu":
                    torch.utils.swap_tensors(owner, cpu_holder)
        streamer._active.clear()
        streamer._pending = []
        for slot in streamer.slots:
            slot.tensors = None

    _LayerStreamer.begin = begin
    _LayerStreamer._upload = upload
    _LayerStreamer.attach = attach
    _LayerStreamer.detach = detach
    _LayerStreamer.end = end
    _LayerStreamer._hayate_quantized_swap = True


def _is_nvfp4_awq(path) -> bool:
    if path is None or not Path(path).is_file():
        return False
    try:
        header = read_safetensors_header(path)
        return (
            detect_quantization(header.metadata, header.tensors).format
            is QuantizationFormat.NVFP4_AWQ
        )
    except Exception:
        return False


def _normalized_text_prefix(prefix: str) -> str | None:
    for known in ("model.language_model.", "language_model.", "model.text_model."):
        if prefix.startswith(known):
            return prefix[len(known) :]
    if prefix.startswith("model."):
        return prefix[len("model.") :]
    return None


def install_nvfp4_conditioner_override() -> None:
    """Adapt upstream's streaming reader while retaining its layer-50 model logic."""

    from minimax_video.conditioner import MiniMaxH3Conditioner
    original_load = MiniMaxH3Conditioner._load_weights

    def load_weights(self, encoder_dir, gpu_layers, text_encoder_path=None, int8_use_int_mm=False):
        if not _is_nvfp4_awq(text_encoder_path):
            return original_load(
                self,
                encoder_dir,
                gpu_layers,
                text_encoder_path=text_encoder_path,
                int8_use_int_mm=int8_use_int_mm,
            )

        import torch
        from comfy_kitchen.tensor import (
            QuantizedTensor,
            TensorCoreNVFP4Layout,
            TensorWiseINT8Layout,
        )
        from hayate.backends.minimax_h3.w4a8_upstream import _assign_model_tensor
        from hayate.loaders.tensor_source import CheckpointTensorSource
        from minimax_video.conditioner import _strip_known_prefixes, _wanted_text_key
        from minimax_video.packing import MINIMAX_H3_TEXT_ENCODER_LAYER

        _install_quantized_streamer_patch()

        captured_pre_quant: dict[str, torch.Tensor] = {}
        loaded_text: set[str] = set()
        loaded_vision: set[str] = set()
        # Linux keeps the mmap owner alive with the conditioner. Windows drains an owned
        # pread dictionary into the model to avoid native faults at the mmap/Torch boundary.
        source = CheckpointTensorSource(text_encoder_path)
        self._hayate_text_encoder_source = source
        logger.info("HAYATE NVFP4/AWQ tensor source: %s", source.backend)
        if source:
            keys = list(source.keys())
            marker_formats: dict[str, str] = {}
            for key in keys:
                if not key.endswith(".comfy_quant"):
                    continue
                try:
                    marker = json.loads(
                        bytes(source.take_tensor(key).tolist()).decode("utf-8")
                    )
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                    continue
                marker_formats[key[: -len(".comfy_quant")]] = marker.get("format", "")

            companion_names = {
                prefix + suffix
                for prefix, format_name in marker_formats.items()
                for suffix in (
                    (".weight_scale", ".weight_scale_2", ".comfy_quant", ".pre_quant_scale")
                    if format_name == "nvfp4"
                    else (".weight_scale", ".comfy_quant")
                )
            }
            for key in keys:
                prefix = key[: -len(".weight")] if key.endswith(".weight") else None
                format_name = marker_formats.get(prefix or "")
                if key in companion_names:
                    continue

                if format_name in {"nvfp4", "int8_tensorwise"}:
                    qdata = source.take_tensor(key)
                    if format_name == "nvfp4":
                        block_scale = source.take_tensor(prefix + ".weight_scale")
                        tensor_scale = source.take_tensor(prefix + ".weight_scale_2")
                        params = TensorCoreNVFP4Layout.Params(
                            scale=tensor_scale,
                            block_scale=block_scale,
                            orig_dtype=self.dtype,
                            orig_shape=(qdata.shape[0], qdata.shape[1] * 2),
                        )
                        value = QuantizedTensor(qdata, "TensorCoreNVFP4Layout", params)
                        pre_key = prefix + ".pre_quant_scale"
                        if pre_key in keys:
                            normalized = _normalized_text_prefix(prefix)
                            if normalized is not None:
                                captured_pre_quant[normalized] = source.take_tensor(pre_key)
                    else:
                        scale = source.take_tensor(prefix + ".weight_scale")
                        params = TensorWiseINT8Layout.Params(
                            scale=scale,
                            orig_dtype=self.dtype,
                            orig_shape=tuple(qdata.shape),
                            is_weight=True,
                            convrot=False,
                        )
                        value = QuantizedTensor(qdata, "TensorWiseINT8Layout", params)
                else:
                    text_part, vision_part = _strip_known_prefixes({key: None})
                    if text_part:
                        normalized = next(iter(text_part))
                        if not _wanted_text_key(normalized, MINIMAX_H3_TEXT_ENCODER_LAYER):
                            continue
                    elif not vision_part:
                        continue
                    value = source.take_tensor(key)
                    if value.is_floating_point() and value.dtype != self.dtype:
                        value = value.to(self.dtype)

                text_part, vision_part = _strip_known_prefixes({key: value})
                for native_key, native_value in text_part.items():
                    if _wanted_text_key(native_key, MINIMAX_H3_TEXT_ENCODER_LAYER):
                        _assign_model_tensor(self.text_model, native_key, native_value)
                        loaded_text.add(native_key)
                for native_key, native_value in vision_part.items():
                    _assign_model_tensor(self.vision_tower, native_key, native_value)
                    loaded_vision.add(native_key)

        expected_text = {
            key for key in self.text_model.state_dict() if not key.startswith("rotary_emb.")
        }
        expected_vision = {
            key for key in self.vision_tower.state_dict() if "rotary_pos_emb" not in key
        }
        if expected_text != loaded_text:
            missing = sorted(expected_text - loaded_text)
            unexpected = sorted(loaded_text - expected_text)
            raise RuntimeError(
                f"text_encoder strict staged load failed: {len(missing)} missing, "
                f"{len(unexpected)} unexpected; examples={missing[:5] + unexpected[:5]}"
            )
        if expected_vision != loaded_vision:
            missing = sorted(expected_vision - loaded_vision)
            unexpected = sorted(loaded_vision - expected_vision)
            raise RuntimeError(
                f"vision strict staged load failed: {len(missing)} missing, "
                f"{len(unexpected)} unexpected; examples={missing[:5] + unexpected[:5]}"
            )
        if source.backend == "pread":
            source.close()
            self._hayate_text_encoder_source = None

        from minimax_video.qwen3vl_text import Qwen3VLTextRotaryEmbedding
        from minimax_video.qwen3vl_vision import Qwen3VLVisionRotaryEmbedding

        text_config = self.text_config
        head_dim = text_config.get(
            "head_dim", text_config["hidden_size"] // text_config["num_attention_heads"]
        )
        rope_scaling = text_config.get("rope_scaling") or {}
        self.text_model.rotary_emb = Qwen3VLTextRotaryEmbedding(
            head_dim=head_dim,
            rope_theta=text_config.get("rope_theta", 1000000.0),
            mrope_section=rope_scaling.get("mrope_section", [24, 20, 20]),
        )
        vision_head_dim = self.vision_config["hidden_size"] // self.vision_config["num_heads"]
        self.vision_tower.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(vision_head_dim // 2)

        installed = 0
        for module_name, scale in captured_pre_quant.items():
            try:
                module = self.text_model.get_submodule(module_name)
            except AttributeError:
                continue
            if not hasattr(module, "weight"):
                continue
            buffer_name = "hayate_awq_pre_quant_scale"
            module.register_buffer(buffer_name, scale, persistent=False)
            original_forward = module.forward

            def awq_forward(this, input_tensor, *args, _original=original_forward, **kwargs):
                awq_scale = getattr(this, buffer_name).to(
                    device=input_tensor.device, dtype=input_tensor.dtype, non_blocking=True
                )
                return _original(input_tensor * awq_scale, *args, **kwargs)

            module.forward = types.MethodType(awq_forward, module)
            installed += 1
        logger.info(
            "HAYATE NVFP4/AWQ conditioner: weight wrappers loaded, %d AWQ scales installed; "
            "Ampere uses full-precision matmul after per-layer dequantization",
            installed,
        )
        self.text_model.eval().requires_grad_(False)
        self.vision_tower.eval().requires_grad_(False)
        old_swap_setting = torch.__future__.get_swap_module_params_on_conversion()
        torch.__future__.set_swap_module_params_on_conversion(True)
        try:
            self.vision_tower.to(self.device)
            self.text_model.embed_tokens.to(self.device)
            if gpu_layers < 0:
                self.text_model.layers.to(self.device)
            elif gpu_layers > 0:
                for layer in list(self.text_model.layers)[:gpu_layers]:
                    layer.to(self.device)
        finally:
            torch.__future__.set_swap_module_params_on_conversion(old_swap_setting)
        for name, tensor in list(self.text_model.named_parameters()) + list(
            self.text_model.named_buffers()
        ):
            if tensor.is_meta:
                raise RuntimeError(f"text_encoder parameter left on meta after load: {name}")
        for name, tensor in list(self.vision_tower.named_parameters()) + list(
            self.vision_tower.named_buffers()
        ):
            if tensor.is_meta:
                raise RuntimeError(f"vision tower parameter left on meta after load: {name}")
        return None

    MiniMaxH3Conditioner._load_weights = load_weights
