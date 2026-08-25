from __future__ import annotations

from typing import Any

from hayate.errors import LoaderNotReadyError
from hayate.loaders.base import BaseModelLoader, LoaderStatus, LoaderValidation
from hayate.models.types import QuantizationFormat


class NVFP4AWQLoader(BaseModelLoader):
    format_name = "NVFP4 + AWQ"

    def validate(self, hardware: Any | None = None) -> LoaderValidation:
        report = self.inspect()
        if report.quantization.format is not QuantizationFormat.NVFP4_AWQ:
            return LoaderValidation(
                LoaderStatus.UNSUPPORTED,
                "header evidence does not identify both NVFP4 and AWQ",
                warnings=("the loader will not infer this format from the filename",),
            )
        tensors = {tensor.name.lower(): tensor for tensor in report.tensors}
        packed_weights = [
            name for name, tensor in tensors.items() if name.endswith(".weight") and tensor.dtype == "U8"
        ]
        primary_scales = [
            name
            for name, tensor in tensors.items()
            if name.endswith(".weight_scale") and tensor.dtype.startswith("F8_")
        ]
        secondary_scales = [name for name in tensors if name.endswith(".weight_scale_2")]
        markers = [name for name in tensors if name.endswith(".comfy_quant")]
        pre_quant = [name for name in tensors if name.endswith(".pre_quant_scale")]
        complete_groups = sum(
            1
            for name in packed_weights
            if name[: -len(".weight")] + ".weight_scale" in tensors
            and name[: -len(".weight")] + ".weight_scale_2" in tensors
            and name[: -len(".weight")] + ".comfy_quant" in tensors
        )
        complete = complete_groups >= 8 and len(pre_quant) > 0
        return LoaderValidation(
            LoaderStatus.SUPPORTED if complete else LoaderStatus.PARTIAL,
            (
                "NVFP4/AWQ layout is complete; mmap staged conditioner binding is enabled"
                if complete
                else "NVFP4/AWQ header recognized but packed groups are incomplete"
            ),
            requirements=(() if complete else ("provide complete NVFP4 scale/marker groups",)),
            warnings=("RTX 3060 uses exact dequantized BF16 matmul, not native NVFP4 tensor cores",),
            details={
                "packed_u8_weight_count": len(packed_weights),
                "fp8_primary_scale_count": len(primary_scales),
                "fp32_secondary_scale_count": len(secondary_scales),
                "comfy_quant_marker_count": len(markers),
                "awq_pre_quant_scale_count": len(pre_quant),
                "complete_nvfp4_groups": complete_groups,
                "rtx_3060_execution": "FULL_STRICT_LOAD_PASS / DEQUANTIZED_BF16_REAL_LAYER_PASS",
                "fallback": "packed layer streaming + AWQ pre_quant_scale + BF16 matmul",
            },
        )

    def load(self, device: str) -> Any:
        validation = self.validate()
        if validation.status is not LoaderStatus.SUPPORTED:
            raise LoaderNotReadyError(f"{validation.reason}. Target device: {device}")
        from hayate.loaders.nvfp4_awq_binding import NVFP4AWQCheckpointBinding

        self._loaded = NVFP4AWQCheckpointBinding(self.spec.path)
        return self._loaded
