from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


_PROBE_SCRIPT = r'''
import importlib.metadata
import json
import sys
import time
import traceback

payload = {"ok": False, "operation_executed": False}
try:
    import torch
    import comfy_kitchen
    from comfy_kitchen.registry import registry
    from comfy_kitchen.tensor import AsymW4A8Int8Layout, QuantizedTensor

    payload.update({
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "comfy_kitchen_version": importlib.metadata.version("comfy-kitchen"),
        "cuda_available": torch.cuda.is_available(),
        "backends": registry.list_backends(),
        "layout_requirements": AsymW4A8Int8Layout.get_requirements(),
    })
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA is not available")

    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    payload["gpu"] = {
        "name": props.name,
        "compute_capability": [props.major, props.minor],
        "total_memory": props.total_memory,
    }
    torch.manual_seed(20260825)
    weight = torch.randn((128, 256), device=device, dtype=torch.bfloat16)
    x = torch.randn((8, 256), device=device, dtype=torch.bfloat16)
    reference = torch.nn.functional.linear(x, weight)
    with registry.use_backend("cuda"):
        quantized = QuantizedTensor.from_float(
            weight,
            "AsymW4A8Int8Layout",
            group_size=16,
            convrot_groupsize=256,
            symmetric=False,
            codebook=True,
        )
        for _ in range(3):
            output = torch.nn.functional.linear(x, quantized)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(20):
            output = torch.nn.functional.linear(x, quantized)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

    numerator = torch.linalg.vector_norm(output.float() - reference.float())
    denominator = torch.linalg.vector_norm(reference.float()).clamp_min(1e-12)
    payload.update({
        "ok": True,
        "operation_executed": True,
        "forced_backend": "cuda",
        "relative_l2": float((numerator / denominator).item()),
        "max_abs_error": float((output.float() - reference.float()).abs().max().item()),
        "average_linear_ms": elapsed * 1000.0 / 20.0,
        "output_shape": list(output.shape),
        "output_dtype": str(output.dtype),
    })
    if len(sys.argv) == 3:
        from hayate.loaders.w4a8_binding import W4A8CheckpointBinding

        checkpoint, layer = sys.argv[1], sys.argv[2]
        actual = W4A8CheckpointBinding(checkpoint).materialize(layer, device="cuda:0")
        actual_input = torch.randn(
            (1, actual.shape[1]), device=device, dtype=actual.dtype
        )
        with registry.use_backend("cuda"):
            actual_output = torch.nn.functional.linear(actual_input, actual)
            actual_reference = torch.nn.functional.linear(actual_input, actual.dequantize())
            for _ in range(3):
                actual_output = torch.nn.functional.linear(actual_input, actual)
            torch.cuda.synchronize()
            actual_started = time.perf_counter()
            for _ in range(10):
                actual_output = torch.nn.functional.linear(actual_input, actual)
            torch.cuda.synchronize()
            actual_elapsed = time.perf_counter() - actual_started
        actual_num = torch.linalg.vector_norm(
            actual_output.float() - actual_reference.float()
        )
        actual_den = torch.linalg.vector_norm(actual_reference.float()).clamp_min(1e-12)
        payload["checkpoint_layer"] = {
            "path": checkpoint,
            "layer": layer,
            "logical_shape": list(actual.shape),
            "storage_shape": list(actual.storage_shape),
            "relative_l2_vs_dequantized": float((actual_num / actual_den).item()),
            "average_linear_ms": actual_elapsed * 1000.0 / 10.0,
            "finite": bool(torch.isfinite(actual_output).all().item()),
        }
except Exception as exc:
    payload["error"] = f"{type(exc).__name__}: {exc}"
    payload["traceback"] = traceback.format_exc()
print(json.dumps(payload, default=str))
'''


@dataclass(frozen=True)
class KernelProbe:
    python: Path
    ok: bool
    operation_executed: bool
    payload: dict
    process_returncode: int
    stderr: str

    def to_dict(self) -> dict:
        return {
            "python": str(self.python),
            "ok": self.ok,
            "operation_executed": self.operation_executed,
            "process_returncode": self.process_returncode,
            "stderr": self.stderr,
            **self.payload,
        }


def probe_w4a8_kernel(
    python: str | Path = sys.executable,
    *,
    checkpoint: str | Path | None = None,
    layer: str | None = None,
    timeout: float = 180.0,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> KernelProbe:
    """Force one W4A8 linear through comfy-kitchen's native CUDA backend.

    The check runs out of process so an extension import or CUDA failure cannot
    poison the long-lived HAYATE process.
    """

    interpreter = Path(python).expanduser().resolve(strict=False)
    if not interpreter.is_file():
        return KernelProbe(
            interpreter,
            False,
            False,
            {"error": f"Python interpreter does not exist: {interpreter}"},
            -1,
            "",
        )
    try:
        command = [str(interpreter), "-c", _PROBE_SCRIPT]
        if checkpoint is not None or layer is not None:
            if checkpoint is None or layer is None:
                return KernelProbe(
                    interpreter,
                    False,
                    False,
                    {"error": "checkpoint and layer must be supplied together"},
                    -1,
                    "",
                )
            command.extend((str(Path(checkpoint).resolve(strict=False)), layer))
        result = runner(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return KernelProbe(interpreter, False, False, {"error": str(exc)}, -1, "")
    payload: dict = {}
    for line in reversed(result.stdout.splitlines()):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            payload = candidate
            break
    if not payload:
        payload = {"error": "kernel probe returned no JSON payload", "stdout": result.stdout}
    return KernelProbe(
        interpreter,
        bool(payload.get("ok")) and result.returncode == 0,
        bool(payload.get("operation_executed")),
        payload,
        result.returncode,
        result.stderr.strip(),
    )
