from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=("nvfp4_awq", "int8_convrot"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--layer", required=True)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    import torch

    torch.manual_seed(20260825)

    if args.format == "nvfp4_awq":
        from hayate.loaders import NVFP4AWQCheckpointBinding

        logical_dtype = args.dtype or "bfloat16"
        weight = NVFP4AWQCheckpointBinding(args.model).materialize(
            args.layer, device="cuda:0", orig_dtype=logical_dtype
        )
    else:
        from hayate.loaders import INT8ConvRotCheckpointBinding

        logical_dtype = args.dtype or "float32"
        weight = INT8ConvRotCheckpointBinding(args.model).materialize(
            args.layer, device="cuda:0", orig_dtype=logical_dtype
        )
    input_tensor = torch.randn(
        (1, weight.shape[1]), device="cuda:0", dtype=weight.dtype
    )
    reference = torch.nn.functional.linear(input_tensor, weight.dequantize())
    for _ in range(3):
        output = torch.nn.functional.linear(input_tensor, weight)
    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(args.iterations):
        output = torch.nn.functional.linear(input_tensor, weight)
    torch.cuda.synchronize()
    duration = time.perf_counter() - started
    relative_l2 = torch.linalg.vector_norm(output.float() - reference.float()) / torch.linalg.vector_norm(
        reference.float()
    ).clamp_min(1e-12)
    payload = {
        "format": args.format,
        "model": str(args.model.resolve()),
        "layer": args.layer,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "logical_shape": list(weight.shape),
        "storage_shape": list(weight.storage_shape),
        "logical_dtype": str(weight.dtype),
        "finite": bool(torch.isfinite(output).all().item()),
        "relative_l2_vs_dequantized": float(relative_l2.item()),
        "average_linear_ms": duration * 1000.0 / args.iterations,
        "iterations": args.iterations,
        "seed": 20260825,
    }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    return 0 if payload["finite"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
