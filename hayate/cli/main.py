from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from rich.console import Console
from rich.table import Table

from hayate import __version__
from hayate.backends.minimax_h3 import ExternalH3GenerationBackend, GenerationRequest
from hayate.benchmark import Benchmark
from hayate.errors import HayateError
from hayate.hardware import HardwareProfile, HardwareProfiler
from hayate.memory import MemoryManager
from hayate.models import ModelRegistry
from hayate.kernels import probe_w4a8_kernel
from hayate.profiles import get_generation_profile
from hayate.runtime import HayateRuntime, ModelRuntimeResult
from hayate.runtime.gpu_lease import GPULease

GIB = 1024**3


def _default_config() -> Path:
    cwd_config = Path.cwd() / "configs" / "models.yaml"
    if cwd_config.is_file():
        return cwd_config
    source_config = Path(__file__).resolve().parents[2] / "configs" / "models.yaml"
    if source_config.is_file():
        return source_config
    return Path(__file__).resolve().parents[1] / "configs" / "models.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hayate",
        description="High-speed AI Yield & Acceleration Technology Engine",
    )
    parser.add_argument("--version", action="version", version=f"HAYATE {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser(
        "inspect", help="profile hardware and inspect registered model headers"
    )
    inspect_parser.add_argument("--config", type=Path, default=None)
    inspect_parser.add_argument("--family", default="minimax_h3")
    inspect_parser.add_argument("--verbose", action="store_true")
    inspect_parser.add_argument("--json", action="store_true", dest="as_json")
    inspect_parser.add_argument("--no-save-benchmark", action="store_true")
    inspect_parser.add_argument("--benchmark-dir", type=Path, default=Path("benchmarks"))

    kernel_parser = subparsers.add_parser(
        "kernel-check", help="force a tiny W4A8 operation through the native CUDA backend"
    )
    kernel_parser.add_argument("--python", type=Path, default=Path(sys.executable))
    kernel_parser.add_argument("--json", action="store_true", dest="as_json")
    kernel_parser.add_argument("--save", type=Path, default=None)
    kernel_parser.add_argument("--model", type=Path, default=None)
    kernel_parser.add_argument("--layer", default=None)

    generate_parser = subparsers.add_parser(
        "generate", help="preflight and run maybleMyers/h3 with HAYATE model overrides"
    )
    generate_parser.add_argument("--prompt", required=True)
    generate_parser.add_argument("--ckpt-dir", type=Path, required=True)
    generate_parser.add_argument("--output", type=Path, required=True)
    generate_parser.add_argument("--config", type=Path, default=None)
    generate_parser.add_argument(
        "--upstream",
        type=Path,
        default=Path(os.environ.get("HAYATE_H3_CHECKOUT", "upstream/h3")),
    )
    generate_parser.add_argument("--python", type=Path, default=Path(sys.executable))
    generate_parser.add_argument(
        "--task", choices=("auto", "t2va", "fl2va", "ref2va"), default="auto"
    )
    generate_parser.add_argument("--image", type=Path, default=None)
    generate_parser.add_argument("--last-image", type=Path, default=None)
    generate_parser.add_argument("--reference", type=Path, action="append", default=[])
    generate_parser.add_argument("--height", type=int, default=None)
    generate_parser.add_argument("--width", type=int, default=None)
    generate_parser.add_argument("--frames", type=int, default=124)
    generate_parser.add_argument("--steps", type=int, default=50)
    generate_parser.add_argument("--seed", type=int, default=20260825)
    generate_parser.add_argument("--blocks-to-swap", type=int, default=49)
    generate_parser.add_argument("--activation-chunk-rows", type=int, default=32768)
    generate_parser.add_argument("--prompt-cache", type=Path, default=None)
    generate_parser.add_argument("--easycache", action="store_true")
    generate_parser.add_argument("--easycache-threshold", type=float, default=0.2)
    generate_parser.add_argument("--easycache-start", type=float, default=0.15)
    generate_parser.add_argument("--easycache-end", type=float, default=0.95)
    generate_parser.add_argument("--easycache-max-consecutive-skips", type=int, default=2)
    generate_parser.add_argument(
        "--pdd-checkpoint",
        type=Path,
        default=None,
        help="Alibaba PAI MiniMax-H3 PDD acceleration checkpoint (mutually exclusive with EasyCache)",
    )
    generate_parser.add_argument(
        "--pdd-adaln-affine",
        type=Path,
        default=None,
        help="AdaLN affine map used to project released PDD adapters onto a pruned transformer",
    )
    generate_parser.add_argument(
        "--vae-tile-size",
        type=int,
        default=256,
        help="video VAE tile size in pixels; values above the validated 256 are experimental",
    )
    generate_parser.add_argument(
        "--attention-backend",
        choices=("sdpa", "sageattn"),
        default="sdpa",
        help="transformer attention backend; sageattn requires a compatible SageAttention build",
    )
    speed_profiles = generate_parser.add_mutually_exclusive_group()
    speed_profiles.add_argument(
        "--rtx3060-fast",
        action="store_true",
        help=(
            "apply the validated RTX 3060 fast profile: 20 points, EasyCache 0.4, "
            "two consecutive skips, 49 swapped blocks, and 32768-row chunks"
        ),
    )
    speed_profiles.add_argument(
        "--rtx3060-fast-sage",
        action="store_true",
        help=(
            "apply the validated RTX 3060 fast profile with SageAttention 2.2; "
            "this is faster but approximate and requires a compatible package"
        ),
    )
    speed_profiles.add_argument(
        "--rtx3060-fast-sage-detail",
        action="store_true",
        help=(
            "apply the RTX 3060 SageAttention detail profile: keep 20 points and "
            "EasyCache 0.4 while protecting the final 15 percent of denoising"
        ),
    )
    speed_profiles.add_argument(
        "--rtx3060-pdd",
        action="store_true",
        help=(
            "apply the safe PDD Acc 8-Step profile with PyTorch SDPA; use the "
            "experimental --rtx3060-pdd-sage only when short-clip output is validated"
        ),
    )
    speed_profiles.add_argument(
        "--rtx3060-pdd-sage",
        action="store_true",
        help="apply experimental PDD Acc 8-Step with SageAttention (8 transformer evaluations)",
    )
    generate_parser.add_argument("--dry-run", action="store_true")
    generate_parser.add_argument("--json", action="store_true", dest="as_json")

    load_parser = subparsers.add_parser(
        "load-check", help="materialize one complete component through the HAYATE/upstream binding"
    )
    load_parser.add_argument("--upstream", type=Path, required=True)
    load_parser.add_argument("--ckpt-dir", type=Path, required=True)
    load_parser.add_argument("--config", type=Path, default=None)
    load_parser.add_argument(
        "--component",
        choices=("transformer", "text_encoder", "video_vae", "audio_vae"),
        required=True,
    )
    load_parser.add_argument("--output", type=Path, default=None)
    load_parser.add_argument("--decode-smoke", action="store_true")
    load_parser.add_argument("--decode-latent-frames", type=int, default=2)
    load_parser.add_argument("--decode-latent-height", type=int, default=8)
    load_parser.add_argument("--decode-latent-width", type=int, default=8)
    load_parser.add_argument("--vae-tile-size", type=int, default=256)
    load_parser.add_argument("--vae-no-tiling", action="store_true")
    load_parser.add_argument("--cudnn-benchmark", action="store_true")

    webui_parser = subparsers.add_parser(
        "webui", help="launch the local HAYATE Studio generation interface"
    )
    webui_parser.add_argument("--host", default="127.0.0.1")
    webui_parser.add_argument("--port", type=int, default=7860)
    webui_parser.add_argument("--open-browser", action="store_true")
    webui_parser.add_argument(
        "--allow-network",
        action="store_true",
        help="allow a non-loopback bind; this exposes local model controls to the network",
    )
    return parser


def _gib(value: int | None) -> str:
    if value is None:
        return "-"
    return f"{value / GIB:.2f} GiB"


def _hardware_table(profile: HardwareProfile) -> Table:
    table = Table(title="Hardware", show_header=False)
    table.add_column("Item", style="cyan")
    table.add_column("Value")
    table.add_row("CPU", profile.cpu)
    table.add_row("System RAM", _gib(profile.system_ram_bytes))
    table.add_row("CUDA driver available", "yes" if profile.cuda_available else "no")
    table.add_row("CUDA driver API max", profile.cuda_driver_api_version or "unknown")
    table.add_row("PyTorch", profile.pytorch_version or "not installed")
    table.add_row("PyTorch CUDA", profile.pytorch_cuda_version or "not available")
    for gpu in profile.gpus:
        selected = " [main v0.1]" if gpu.selected_for_inference else " [detect only]"
        table.add_row(
            f"GPU {gpu.index}",
            f"{gpu.name} | {_gib(gpu.vram_total_bytes)} | sm_{(gpu.compute_capability or '?').replace('.', '')}{selected}",
        )
    return table


def _models_table(results: list[ModelRuntimeResult]) -> Table:
    table = Table(title="Models")
    table.add_column("Role", style="cyan")
    table.add_column("File")
    table.add_column("Header format")
    table.add_column("Storage")
    table.add_column("Loader")
    table.add_column("Status")
    for result in results:
        if result.error:
            table.add_row(
                result.spec.role.value,
                result.spec.path.name,
                "-",
                "-",
                "-",
                f"[red]ERROR[/red]: {result.error}",
            )
            continue
        assert result.inspection is not None and result.loader is not None
        assert result.validation is not None
        status_style = {
            "SUPPORTED": "green",
            "PARTIAL": "yellow",
            "UNSUPPORTED": "red",
        }[result.validation.status.value]
        table.add_row(
            result.spec.role.value,
            result.spec.path.name,
            f"{result.inspection.quantization.format.value} ({result.inspection.quantization.confidence.value})",
            _gib(result.inspection.tensor_storage_bytes),
            type(result.loader).__name__,
            f"[{status_style}]{result.validation.status.value}[/{status_style}]",
        )
    return table


def _benchmark_table(benchmark: Benchmark) -> Table:
    table = Table(title="HAYATE Benchmark")
    table.add_column("Stage", style="cyan")
    table.add_column("Seconds", justify="right")
    table.add_column("Peak process RAM", justify="right")
    table.add_column("Peak sampled VRAM", justify="right")
    for stage in benchmark.stages:
        table.add_row(
            stage.name,
            f"{stage.duration_seconds:.3f}",
            _gib(stage.ram_peak_bytes),
            _gib(stage.vram_peak_bytes),
        )
    return table


def _print_verbose(console: Console, results: list[ModelRuntimeResult]) -> None:
    for result in results:
        if result.inspection is None:
            continue
        console.rule(f"{result.spec.role.value}: tensors")
        console.print(f"Path: {result.inspection.path}")
        console.print(f"Metadata: {json.dumps(result.inspection.metadata, ensure_ascii=False, sort_keys=True)}")
        console.print("Detection evidence: " + "; ".join(result.inspection.quantization.evidence))
        for tensor in result.inspection.tensors:
            console.print(
                f"{tensor.name} | shape={list(tensor.shape)} | dtype={tensor.dtype} | bytes={tensor.storage_bytes}"
            )
        if result.validation:
            console.print("Validation: " + result.validation.reason)
            for requirement in result.validation.requirements:
                console.print(f"  TODO: {requirement}")
        for warning in (*result.inspection.warnings, *result.warnings):
            console.print(f"[yellow]Warning:[/yellow] {warning}")


def run_inspect(args: argparse.Namespace, console: Console) -> int:
    config_path = (args.config or _default_config()).resolve(strict=False)
    memory = MemoryManager()
    memory.reset_peak()
    benchmark = Benchmark(memory)
    with benchmark.stage("hardware_profiling"):
        hardware = HardwareProfiler().profile()
    registry = ModelRegistry.load(config_path)
    runtime = HayateRuntime(registry)
    with benchmark.stage("model_inspection_and_validation"):
        results = runtime.inspect_all(args.family, hardware)
    benchmark_path = None
    if not args.no_save_benchmark:
        benchmark_path = benchmark.save_json(args.benchmark_dir)

    payload = {
        "hayate_version": __version__,
        "config": str(config_path),
        "hardware": hardware.to_dict(),
        "models": [result.to_dict(include_tensors=args.verbose) for result in results],
        "benchmark": benchmark.to_dict(),
        "benchmark_path": str(benchmark_path) if benchmark_path else None,
    }
    if args.as_json:
        console.print_json(json.dumps(payload, ensure_ascii=False))
    else:
        console.print("[bold]HAYATE[/bold]")
        console.print("High-speed AI Yield & Acceleration Technology Engine")
        console.print(_hardware_table(hardware))
        for warning in hardware.warnings:
            console.print(f"[yellow]Hardware note:[/yellow] {warning}")
        console.print(_models_table(results))
        if args.verbose:
            _print_verbose(console, results)
        console.print(_benchmark_table(benchmark))
        if benchmark_path:
            console.print(f"Benchmark JSON: {benchmark_path}")
    return 0 if all(result.error is None for result in results) else 1


def run_kernel_check(args: argparse.Namespace, console: Console) -> int:
    lease = GPULease(owner={"pid": os.getpid(), "kind": "kernel-check"})
    if not lease.acquire():
        owner = lease.busy_owner() or {}
        raise HayateError(f"GPU 0 is already in use by HAYATE (pid={owner.get('pid', '?')})")
    try:
        result = probe_w4a8_kernel(
            args.python,
            checkpoint=args.model,
            layer=args.layer,
        )
    finally:
        lease.release()
    payload = result.to_dict()
    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    if args.as_json:
        console.print_json(json.dumps(payload, ensure_ascii=False))
    else:
        console.print("[bold]W4A8 CUDA kernel check[/bold]")
        console.print(f"Python: {result.python}")
        console.print(f"Native operation: {'PASS' if result.ok else 'FAIL'}")
        if payload.get("gpu"):
            console.print(f"GPU: {payload['gpu'].get('name')}")
        if payload.get("relative_l2") is not None:
            console.print(f"Relative L2: {payload['relative_l2']:.6f}")
            console.print(f"Average tiny linear: {payload['average_linear_ms']:.3f} ms")
        if payload.get("checkpoint_layer"):
            layer = payload["checkpoint_layer"]
            console.print(
                f"Checkpoint layer: {layer['layer']} {layer['logical_shape']} | "
                f"relative L2={layer['relative_l2_vs_dequantized']:.6f} | "
                f"{layer['average_linear_ms']:.3f} ms"
            )
        if payload.get("error"):
            console.print(f"[red]{payload['error']}[/red]")
        if result.stderr:
            console.print(f"[yellow]{result.stderr}[/yellow]")
    return 0 if result.ok and result.operation_executed else 1


def run_generate(args: argparse.Namespace, console: Console) -> int:
    selected_profile = (
        "fast"
        if args.rtx3060_fast
        else "fast_sage"
        if args.rtx3060_fast_sage
        else "fast_sage_detail"
        if args.rtx3060_fast_sage_detail
        else "pdd"
        if args.rtx3060_pdd
        else "pdd_sage"
        if args.rtx3060_pdd_sage
        else None
    )
    if selected_profile is not None:
        profile = get_generation_profile(selected_profile)
        for field in (
            "steps",
            "attention_backend",
            "easycache",
            "easycache_threshold",
            "easycache_start",
            "easycache_end",
            "easycache_max_consecutive_skips",
            "blocks_to_swap",
            "activation_chunk_rows",
            "vae_tile_size",
        ):
            setattr(args, field, getattr(profile, field))
        if profile.pdd:
            args.pdd_checkpoint = args.pdd_checkpoint or Path(
                "models/lora/MiniMax-H3-FL2VA-Acc-8Step.safetensors"
            )
            args.pdd_adaln_affine = args.pdd_adaln_affine or Path(
                "models/lora/adaln_affine.safetensors"
            )
    config_path = (args.config or _default_config()).resolve(strict=False)
    backend = ExternalH3GenerationBackend(
        args.upstream,
        ModelRegistry.load(config_path),
        python=args.python,
    )
    request = GenerationRequest(
        prompt=args.prompt,
        checkpoint_dir=args.ckpt_dir,
        output=args.output,
        task=args.task,
        image_path=args.image,
        last_image_path=args.last_image,
        references=tuple(args.reference),
        height=args.height,
        width=args.width,
        frames=args.frames,
        steps=args.steps,
        seed=args.seed,
        blocks_to_swap=args.blocks_to_swap,
        activation_chunk_rows=args.activation_chunk_rows,
        prompt_cache=args.prompt_cache,
        easycache=args.easycache,
        easycache_threshold=args.easycache_threshold,
        easycache_start=args.easycache_start,
        easycache_end=args.easycache_end,
        easycache_max_consecutive_skips=args.easycache_max_consecutive_skips,
        vae_tile_size=args.vae_tile_size,
        attention_backend=args.attention_backend,
        pdd_checkpoint=args.pdd_checkpoint,
        pdd_adaln_affine=args.pdd_adaln_affine,
    )
    plan = backend.plan(request)
    payload = plan.to_dict()
    if args.as_json:
        console.print_json(json.dumps(payload, ensure_ascii=False))
    else:
        state = "READY" if plan.executable else "BLOCKED"
        console.print(f"[bold]HAYATE generation preflight: {state}[/bold]")
        console.print("Command:")
        console.print("  " + subprocess.list2cmdline(list(plan.command)))
        for issue in plan.issues:
            console.print(f"[red]BLOCKER:[/red] {issue}")
        for warning in plan.warnings:
            console.print(f"[yellow]Warning:[/yellow] {warning}")
    if args.dry_run:
        return 0 if plan.executable else 1
    result = backend.execute(plan)
    console.print(f"Generation finished in {result.duration_seconds:.1f}s (exit {result.returncode})")
    console.print(f"Log: {result.log_path}")
    return result.returncode


def run_load_check(args: argparse.Namespace, console: Console) -> int:
    config = (args.config or _default_config()).resolve(strict=False)
    command = [
        sys.executable,
        "-m",
        "hayate.backends.minimax_h3.load_check",
        "--upstream",
        str(args.upstream),
        "--ckpt-dir",
        str(args.ckpt_dir),
        "--config",
        str(config),
        "--component",
        args.component,
    ]
    if args.output is not None:
        command.extend(("--output", str(args.output)))
    if args.decode_smoke:
        command.append("--decode-smoke")
        command.extend(
            (
                "--decode-latent-frames",
                str(args.decode_latent_frames),
                "--decode-latent-height",
                str(args.decode_latent_height),
                "--decode-latent-width",
                str(args.decode_latent_width),
                "--vae-tile-size",
                str(args.vae_tile_size),
            )
        )
        if args.vae_no_tiling:
            command.append("--vae-no-tiling")
        if args.cudnn_benchmark:
            command.append("--cudnn-benchmark")
    lease = GPULease(owner={"pid": os.getpid(), "kind": f"load-check:{args.component}"})
    if not lease.acquire():
        owner = lease.busy_owner() or {}
        raise HayateError(f"GPU 0 is already in use by HAYATE (pid={owner.get('pid', '?')})")
    try:
        return subprocess.run(command, check=False).returncode
    finally:
        lease.release()


def run_webui_command(args: argparse.Namespace) -> int:
    try:
        from hayate.webui import run_webui
    except ImportError as exc:
        raise HayateError(
            "WebUI dependencies are missing; run `uv sync --extra webui --extra generation`"
        ) from exc
    run_webui(
        host=args.host,
        port=args.port,
        open_browser=args.open_browser,
        allow_network=args.allow_network,
        workspace=Path.cwd(),
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    console = Console()
    try:
        if args.command == "inspect":
            return run_inspect(args, console)
        if args.command == "kernel-check":
            return run_kernel_check(args, console)
        if args.command == "generate":
            return run_generate(args, console)
        if args.command == "load-check":
            return run_load_check(args, console)
        if args.command == "webui":
            return run_webui_command(args)
        parser.error(f"unknown command: {args.command}")
    except (HayateError, OSError, ValueError) as exc:
        console.print(f"[red][HAYATE] {exc}[/red]")
        return 2
    return 2


def entrypoint() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    entrypoint()
