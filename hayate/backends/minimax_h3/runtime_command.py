"""Runtime command helpers for the optional FastH3 backend.

FastVideo's supported NVIDIA install target is Linux or Windows WSL.  HAYATE
itself remains a native Windows application, so the FastH3 adapter accepts a
small, explicit ``wsl://<distribution>/<python>`` URI in addition to a normal
Windows ``python.exe`` path.  Keeping this translation here prevents model and
output paths from being accidentally passed to the wrong operating system and
keeps the normal mayble H3 path unchanged.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
from urllib.parse import unquote, urlparse


_WINDOWS_DRIVE = re.compile(r"^([A-Za-z]):[\\/](.*)$")
_WSL_UNC = re.compile(r"^\\\\wsl(?:\.localhost|\$)\\([^\\]+)\\(.*)$", re.IGNORECASE)

# Only variables that are intentionally set by HAYATE/FastVideo are forwarded
# through ``env``.  In particular, never serialize the complete Windows
# process environment into a WSL command line.
WSL_ENV_ALLOWLIST = frozenset(
    {
        "CUDA_DEVICE_ORDER",
        "CUDA_VISIBLE_DEVICES",
        "FASTVIDEO_ATTENTION_BACKEND",
        "FASTVIDEO_DMD_DENOISING_STEPS",
        "FASTVIDEO_FA4",
        "FASTVIDEO_H3_VSA_PROBE",
        "FASTVIDEO_INFERENCE_TORCH_COMPILE",
        "FASTVIDEO_MINIMAX_H3_FA4_PACKED_VARLEN",
        "FASTVIDEO_MINIMAX_H3_FUSIONS",
        "FASTVIDEO_NVFP4_FA4",
        "FASTVIDEO_STAGE_LOGGING",
        "FASTVIDEO_ULYSSES_A2A",
        "FASTVIDEO_VAE_PARALLEL_DECODE",
        "FASTVIDEO_VAE_PARALLEL_DECODE_STRATEGY",
        "FASTVIDEO_VAE_PARALLEL_ENCODE",
        "FASTVIDEO_VSA_CUTEDSL",
        "FASTVIDEO_VSA_SM100A",
        "HAYATE_FASTH3_PROFILE",
        "HAYATE_FASTH3_REQUIRE_BLACKWELL",
        "HAYATE_FASTH3_MIN_VRAM_BYTES",
        "HAYATE_GPU_INDEX",
        "HAYATE_GPU_RUNTIME_LEASE",
        "HAYATE_GPU_SELECTOR",
        "HAYATE_GPU_UUID",
        "HAYATE_WSL_PYTHONPATH",
        "PYTHONIOENCODING",
        "PYTHONPATH",
    }
)


@dataclass(frozen=True)
class RuntimeSpec:
    """Parsed FastVideo interpreter descriptor."""

    kind: str
    executable: Path | str
    distribution: str | None = None

    @property
    def is_wsl(self) -> bool:
        return self.kind == "wsl"

    @property
    def label(self) -> str:
        if self.is_wsl:
            return f"wsl://{self.distribution}{self.executable}"
        return str(self.executable)


def parse_runtime_spec(value: str | Path | RuntimeSpec) -> RuntimeSpec:
    """Parse a native path or explicit ``wsl://Distro/path`` descriptor."""

    if isinstance(value, RuntimeSpec):
        return value
    raw = str(value).strip()
    if raw.lower().startswith("wsl://"):
        parsed = urlparse(raw)
        distribution = unquote(parsed.netloc).strip()
        executable = unquote(parsed.path)
        if not distribution or not executable.startswith("/"):
            raise ValueError(
                "WSL runtime must use wsl://<distribution>/<absolute-linux-python>"
            )
        return RuntimeSpec("wsl", executable, distribution)
    return RuntimeSpec("local", Path(raw).expanduser().resolve(strict=False))


def runtime_exists(spec: RuntimeSpec) -> bool:
    """Check a native interpreter locally; WSL is checked by its probe."""

    return spec.is_wsl or Path(spec.executable).is_file()


def windows_to_wsl_path(value: str | Path) -> str:
    """Translate a Windows absolute path to the WSL ``/mnt/<drive>`` mount."""

    raw = str(value)
    if raw.startswith("/"):
        return raw
    unc = _WSL_UNC.match(raw)
    if unc:
        # The distribution component is intentionally ignored here; callers
        # already select it in RuntimeSpec and the path under /home or /mnt is
        # what WSL needs.
        return "/" + unc.group(2).replace("\\", "/").lstrip("/")
    drive = _WINDOWS_DRIVE.match(raw)
    if drive:
        rest = drive.group(2).replace("\\", "/")
        return f"/mnt/{drive.group(1).lower()}/{rest}"
    return raw.replace("\\", "/")


def runtime_path(value: str | Path, spec: RuntimeSpec) -> str:
    """Return a path suitable for the selected interpreter."""

    if not spec.is_wsl:
        return str(Path(value).expanduser().resolve(strict=False))
    return windows_to_wsl_path(value)


def build_runtime_command(
    value: str | Path | RuntimeSpec,
    args: Sequence[str],
    *,
    environment: Mapping[str, str] | None = None,
    cwd: str | Path | None = None,
    project_root: str | Path | None = None,
) -> tuple[list[str], str | None]:
    """Build a subprocess command and its runtime working directory.

    The WSL form uses ``/usr/bin/env`` with a small allowlist of variables so
    GPU selection and FastVideo profile switches survive the Windows/WSL
    boundary without leaking unrelated secrets or host-only paths.
    """

    spec = parse_runtime_spec(value)
    if not spec.is_wsl:
        return [str(spec.executable), *map(str, args)], (
            str(Path(cwd).expanduser().resolve(strict=False)) if cwd is not None else None
        )

    command = ["wsl.exe", "--distribution", str(spec.distribution)]
    linux_cwd = runtime_path(cwd, spec) if cwd is not None else None
    if linux_cwd:
        command.extend(["--cd", linux_cwd])
    command.extend(["--", "/usr/bin/env"])
    forwarded = dict(environment or {})
    if project_root is not None:
        project_path = windows_to_wsl_path(project_root)
        current = str(forwarded.get("PYTHONPATH", "")).strip()
        forwarded["PYTHONPATH"] = (
            f"{project_path}:{current}" if current and current != project_path else project_path
        )
    for name in sorted(WSL_ENV_ALLOWLIST):
        raw = forwarded.get(name)
        if raw is None:
            continue
        value_text = str(raw)
        if name == "PYTHONPATH":
            # A host path in an inherited value is not usable inside WSL.  The
            # project root above is the only path HAYATE intentionally adds.
            value_text = ":".join(
                windows_to_wsl_path(part) for part in value_text.split(os.pathsep) if part
            )
        command.append(f"{name}={value_text}")
    command.extend([str(spec.executable), *map(str, args)])
    return command, linux_cwd


def update_runtime_command_environment(
    command: Sequence[str],
    environment: Mapping[str, str],
    *,
    project_root: str | Path | None = None,
) -> list[str]:
    """Refresh allow-listed variables embedded in a WSL command.

    ``wsl.exe`` receives the FastVideo environment as command-line arguments
    (``/usr/bin/env NAME=value ... python ...``).  The WebUI chooses a GPU
    after a plan is built, so changing only ``Popen(env=...)`` would not be
    sufficient for a WSL child on every host.  This helper keeps the original
    command/cwd/arguments intact while replacing only the allow-listed
    assignments.  Native commands are returned unchanged.
    """

    updated = [str(item) for item in command]
    if not updated or Path(updated[0]).name.lower() not in {"wsl", "wsl.exe"}:
        return updated
    try:
        env_marker = updated.index("/usr/bin/env")
    except ValueError:
        return updated

    assignment_end = env_marker + 1
    assignments: dict[str, str] = {}
    while assignment_end < len(updated):
        token = updated[assignment_end]
        if "=" not in token:
            break
        name, value = token.split("=", 1)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            break
        assignments[name] = value
        assignment_end += 1
    if assignment_end >= len(updated):
        return updated

    forwarded = dict(assignments)
    for name in WSL_ENV_ALLOWLIST:
        if name not in environment:
            continue
        value = str(environment[name])
        if name == "PYTHONPATH":
            value = ":".join(
                windows_to_wsl_path(part)
                for part in value.split(os.pathsep)
                if part
            )
            if project_root is not None:
                root = windows_to_wsl_path(project_root)
                if root and root not in value.split(":"):
                    value = f"{root}:{value}" if value else root
        forwarded[name] = value

    refreshed = [
        f"{name}={forwarded[name]}"
        for name in sorted(forwarded)
        if name in WSL_ENV_ALLOWLIST
    ]
    return [*updated[: env_marker + 1], *refreshed, *updated[assignment_end:]]


__all__ = [
    "RuntimeSpec",
    "WSL_ENV_ALLOWLIST",
    "build_runtime_command",
    "update_runtime_command_environment",
    "parse_runtime_spec",
    "runtime_exists",
    "runtime_path",
    "windows_to_wsl_path",
]
