"""Install the direct H3 backend into Colab's WebUI Python without starting it."""
from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import subprocess
import sys


def install(root: Path = Path("/content/HAYATE-webui"), *, sageattention: bool = False) -> None:
    if sys.platform != "linux" or not Path("/content").is_dir():
        raise RuntimeError("This installer is for a Colab VM only")
    if not (root / "pyproject.toml").is_file():
        raise RuntimeError(f"HAYATE checkout is missing: {root}")
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError("ネイティブH3にはSM80以上のCUDA GPUが必要です。T4/TPUでは起動確認のみ可能です")
    constraints = root / "data" / "webui" / "native-torch-constraints.txt"
    constraints.parent.mkdir(parents=True, exist_ok=True)
    pinned = []
    for package in ("torch", "torchvision", "torchaudio"):
        try:
            pinned.append(f"{package}=={version(package)}")
        except PackageNotFoundError:
            pass
    constraints.write_text("\n".join(pinned) + "\n", encoding="utf-8")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-c", str(constraints), "-e", ".[generation]"],
        cwd=root, check=True,
    )
    if sageattention:
        env = os.environ.copy()
        env.setdefault("MAX_JOBS", "4")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "--no-build-isolation",
             "-c", str(constraints), "sageattention==2.2.0"],
            cwd=root, env=env, check=True,
        )
        subprocess.run([sys.executable, "-c", "import sageattention"], check=True)
    print("ネイティブH3の依存環境を準備しました。WebUIの設定でA100構成を取得・適用してください。")


if __name__ == "__main__":
    install()
