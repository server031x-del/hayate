"""Install the article's pinned FastH3 engine on Colab; never starts a server."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

COMFY = "10febb01d7be73d1491cf5e5347b5ab8b6c2c09e"
KJNODES = "57105374f47d0fbb49c9c3926fb981702e0a4b5c"
FASTVAE = "b719329e0ecf35f0ae08d241c363ed1e56adbb95"
SOL_URL = "https://github.com/user-attachments/files/31576773/sol_attn_minimax_v5.py"
SOL_SHA = "97c9d56fdc7c9a102e59bff9ac8d79503299514d061892088a03d99dcf415b0c"


def install(root=Path("/content/HAYATE-webui")):
    if sys.platform != "linux" or not Path("/content").is_dir():
        raise RuntimeError("This installer is for a Colab VM only")
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError("FastH3 VSAはSM80以上のCUDA GPUが必要です。CPU/T4/TPUではWebUI確認のみ可能です")
    root = root.resolve()
    runtime = root / "upstream/comfy-fasth3"
    def run(*args, **kwargs):
        subprocess.run([str(arg) for arg in args], check=True, **kwargs)
    def checkout(url, path, revision):
        if not path.exists():
            run("git", "clone", "--no-checkout", url, path)
            run("git", "checkout", "--detach", revision, cwd=path)
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
        if head != revision:
            raise RuntimeError(f"Existing checkout differs: {path}; refusing to overwrite")
    checkout("https://github.com/Kijai/ComfyUI.git", runtime, COMFY)
    marker = runtime / "hayate-ready.json"
    marker.unlink(missing_ok=True)
    run(sys.executable, "-m", "venv", "--system-site-packages", runtime / ".venv")
    python = runtime / ".venv/bin/python"
    from importlib.metadata import version
    constraints = runtime / "hayate-constraints.txt"
    constraints.write_text("\n".join(f"{name}=={version(name)}" for name in ("torch", "torchvision", "torchaudio")))
    kj = runtime / "custom_nodes/ComfyUI-KJNodes"
    fast = runtime / "custom_nodes/ComfyUI-MiniMax-H3-MotionCache-FastVAE"
    checkout("https://github.com/kijai/ComfyUI-KJNodes.git", kj, KJNODES)
    checkout("https://github.com/Mozer/ComfyUI-MiniMax-H3-MotionCache-FastVAE.git", fast, FASTVAE)
    # This VSA branch pins kitchen 0.2.31, before the published VSA kernel.
    # Override only that exact requirement, inside this separate environment.
    requirements = runtime / "hayate-requirements.txt"
    original = (runtime / "requirements.txt").read_text()
    if "comfy-kitchen==0.2.31" not in original:
        raise RuntimeError("Pinned ComfyUI requirements changed")
    requirements.write_text(original.replace("comfy-kitchen==0.2.31", "comfy-kitchen==0.2.33"))
    run(python, "-m", "pip", "install", "-c", constraints, "-r", requirements,
        "-r", kj / "requirements.txt", "-r", fast / "requirements.txt", "websocket-client", "comfy-kitchen==0.2.33")
    with urllib.request.urlopen(SOL_URL, timeout=60) as response:
        source = response.read()
    if hashlib.sha256(source).hexdigest() != SOL_SHA:
        raise RuntimeError("SolAttn node checksum differs")
    (runtime / "custom_nodes/sol_attn_minimax_v5.py").write_bytes(source)
    # Extra paths let the model downloader remain the sole owner of weights.
    import yaml
    models = root / "models"
    (runtime / "hayate-models.yaml").write_text(yaml.safe_dump({"hayate": {
        "base_path": str(models), "diffusion_models": ".", "text_encoders": "text_encoders",
        "vae": ".\nvae"}}))
    run(python, "-c", "import torch, comfy_kitchen as ck; "
        "q=torch.randn(1,128,1,128,device='cuda',dtype=torch.bfloat16); "
        "y=ck.sol_attn(q,q,q,topk_ratio=0.1); "
        "assert torch.isfinite(y).all(); print('SolAttn CUDA kernel OK')", cwd=runtime)
    marker.write_text(json.dumps({"comfy": COMFY, "kjnodes": KJNODES,
                                 "fastvae": FASTVAE, "sol_sha256": SOL_SHA}))
    print("FastH3環境を準備しました。HAYATE設定画面でFastH3構成のモデルを取得してください。")


if __name__ == "__main__":
    install()
