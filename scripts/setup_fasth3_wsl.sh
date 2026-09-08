#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Prepare the isolated FastVideo runtime used by HAYATE's optional FastH3
# backend. The model is deliberately not downloaded unless
# --download-model is supplied: the official preview is roughly 148 GB (about 138 GiB) and
# should never be fetched as a side effect of a normal HAYATE setup.

set -Eeuo pipefail

REPOSITORY="FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree"
REVISION="5ea076f35b84da4c3c82217112fa733d8eea2ae1"
VENV_DIR="${HOME}/hayate-fasth3-venv"
MODEL_DIR=""
DOWNLOAD_MODEL=0

usage() {
  cat <<'EOF'
Usage: setup_fasth3_wsl.sh [options]

Options:
  --venv DIR          isolated Python environment (default: ~/hayate-fasth3-venv)
  --model-dir DIR    FastVideo snapshot directory (Linux/WSL path)
  --download-model   download the pinned ~148 GB official snapshot
  -h, --help         show this help

The model download is opt-in. The command checks free space first and can be
re-run safely; huggingface_hub resumes unchanged files from its local cache.
EOF
}

while (($#)); do
  case "$1" in
    --venv)
      (($# >= 2)) || { echo "--venv requires a path" >&2; exit 2; }
      VENV_DIR="$2"
      shift 2
      ;;
    --model-dir)
      (($# >= 2)) || { echo "--model-dir requires a path" >&2; exit 2; }
      MODEL_DIR="$2"
      shift 2
      ;;
    --download-model)
      DOWNLOAD_MODEL=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

UV_BIN="${HAYATE_UV_BIN:-$(command -v uv 2>/dev/null || true)}"
if [[ -z "${UV_BIN}" && -x "${HOME}/.local/bin/uv" ]]; then
  UV_BIN="${HOME}/.local/bin/uv"
fi
if [[ -z "${UV_BIN}" && -x "/usr/local/bin/uv" ]]; then
  UV_BIN="/usr/local/bin/uv"
fi
[[ -n "${UV_BIN}" ]] || {
  echo "uv is required. Install it from https://docs.astral.sh/uv/ and retry." >&2
  exit 2
}

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo "Creating isolated Python 3.12 environment: ${VENV_DIR}"
  "${UV_BIN}" venv --python 3.12 --seed "${VENV_DIR}"
fi

echo "Installing the pinned FastVideo runtime (CUDA 13.0 wheels)..."
UV_TORCH_BACKEND=cu130 "${UV_BIN}" pip install \
  --python "${VENV_DIR}/bin/python" \
  "fastvideo==0.2.1" \
  "fastvideo-kernel==0.3.5" \
  "torch==2.12.0+cu130" \
  "torchvision==0.27.0+cu130" \
  "torchaudio==2.11.0+cu130"

echo "Running the FastH3 runtime probe..."
if [[ ! -f "/usr/include/python3.12/Python.h" ]]; then
  echo "Note: Python.h is not installed. If the Triton probe fails, run as WSL root:"
  echo "  apt-get update && apt-get install -y python3.12-dev"
fi
CUDA_DEVICE_ORDER=PCI_BUS_ID "${VENV_DIR}/bin/python" -c '
import importlib.metadata, importlib.util, json, torch
checks = {
    "fastvideo": importlib.util.find_spec("fastvideo") is not None,
    "api": importlib.util.find_spec("fastvideo.api") is not None,
    "minimax_h3": importlib.util.find_spec("fastvideo.models.dits.minimax_h3") is not None,
    "vsa_kernel": importlib.util.find_spec("fastvideo_kernel") is not None,
    "fastvideo_version": importlib.metadata.version("fastvideo"),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "cuda": bool(torch.cuda.is_available()),
    "gpu_capabilities": [".".join(map(str, torch.cuda.get_device_capability(i))) for i in range(torch.cuda.device_count())],
}
checks["available"] = all(checks[key] for key in ("fastvideo", "api", "minimax_h3", "cuda"))
print(json.dumps(checks))
if not checks["available"]:
    raise SystemExit(1)
'

if ((DOWNLOAD_MODEL)); then
  [[ -n "${MODEL_DIR}" ]] || {
    echo "--download-model requires --model-dir; choose a drive with at least 160 GiB free." >&2
    exit 2
  }
  mkdir -p "${MODEL_DIR}"
  free_bytes="$("${VENV_DIR}/bin/python" -c 'import shutil,sys; print(shutil.disk_usage(sys.argv[1]).free)' "${MODEL_DIR}")"
  # Keep a safety margin for the HF cache and JSON metadata. This is a guard,
  # not a claim about exact model size.
  minimum_bytes=$((160 * 1024 * 1024 * 1024))
  if ((free_bytes < minimum_bytes)); then
    echo "Insufficient free space at ${MODEL_DIR}: ${free_bytes} bytes; need at least ${minimum_bytes}." >&2
    exit 2
  fi
  echo "Downloading ${REPOSITORY}@${REVISION} into ${MODEL_DIR} (resumable)..."
  "${VENV_DIR}/bin/python" - "${MODEL_DIR}" "${REPOSITORY}" "${REVISION}" <<'PY'
import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

target = Path(sys.argv[1]).expanduser().resolve()
repo_id = sys.argv[2]
revision = sys.argv[3]
target.mkdir(parents=True, exist_ok=True)
snapshot_download(
    repo_id=repo_id,
    revision=revision,
    local_dir=str(target),
    max_workers=2,
    # Use an explicitly supplied token when the repository requires an
    # accepted license; otherwise let huggingface_hub use its local login.
    token=os.environ.get("HF_TOKEN") or None,
)
print(f"snapshot ready: {target}")
PY
fi

echo ""
echo "FastH3 runtime ready. Configure HAYATE FastVideo Python as:"
echo "  wsl://${WSL_DISTRO_NAME:-<distribution>}/${VENV_DIR#/}/bin/python"
if [[ -n "${MODEL_DIR}" ]]; then
  echo "FastVideo model directory: ${MODEL_DIR}"
fi
