from __future__ import annotations

import json
from pathlib import Path


PATH = Path(__file__).resolve().parents[1] / "colab" / "HAYATE.ipynb"
MARKER = "# HAYATE Studio WebUI on Colab"
nb = json.loads(PATH.read_text(encoding="utf-8"))
WEBUI_MARKERS = ("HAYATE Studio WebUI on Colab", "HAYATE Studio WebUI:")
nb["cells"] = [
    c
    for c in nb["cells"]
    if not any(marker in "".join(c.get("source", [])) for marker in WEBUI_MARKERS)
]
# Keep this notebook focused on the HAYATE Studio WebUI.  The older FastH3 /
# ComfyUI cells remain available in colab/runtime.py for separate experiments,
# but they must not be silently presented as the WebUI setup path.
nb["cells"] = nb["cells"][:2]

top_markdown = """# HAYATE Studio WebUI — Google Colab

無料T4ではWebUIの起動、モデル配置、設定確認までを行います。有料GPUでは
対応するSM80以上のGPUを選択した場合にだけ生成を有効化します。
このPCではサーバーを起動せず、すべてColabのホスト型ランタイム内で実行します。
大容量モデルの取得セルは明示的なフラグを有効にした場合だけ実行します。
"""
if nb["cells"] and nb["cells"][0].get("cell_type") == "markdown":
    nb["cells"][0]["source"] = top_markdown.splitlines(True)

markdown = """## HAYATE Studio WebUI on Colab

The WebUI runs inside the Colab VM. Upload colab-webui-bundle.zip from the
local project, optionally run the opt-in model download cell, run the setup
cell, then use the HTTPS Quick Tunnel URL. The free T4 runtime is for UI and
readiness checks; generation stays disabled until a paid SM80+ runtime is
selected. A separate opt-in FastH3 v1 cell can install FastVideo and download
the approximately 148 GB (about 138 GiB) T2VA snapshot. It is skipped by default.
"""

upload = """#@title 1. GitHubから最新版を取得
import io
import zipfile
import urllib.request
from pathlib import Path
REPOSITORY_REF = 'codex/colab-fast-h3' #@param {type:"string"}
url = 'https://api.github.com/repos/server031x-del/hayate/zipball/' + REPOSITORY_REF
request = urllib.request.Request(url, headers={'User-Agent': 'HAYATE-Colab'})
with urllib.request.urlopen(request, timeout=120) as response:
    payload = response.read()
BUNDLE = Path('/content/colab-webui-bundle.zip')
with zipfile.ZipFile(io.BytesIO(payload)) as source, zipfile.ZipFile(BUNDLE, 'w', zipfile.ZIP_DEFLATED) as target:
    for member in source.infolist():
        parts = Path(member.filename).parts[1:]
        if parts and not member.is_dir():
            target.writestr('/'.join(parts), source.read(member))
print('取得完了:', REPOSITORY_REF)
"""

model_prepare = """# HAYATE Studio WebUI: optionally download and verify its direct model set
# This cell is deliberately opt-in.  The four files are about 29.8 GiB total;
# a free T4 can download them, but it cannot run HAYATE generation.
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path

DOWNLOAD_HAYATE_MODELS = False
ACCEPT_HAYATE_MODEL_LICENSE = False
MODEL_ROOT = Path('/content/hayate-download-test')
HAYATE_MODEL_ASSETS = (
    {
        'repo_id': 'Kijai/MiniMax-H3-experimental',
        'revision': 'f9c521b15b6883fa1c353f86f806ad76311aa5a4',
        'remote_path': 'minimax_h3_fl2va_pruned_w4a8_mixed.safetensors',
        'relative_path': 'minimax_h3_fl2va_pruned_w4a8_mixed.safetensors',
        'size_bytes': 12540858008,
        'sha256': '01aa7b92c007c599890461c325f9b7e3c96fb06c36f242f95b62f7f20e538dec',
    },
    {
        'repo_id': 'Comfy-Org/MiniMax-H3',
        'revision': '4cc1d817b6184899b41293954329f576cb5ae86b',
        'remote_path': 'text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors',
        'relative_path': 'text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors',
        'size_bytes': 15687142551,
        'sha256': '35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6',
    },
    {
        'repo_id': 'Kijai/MiniMax-H3-experimental',
        'revision': 'f9c521b15b6883fa1c353f86f806ad76311aa5a4',
        'remote_path': 'minimax_h3_video_vae_int8_convrot.safetensors',
        'relative_path': 'minimax_h3_video_vae_int8_convrot.safetensors',
        'size_bytes': 3171670912,
        'sha256': '9bb2d96f218c76babd85e0611b85ca8fb330a90546c01a0005e8a58a59593410',
    },
    {
        'repo_id': 'Comfy-Org/MiniMax-H3',
        'revision': '4cc1d817b6184899b41293954329f576cb5ae86b',
        'remote_path': 'vae/minimax_h3_audio_vae_fp32.safetensors',
        'relative_path': 'vae/minimax_h3_audio_vae_fp32.safetensors',
        'size_bytes': 605254808,
        'sha256': '8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48',
    },
)

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

required_bytes = sum(item['size_bytes'] for item in HAYATE_MODEL_ASSETS)
print('HAYATE_MODEL_SET_GIB', round(required_bytes / 2**30, 2), flush=True)
if not DOWNLOAD_HAYATE_MODELS:
    print('HAYATE_MODEL_DOWNLOAD_SKIPPED', 'Set DOWNLOAD_HAYATE_MODELS=True after reviewing the model terms.', flush=True)
else:
    if not ACCEPT_HAYATE_MODEL_LICENSE:
        raise RuntimeError('Set ACCEPT_HAYATE_MODEL_LICENSE=True only after reviewing the repository model terms.')
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'huggingface_hub'], check=True)
    from huggingface_hub import hf_hub_download

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    missing_bytes = 0
    for item in HAYATE_MODEL_ASSETS:
        target = MODEL_ROOT / item['relative_path']
        if target.is_file():
            if target.stat().st_size == item['size_bytes'] and sha256_file(target) == item['sha256']:
                print('MODEL_REUSE', target, flush=True)
                continue
            raise RuntimeError(f'Existing model does not match the audited manifest; refusing to overwrite: {target}')
        if target.exists():
            raise RuntimeError(f'Model target is not a regular file: {target}')
        missing_bytes += item['size_bytes']
    free_bytes = shutil.disk_usage(MODEL_ROOT).free
    reserve_bytes = 8 * 2**30
    if free_bytes < missing_bytes + reserve_bytes:
        raise RuntimeError(
            f'Insufficient Colab disk: need {(missing_bytes + reserve_bytes) / 2**30:.2f} GiB, '
            f'free {free_bytes / 2**30:.2f} GiB'
        )
    for item in HAYATE_MODEL_ASSETS:
        target = MODEL_ROOT / item['relative_path']
        if target.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        downloaded = Path(hf_hub_download(
            repo_id=item['repo_id'],
            filename=item['remote_path'],
            revision=item['revision'],
            local_dir=str(MODEL_ROOT),
            token=False,
        ))
        if downloaded.resolve() != target.resolve():
            raise RuntimeError(f'Unexpected Hugging Face destination: {downloaded}')
        if downloaded.stat().st_size != item['size_bytes']:
            raise RuntimeError(f'Size mismatch: {item["remote_path"]}')
        digest = sha256_file(downloaded)
        if digest != item['sha256']:
            raise RuntimeError(f'SHA256 mismatch: {item["remote_path"]}')
        print('MODEL_VERIFIED', downloaded, flush=True)
"""

fasth3_prepare = """# HAYATE Studio WebUI: optionally prepare the official FastH3 v1 route
# This is separate from the normal H3 model set and is deliberately opt-in.
# The official snapshot is approximately 148 GB (about 138 GiB) and is T2VA-only. Free T4
# runtimes can inspect the WebUI, but cannot run this route.
import shutil
import subprocess
import sys
from pathlib import Path

INSTALL_FASTH3_RUNTIME = False
DOWNLOAD_FASTH3_MODEL = False
ACCEPT_FASTH3_MODEL_LICENSE = False
FASTH3_MODEL_ROOT = Path('/content/HAYATE-webui/models/fastvideo')
FASTH3_REPOSITORY = 'FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree'
FASTH3_REVISION = '5ea076f35b84da4c3c82217112fa733d8eea2ae1'
FASTH3_SIZE_BYTES = 147835483860

print('FASTH3_REPOSITORY', FASTH3_REPOSITORY, flush=True)
print('FASTH3_REVISION', FASTH3_REVISION, flush=True)
print('FASTH3_MODEL_GIB', round(FASTH3_SIZE_BYTES / 2**30, 2), flush=True)
if not INSTALL_FASTH3_RUNTIME and not DOWNLOAD_FASTH3_MODEL:
    print('FASTH3_PREPARE_SKIPPED', 'Set the explicit flags in this cell to enable it.', flush=True)
else:
    if not ACCEPT_FASTH3_MODEL_LICENSE:
        raise RuntimeError('Set ACCEPT_FASTH3_MODEL_LICENSE=True only after reviewing the MiniMax H3 terms.')
    if INSTALL_FASTH3_RUNTIME:
        # Keep Colab's already-selected torch build. FastVideo 0.2.1 and its
        # published kernel are the pinned API used by HAYATE's adapter.
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-q',
            'fastvideo==0.2.1', 'fastvideo-kernel==0.3.5',
        ], check=True)
    if DOWNLOAD_FASTH3_MODEL:
        FASTH3_MODEL_ROOT.mkdir(parents=True, exist_ok=True)
        free_bytes = shutil.disk_usage(FASTH3_MODEL_ROOT).free
        reserve_bytes = 8 * 2**30
        if free_bytes < FASTH3_SIZE_BYTES + reserve_bytes:
            raise RuntimeError(
                f'Insufficient Colab disk for FastH3: need {(FASTH3_SIZE_BYTES + reserve_bytes) / 2**30:.2f} GiB, '
                f'free {free_bytes / 2**30:.2f} GiB'
            )
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'huggingface_hub'], check=True)
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=FASTH3_REPOSITORY,
            revision=FASTH3_REVISION,
            local_dir=str(FASTH3_MODEL_ROOT),
            max_workers=2,
            token=False,
        )
        print('FASTH3_MODEL_READY', FASTH3_MODEL_ROOT, flush=True)
"""

setup = """# HAYATE Studio WebUI: install, place the VM-local model set, and start
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import urllib.error
import zipfile
from pathlib import Path

BUNDLE = Path(globals().get('BUNDLE', '/content/colab-webui-bundle.zip'))
if not BUNDLE.is_file():
    raise RuntimeError('Run the GitHub download cell first')
WEBUI_ROOT = Path('/content/HAYATE-webui')
if 'HAYATE_WEBUI_PROCESS' in globals() and HAYATE_WEBUI_PROCESS.poll() is None:
    HAYATE_WEBUI_PROCESS.terminate()
    try:
        HAYATE_WEBUI_PROCESS.wait(timeout=10)
    except subprocess.TimeoutExpired:
        HAYATE_WEBUI_PROCESS.kill()
WEBUI_ROOT.mkdir(parents=True, exist_ok=True)

# Extract only paths contained by the target directory (ZIP-slip guard).
with zipfile.ZipFile(BUNDLE) as archive:
    root_resolved = WEBUI_ROOT.resolve()
    for member in archive.infolist():
        target = (WEBUI_ROOT / member.filename).resolve()
        if root_resolved not in target.parents and target != root_resolved:
            raise RuntimeError(f'Unsafe bundle path: {member.filename}')
    archive.extractall(WEBUI_ROOT)

subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '-q', '-e', '.[webui]'],
    cwd=WEBUI_ROOT,
    check=True,
)
for relative in (
    'models', 'models/minimax-h3-snapshot', 'models/text_encoders',
    'models/vae', 'models/lora', 'models/fastvideo', 'outputs',
    'outputs/prompt_cache', 'data/webui', 'upstream', 'inputs',
):
    (WEBUI_ROOT / relative).mkdir(parents=True, exist_ok=True)

# Keep the bytes inside HAYATE's own models tree.  The compatibility links in
# MODEL_ROOT avoid breaking earlier notebook cells without making a second
# 51 GiB copy.  ModelSetupService can then verify files without resolving a
# link outside its allowed models directory.
MODEL_ROOT = Path('/content/hayate-download-test')
MODEL_LINKS = {
    MODEL_ROOT / 'minimax_h3_fl2va_pruned_w4a8_mixed.safetensors': WEBUI_ROOT / 'models' / 'minimax_h3_fl2va_pruned_w4a8_mixed.safetensors',
    MODEL_ROOT / 'text_encoders' / 'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors': WEBUI_ROOT / 'models' / 'text_encoders' / 'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors',
    MODEL_ROOT / 'minimax_h3_video_vae_int8_convrot.safetensors': WEBUI_ROOT / 'models' / 'minimax_h3_video_vae_int8_convrot.safetensors',
    MODEL_ROOT / 'vae' / 'minimax_h3_audio_vae_fp32.safetensors': WEBUI_ROOT / 'models' / 'vae' / 'minimax_h3_audio_vae_fp32.safetensors',
}
placed = []
for source, target in MODEL_LINKS.items():
    source_resolved = source.resolve(strict=False) if source.is_symlink() else None
    if source.is_symlink() and source_resolved == target.resolve(strict=False) and target.is_file():
        placed.append(str(target.relative_to(WEBUI_ROOT)))
        print('MODEL_REUSE', target, flush=True)
        continue
    if target.is_symlink() or target.is_file():
        target.unlink()
    elif target.exists():
        raise RuntimeError(f'Unexpected model target: {target}')
    if not source.is_file() and not source.is_symlink():
        print('MODEL_MISSING', source, flush=True)
        continue
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_symlink():
        resolved = source.resolve(strict=False)
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        shutil.move(str(resolved), str(target))
        source.unlink()
    else:
        shutil.move(str(source), str(target))
    source.parent.mkdir(parents=True, exist_ok=True)
    source.symlink_to(target)
    placed.append(str(target.relative_to(WEBUI_ROOT)))

# Pin the audited maybleMyers/h3 checkout used by the direct HAYATE route.
UPSTREAM = WEBUI_ROOT / 'upstream' / 'h3'
UPSTREAM_REPO = 'https://github.com/maybleMyers/h3'
UPSTREAM_COMMIT = '94220c1fdf14d6d9d40be06fb99f55b27c0d9024'
UPSTREAM.parent.mkdir(parents=True, exist_ok=True)
if UPSTREAM.exists() and not (UPSTREAM / '.git').exists():
    if any(UPSTREAM.iterdir()):
        raise RuntimeError(f'upstream path is non-empty and not a git checkout: {UPSTREAM}')
    UPSTREAM.rmdir()
if not UPSTREAM.exists():
    subprocess.run(['git', 'clone', '--no-checkout', UPSTREAM_REPO, str(UPSTREAM)], check=True)
current_commit = subprocess.check_output(
    ['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True
).strip()
if current_commit != UPSTREAM_COMMIT or not (UPSTREAM / 'minimax_engine').is_dir():
    subprocess.run(
        ['git', '-C', str(UPSTREAM), 'fetch', '--depth', '1', 'origin', UPSTREAM_COMMIT],
        check=True,
    )
    subprocess.run(
        ['git', '-C', str(UPSTREAM), 'checkout', '--detach', UPSTREAM_COMMIT],
        check=True,
    )
(WEBUI_ROOT / 'upstream' / 'h3.lock.json').write_text(
    json.dumps({
        'name': 'maybleMyers/h3',
        'repository': UPSTREAM_REPO,
        'commit': UPSTREAM_COMMIT,
        'audited_at': '2026-08-25',
        'engine_root': 'minimax_engine',
    }, ensure_ascii=False, indent=2) + chr(10),
    encoding='utf-8',
)

# The requested target is HAYATE Studio.  Stop any earlier temporary ComfyUI
# process/tree from this VM without touching the shared model bytes.
comfy = globals().get('HAYATE_COMFY_PROCESS')
if comfy is not None and comfy.poll() is None:
    comfy.terminate()
    try:
        comfy.wait(timeout=15)
    except subprocess.TimeoutExpired:
        comfy.kill()
# Keep previous ComfyUI files and user data intact.

log_path = WEBUI_ROOT / 'data' / 'webui' / 'colab-webui.log'
log = log_path.open('a', encoding='utf-8')
env = os.environ.copy()
env['PYTHONUNBUFFERED'] = '1'
HAYATE_WEBUI_PROCESS = subprocess.Popen(
    [sys.executable, '-m', 'hayate', 'webui', '--host', '0.0.0.0', '--port', '7860', '--allow-network'],
    cwd=WEBUI_ROOT,
    env=env,
    stdout=log,
    stderr=subprocess.STDOUT,
)
ready = False
bootstrap = None
for _ in range(90):
    try:
        with urllib.request.urlopen('http://127.0.0.1:7860/api/bootstrap', timeout=2) as response:
            bootstrap = json.loads(response.read().decode('utf-8'))
            ready = response.status == 200
            break
    except Exception:
        if HAYATE_WEBUI_PROCESS.poll() is not None:
            break
        time.sleep(1)
print('HAYATE_WEBUI_LISTEN 0.0.0.0:7860', flush=True)
print('HAYATE_WEBUI_READY', ready, 'PID', HAYATE_WEBUI_PROCESS.pid, flush=True)
print('HAYATE_WEBUI_LOG', log_path, flush=True)
print('HAYATE_MODEL_FILES', placed, flush=True)
if bootstrap is not None:
    print('HAYATE_BOOTSTRAP_TITLE', bootstrap.get('title'), flush=True)
    print('HAYATE_READINESS', bootstrap.get('readiness'), flush=True)

def post_json(path, payload):
    data = json.dumps(payload).encode('utf-8')
    request = urllib.request.Request(
        'http://127.0.0.1:7860' + path,
        data=data,
        method='POST',
        headers={'Content-Type': 'application/json', 'X-HAYATE-UI': '1'},
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.loads(response.read().decode('utf-8'))

# Materialize and hash-check the small checkpoint support set.  Large weights
# remain in place and are verified by the dedicated download/verification cell.
try:
    post_json('/api/models/setup/prepare', {})
    post_json('/api/models/setup/apply-standard', {})
    support = post_json(
        '/api/models/setup/download',
        {'asset_id': 'checkpoint_support', 'license_accepted': True},
    )
    support_id = support.get('id')
    if support_id:
        for _ in range(180):
            with urllib.request.urlopen(
                'http://127.0.0.1:7860/api/models/setup/downloads', timeout=10
            ) as response:
                downloads = json.loads(response.read().decode('utf-8')).get('downloads', {})
            row = downloads.get(support_id, {})
            if row.get('status') in {'completed', 'failed'}:
                print('HAYATE_SUPPORT_STATUS', row.get('status'), row.get('message'), flush=True)
                if row.get('status') == 'failed':
                    raise RuntimeError(row.get('message'))
                break
            time.sleep(1)
except urllib.error.HTTPError as exc:
    print('HAYATE_SUPPORT_HTTP_ERROR', exc.code, exc.read().decode('utf-8', errors='replace'), flush=True)
with urllib.request.urlopen('http://127.0.0.1:7860/api/models/setup', timeout=15) as response:
    model_status = json.loads(response.read().decode('utf-8'))
print('HAYATE_MODEL_STATUS', [(item['id'], item['status'], item['exists'], item['verified']) for item in model_status.get('assets', [])], flush=True)
"""

tunnel = """# HAYATE Studio WebUI: publish the VM-local port through HTTPS
import re
import stat
import subprocess
import time
import urllib.request
from pathlib import Path

cloudflared = Path('/content/cloudflared')
if not cloudflared.exists():
    urllib.request.urlretrieve(
        'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64',
        cloudflared,
    )
    cloudflared.chmod(cloudflared.stat().st_mode | stat.S_IXUSR)
tunnel_log = Path('/content/hayate-tunnel.log')
if 'HAYATE_WEBUI_TUNNEL_PROCESS' not in globals() or HAYATE_WEBUI_TUNNEL_PROCESS.poll() is not None:
    with tunnel_log.open('w') as log:
        HAYATE_WEBUI_TUNNEL_PROCESS = subprocess.Popen(
            [str(cloudflared), 'tunnel', '--no-autoupdate', '--url', 'http://127.0.0.1:7860'],
            stdout=log, stderr=subprocess.STDOUT,
        )
HAYATE_WEBUI_TUNNEL_URL = None
deadline = time.monotonic() + 60
while time.monotonic() < deadline:
    contents = tunnel_log.read_text(errors='replace') if tunnel_log.exists() else ''
    match = re.search(r'https://[a-z0-9-]+\\.trycloudflare\\.com', contents)
    if match:
        HAYATE_WEBUI_TUNNEL_URL = match.group(0)
        break
    if HAYATE_WEBUI_TUNNEL_PROCESS.poll() is not None:
        raise RuntimeError(contents[-2000:])
    time.sleep(1)
if not HAYATE_WEBUI_TUNNEL_URL:
    raise RuntimeError('Tunnel URL unavailable; rerun this cell. Log: ' + str(tunnel_log))
from IPython.display import display, Markdown
display(Markdown('[HAYATE Studioを開く](' + HAYATE_WEBUI_TUNNEL_URL + ')'))
print(HAYATE_WEBUI_TUNNEL_URL)

"""

stop = """# HAYATE Studio WebUI: stop only the Colab VM processes created by this notebook
STOP_SERVICES = False #@param {type:"boolean"}
for name in (('HAYATE_WEBUI_TUNNEL_PROCESS', 'HAYATE_WEBUI_PROCESS') if STOP_SERVICES else ()):
    process = globals().get(name)
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except Exception:
            process.kill()
print('停止完了' if STOP_SERVICES else '停止をスキップ（終了時にSTOP_SERVICESを有効にしてください）', flush=True)
"""


def code(source: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(True)}


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}

nb["cells"] = [
    md("# HAYATE Studio — Colab\n品質と速度のバランスを重視したWebUIです。上から順に実行してください。\n\n- 無料環境は起動・設定確認用。生成は対応GPUとモデルが必要です。課金だけでGPU互換性は保証されません。\n- モデル取得は約30 GiB。ライセンス確認後にフラグを有効にしてください。\n- VM終了でモデル・出力は消えます。必要な出力は終了前に保存してください。\n- 公開URLを知る人はWebUIへアクセスできます。共有範囲に注意してください。"),
    code(upload),
    md("## 2. モデル取得（任意）\n既存ファイルは検証して再利用します。初回はダウンロードに時間がかかります。"),
    code(model_prepare),
    md("## 3. WebUIを起動・更新\nソースを更新し、モデル・設定・出力を保持します。標準は『高速・画質優先』です。"),
    code(setup),
    md("## 4. ブラウザで開く\n表示されるHTTPS URLを開いてください。Colab VMのlocalhostへ直接アクセスする必要はありません。"),
    code(tunnel),
    md("## 5. 終了時のみ実行\nWebUIと公開トンネルを停止します。"),
    code(stop),
]
PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"updated {PATH} cells={len(nb['cells'])}")
