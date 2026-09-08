"""Notebook-controlled FastH3 worker. No public tunnel or local-PC execution."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import uuid

ROOT = Path('/content/hayate-runtime')
COMFY_COMMIT = '10febb01d7be73d1491cf5e5347b5ab8b6c2c09e'
VSA_URL = 'https://files.pythonhosted.org/packages/8b/4a/fb5ff06de1cd108e4ff0a44e9dfe6aa7fda782d1e86a6bea052e731eed86/vsa-0.0.3.tar.gz'
VSA_SHA = '3924941c07ef3242ba5b7525483c78d7f5c0c09cc73966a1b2b4a9714b15c5be'
MODEL = 'minimax_h3_fastvideo_vsa_datafree_1300step_4step_int8_convrot.safetensors'
ENCODER = 'qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors'
VAE = 'minimax_h3_video_vae_int8_convrot.safetensors'
AUDIO = 'minimax_h3_audio_vae_fp32.safetensors'
# Repository revisions, sizes and SHA256 from HAYATE's audited model catalog.
ASSETS = [
 ('Kijai/MiniMax-H3-experimental', 'f4cac997f880e93cf6940af61ee8d58ef31ff7f7', MODEL, 'diffusion_models/'+MODEL, 22898594920, '7221ae65d78780354d51e5048d29728d9f1f8fb9baf50b1dd3df85f5101413d'),
 ('Comfy-Org/MiniMax-H3', '4cc1d817b6184899b41293954329f576cb5ae86b', 'text_encoders/'+ENCODER, 'text_encoders/'+ENCODER, 15687142551, '35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6'),
 ('Kijai/MiniMax-H3-experimental', 'f9c521b15b6883fa1c353f86f806ad76311aa5a4', VAE, 'vae/'+VAE, 3171670912, '9bb2d96f218c76babd85e0611b85ca8fb330a90546c01a0005e8a58a59593410'),
 ('Comfy-Org/MiniMax-H3', '4cc1d817b6184899b41293954329f576cb5ae86b', 'vae/'+AUDIO, 'vae/'+AUDIO, 605254808, '8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48'),
]


def require_colab():
    if sys.platform != 'linux' or not Path('/content').is_dir() or 'google.colab' not in sys.modules:
        raise RuntimeError('Run this from a Google Colab notebook, never on the local PC.')


def inspect_environment():
    require_colab()
    import psutil
    import torch
    gpu = torch.cuda.is_available()
    props = torch.cuda.get_device_properties(0) if gpu else None
    report = dict(python=sys.version.split()[0], torch=torch.__version__, cuda=torch.version.cuda,
                  gpu=props.name if props else None,
                  capability=list(torch.cuda.get_device_capability(0)) if gpu else None,
                  vram_gib=round(props.total_memory/2**30, 2) if props else 0,
                  ram_gib=round(psutil.virtual_memory().total/2**30, 2),
                  available_ram_gib=round(psutil.virtual_memory().available/2**30, 2),
                  disk_free_gib=round(shutil.disk_usage('/content').free/2**30, 2))
    blockers = []
    if not gpu or report['capability'][0] < 8:
        blockers.append('FastH3 VSA requires native BF16 CUDA (SM80+); T4/TPU are not supported by this route.')
    if report['ram_gib'] < 30:
        blockers.append('This offload profile requires at least 30 GiB host RAM; 64 GiB or more is preferred. This is a conservative policy, not a measured minimum.')
    if report['vram_gib'] < 20:
        blockers.append('Paid test profile requires at least 20 GiB VRAM; start at 608x352, 124 frames.')
    report['generation_blockers'] = blockers
    return report


def check_generation():
    report = inspect_environment()
    if report['generation_blockers']:
        raise RuntimeError('\n'.join(report['generation_blockers']))
    return report


def run(*args, cwd=None):
    subprocess.run([str(x) for x in args], cwd=cwd, check=True)


def setup():
    """Installs only on an eligible Colab VM. Preserves Colab's torch wheel family."""
    report = check_generation()
    ROOT.mkdir(exist_ok=True)
    comfy = ROOT/'ComfyUI'
    if not comfy.exists():
        run('git', 'clone', '--no-checkout', 'https://github.com/Kijai/ComfyUI.git', comfy)
        run('git', 'checkout', '--detach', COMFY_COMMIT, cwd=comfy)
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=comfy, text=True).strip()
    if actual != COMFY_COMMIT:
        raise RuntimeError('Existing ComfyUI revision differs; use a fresh Colab runtime.')
    # Prevent pip replacing the mutually compatible torch / torchvision / audio supplied by Colab.
    from importlib.metadata import version
    constraints = ROOT/'constraints.txt'
    constraints.write_text('\n'.join(f'{p}=={version(p)}' for p in ('torch', 'torchvision', 'torchaudio'))+'\n')
    run(sys.executable, '-m', 'pip', 'install', '-c', constraints, '-r', comfy/'requirements.txt', 'huggingface_hub', 'pytest')
    archive = ROOT/'vsa.tar.gz'
    urllib.request.urlretrieve(VSA_URL, archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != VSA_SHA:
        raise RuntimeError('VSA archive checksum mismatch')
    # Copy only regular Python/package files, never execute setup.py's Hopper build.
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            parts = Path(member.name).parts
            if len(parts) < 3 or parts[1] != 'vsa' or not member.isfile():
                continue
            target = (comfy / Path(*parts[1:])).resolve()
            if comfy.resolve() not in target.parents:
                raise RuntimeError('Unsafe VSA archive path')
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as src, target.open('wb') as dst:
                shutil.copyfileobj(src, dst)
    shutil.copytree(Path(__file__).parent/'vendor/h3_vsa', comfy/'custom_nodes/hayate_vsa', dirs_exist_ok=True)
    run(sys.executable, '-m', 'pip', 'check')
    run(sys.executable, '-c', 'import torch, triton, vsa; print(torch.__version__, triton.__version__)', cwd=comfy)
    (ROOT/'environment.json').write_text(json.dumps(report, indent=2))
    return report


def download_models():
    check_generation()
    from huggingface_hub import hf_hub_download
    models = ROOT/'ComfyUI/models'
    models.mkdir(parents=True, exist_ok=True)
    missing = sum(size for _, _, _, dest, size, _ in ASSETS if not (models/dest).exists())
    if shutil.disk_usage(models).free < missing + 8*2**30:
        raise RuntimeError('Insufficient disk for the approximately 39.5 GiB model set plus 8 GiB reserve.')
    for repo, revision, remote, dest, size, digest in ASSETS:
        target = models/dest
        target.parent.mkdir(parents=True, exist_ok=True)
        # Use local_dir to avoid an extra permanent HF cache copy.
        cached = Path(hf_hub_download(repo, remote, revision=revision, local_dir=ROOT/'downloads'))
        if cached.stat().st_size != size:
            raise RuntimeError(f'Size mismatch: {remote}')
        h = hashlib.sha256()
        with cached.open('rb') as f:
            for block in iter(lambda: f.read(8*1024*1024), b''):
                h.update(block)
        if h.hexdigest() != digest:
            raise RuntimeError(f'Checksum mismatch: {remote}')
        if target.is_symlink():
            if target.resolve() != cached.resolve():
                raise RuntimeError(f'Unexpected existing model link: {target}')
        elif not target.exists():
            target.symlink_to(cached)
        else:
            raise RuntimeError(f'Existing unverified model: {target}')
        print('Verified:', remote, flush=True)


def graph(prompt, seed, width=608, height=352, frames=124):
    if not prompt.strip() or width <= 0 or height <= 0 or width % 32 or height % 32 or frames < 5 or frames % 17 != 5:
        raise ValueError('Use a prompt, 32-pixel dimensions and 17*n+5 frames (124 = about 5.17 seconds).')
    return {
      '1': {'class_type':'UNETLoader','inputs':{'unet_name':MODEL,'weight_dtype':'default'}},
      '2': {'class_type':'CLIPLoader','inputs':{'clip_name':ENCODER,'type':'minimax','device':'default'}},
      '3': {'class_type':'VAELoader','inputs':{'vae_name':VAE}},
      '4': {'class_type':'VAELoader','inputs':{'vae_name':AUDIO}},
      '5': {'class_type':'MiniMaxH3SigmaShift','inputs':{'model':['1',0],'shift_video':12.0,'shift_audio':3.0}},
      '6': {'class_type':'H3VSA','inputs':{'model':['5',0],'gate_file':'<model-embedded-gates>','topk_ratio':0.1,'min_tokens':4096}},
      '7': {'class_type':'MiniMaxH3ImageToVideo','inputs':{'clip':['2',0],'vae':['3',0],'prompt':prompt,'width':width,'height':height,'length':frames}},
      '8': {'class_type':'RandomNoise','inputs':{'noise_seed':seed}},
      '9': {'class_type':'KSamplerSelect','inputs':{'sampler_name':'euler'}},
      '10': {'class_type':'BasicScheduler','inputs':{'model':['6',0],'scheduler':'simple','steps':4,'denoise':1.0}},
      '11': {'class_type':'BasicGuider','inputs':{'model':['6',0],'conditioning':['7',0]}},
      '12': {'class_type':'SamplerCustomAdvanced','inputs':{'noise':['8',0],'guider':['11',0],'sampler':['9',0],'sigmas':['10',0],'latent_image':['7',1]}},
      '13': {'class_type':'VAEDecode','inputs':{'samples':['12',0],'vae':['3',0]}},
      '14': {'class_type':'VAEDecodeAudio','inputs':{'samples':['12',0],'vae':['4',0]}},
      '15': {'class_type':'CreateVideo','inputs':{'images':['13',0],'audio':['14',0],'fps':24,'bit_depth':8,'color_space':'sRGB'}},
      '16': {'class_type':'SaveVideo','inputs':{'video':['15',0],'filename_prefix':'hayate/'+uuid.uuid4().hex,'format':'mp4','codec':'h264'}},
    }


def cost_report(elapsed_seconds, successful, yen_per_cu, cu_per_hour):
    values = (elapsed_seconds, yen_per_cu, cu_per_hour)
    if any(not math.isfinite(x) or x < 0 for x in values) or successful < 0:
        raise ValueError('Costs, time and successful output count must be nonnegative and finite.')
    priced = yen_per_cu > 0 and cu_per_hour > 0
    total = elapsed_seconds/3600*yen_per_cu*cu_per_hour if priced else None
    each = total/successful if total is not None and successful else None
    return dict(session_seconds=elapsed_seconds, successful_outputs=successful,
                estimated_session_yen=total, estimated_yen_per_success=each,
                target_met=each <= 1 if each is not None else None,
                seconds_budget_per_success=3600/(yen_per_cu*cu_per_hour) if priced else None)


class Session:
    """One persistent VM-local worker; repeat seeds, retaining cached conditioning/models."""
    def __init__(self):
        self.process = None
        self.log = None
        self.successful = 0

    def api(self, path, payload=None):
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request('http://127.0.0.1:8189'+path, data=data, headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)

    def start(self):
        report = check_generation()
        if self.process is not None and self.process.poll() is None:
            return
        self.log = (ROOT/'worker.log').open('a')
        args = [sys.executable, 'main.py', '--listen', '127.0.0.1', '--port', '8189', '--disable-auto-launch',
                '--enable-dynamic-vram', '--enable-triton-backend', '--async-offload', '2', '--cache-lru', '2']
        if report['available_ram_gib'] < 48:
            args.append('--disable-pinned-memory')
        # No blanket --fast, FP8 or forced high-VRAM allocation before output-quality validation.
        self.process = subprocess.Popen(args, cwd=ROOT/'ComfyUI', stdout=self.log, stderr=subprocess.STDOUT)
        deadline = time.monotonic()+180
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError((ROOT/'worker.log').read_text()[-6000:])
            try:
                info = self.api('/object_info')
                required = {n['class_type'] for n in graph('test', 1).values()}
                if required - info.keys():
                    self.close()
                    raise RuntimeError(f'Missing nodes: {required - info.keys()}')
                return
            except (OSError, ValueError):
                time.sleep(2)
        self.close()
        raise TimeoutError('Worker did not become ready; inspect worker.log.')

    def generate(self, prompt, seed, width=608, height=352, frames=124, timeout=1800):
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError('Start the session first.')
        request = graph(prompt, seed, width, height, frames)
        started = time.monotonic()
        queued = self.api('/prompt', {'prompt':request, 'client_id':uuid.uuid4().hex})
        if queued.get('node_errors') or not queued.get('prompt_id'):
            raise RuntimeError(str(queued))
        pid = queued['prompt_id']
        try:
            while time.monotonic()-started < timeout:
                if self.process.poll() is not None:
                    raise RuntimeError('Worker exited. Inspect worker.log (possible RAM/OOM failure).')
                entry = self.api('/history/'+pid).get(pid)
                if entry:
                    status = entry.get('status', {})
                    if status.get('status_str') == 'error':
                        raise RuntimeError(str(status))
                    if status.get('completed') and status.get('status_str') == 'success':
                        output_root = (ROOT/'ComfyUI/output').resolve()
                        paths = []
                        for out in entry.get('outputs', {}).values():
                            for key in ('videos','gifs','images'):
                                for item in out.get(key, []):
                                    path = (output_root/item.get('subfolder','')/item['filename']).resolve()
                                    if output_root in path.parents and path.suffix == '.mp4' and path.is_file():
                                        # Decode container metadata before counting a successful output.
                                        import av
                                        with av.open(str(path)) as media:
                                            if not media.streams.video or next(media.decode(video=0), None) is None:
                                                raise RuntimeError('Video is empty/undecodable')
                                        paths.append(str(path))
                        if not paths:
                            raise RuntimeError('No verified MP4 output')
                        self.successful += len(paths)
                        result = dict(seconds=time.monotonic()-started, seed=seed, width=width, height=height, frames=frames, files=paths)
                        with (ROOT/'generations.jsonl').open('a') as f:
                            f.write(json.dumps(result)+'\n')
                        return result
                time.sleep(2)
            raise TimeoutError('Generation timed out; worker stopped to avoid ongoing compute.')
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.log is not None:
            self.log.close()
