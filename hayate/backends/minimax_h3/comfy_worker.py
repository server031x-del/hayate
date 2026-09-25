"""Owned, loopback-only ComfyUI process and API bridge for one HAYATE job."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
from urllib.parse import urlsplit
import urllib.request
import uuid


def emit(stage, detail, progress):
    print("HAYATE_EVENT " + json.dumps({"schema": 1, "type": "progress", "stage": stage,
          "detail": detail, "progress": progress}, ensure_ascii=True), flush=True)


def output_video(entry, root):
    root = root.resolve()
    for result in entry.get("outputs", {}).values():
        for key in ("videos", "gifs", "images"):
            for item in result.get(key, []):
                path = (root / item.get("subfolder", "") / item.get("filename", "")).resolve()
                if root in path.parents and path.suffix.lower() == ".mp4" and path.is_file():
                    return path
    raise RuntimeError("ComfyUI completed without an MP4 output")


def validate_nodes(info, graph):
    for node in graph.values():
        kind = node["class_type"]
        if kind not in info:
            raise RuntimeError(f"ComfyUI node missing: {kind}")
        required = info[kind].get("input", {}).get("required", {})
        for name, definition in required.items():
            if name not in node["inputs"]:
                raise RuntimeError(f"Missing required input {kind}.{name}")
            value = node["inputs"][name]
            if isinstance(definition[0], list) and not isinstance(value, list) and value not in definition[0]:
                raise RuntimeError(f"Unsupported choice/model {kind}.{name}: {value}")


def run(runtime, output, graph, timeout=7200, first_image=None, last_image=None):
    import websocket
    model_label = "FL2VA" if any(node.get("inputs", {}).get("unet_name", "").startswith("minimax_h3_fl2va")
                                  for node in graph.values()) else "FastH3"
    output.parent.mkdir(parents=True, exist_ok=True)
    warm_base = os.environ.get("HAYATE_COMFY_BASE_URL", "")
    if warm_base:
        parsed = urlsplit(warm_base)
        if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.port is None or parsed.username or parsed.password or parsed.path not in ("", "/"):
            raise ValueError("Warm ComfyUI must be bound to 127.0.0.1")
        warm_base = f"http://127.0.0.1:{parsed.port}"
    if warm_base:
        base = warm_base
        raw_dir = Path(os.environ["HAYATE_COMFY_OUTPUT_DIR"])
        input_dir = Path(os.environ["HAYATE_COMFY_INPUT_DIR"])
        server_log = Path(os.environ["HAYATE_COMFY_SERVER_LOG"])
    else:
        # Direct CLI runs retain a per-job owned server.
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        raw_dir = output.parent / ".comfy" / output.stem
        input_dir = raw_dir / "input"
        server_log = None
    def api(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as response:
            return json.load(response)
    raw_dir.mkdir(parents=True, exist_ok=True)
    input_dir.mkdir(parents=True, exist_ok=True)
    from PIL import Image, ImageOps
    staged_inputs = []
    for source, name, node_id in ((first_image, "first.png", "18"), (last_image, "last.png", "19")):
        if source:
            if warm_base:
                name = f"{uuid.uuid4().hex}_{name}"
                graph[node_id]["inputs"]["image"] = name
            with Image.open(source) as img:
                ImageOps.exif_transpose(img).convert("RGB").save(input_dir / name)
            staged_inputs.append(input_dir / name)
    log_path = output.with_suffix(".comfy.log")
    args = [] if warm_base else [sys.executable, "-u", str(runtime / "main.py"), "--listen", "127.0.0.1", "--port", str(port),
            "--disable-auto-launch", "--output-directory", str(raw_dir), "--input-directory", str(input_dir),
            "--extra-model-paths-config", str(runtime / "hayate-models.yaml"),
            "--enable-dynamic-vram", "--disable-pinned-memory", "--async-offload", "2"]
    process = None
    ws = None
    log_start = server_log.stat().st_size if server_log is not None else 0
    def new_server_log():
        assert server_log is not None
        with server_log.open("rb") as source:
            source.seek(log_start)
            return source.read().decode("utf-8", errors="replace")
    def interrupted(*_):
        raise KeyboardInterrupt("HAYATE cancelled this generation")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, interrupted)
    try:
        with (nullcontext() if warm_base else log_path.open("w", encoding="utf-8")) as log:
            if warm_base:
                emit("起動準備", f"ComfyUI {model_label}を再利用しています" if os.environ.get("HAYATE_COMFY_REUSED") == "1" else f"ComfyUI {model_label}を準備しました", 1)
                info = api("/object_info")
            else:
                process = subprocess.Popen(args, cwd=runtime, stdout=log, stderr=subprocess.STDOUT)
                emit("起動準備", f"ComfyUI {model_label}を起動しています", 1)
                deadline = time.monotonic() + 180
                while True:
                    if process.poll() is not None:
                        raise RuntimeError("ComfyUI startup failed: " + log_path.read_text(errors="replace")[-3000:])
                    try:
                        info = api("/object_info")
                        break
                    except (OSError, ValueError):
                        if time.monotonic() > deadline:
                            raise TimeoutError("ComfyUI startup timeout")
                        time.sleep(1)
            validate_nodes(info, graph)
            client = uuid.uuid4().hex
            ws = websocket.create_connection(base.replace("http://", "ws://") + f"/ws?clientId={client}", timeout=2)
            queued = api("/prompt", {"prompt": graph, "client_id": client})
            if queued.get("node_errors") or not queued.get("prompt_id"):
                raise RuntimeError("ComfyUI rejected graph: " + json.dumps(queued))
            pid = queued["prompt_id"]
            emit("モデル読込", f"{model_label}モデルを読み込んでいます", 3)
            started = time.monotonic()
            last_history = 0.
            while time.monotonic() - started < timeout:
                if process is not None and process.poll() is not None:
                    raise RuntimeError("ComfyUI exited: " + log_path.read_text(errors="replace")[-3000:])
                try:
                    raw = ws.recv()
                    message = json.loads(raw) if isinstance(raw, str) else {}
                    data = message.get("data", {})
                    if data.get("prompt_id", pid) == pid:
                        if message.get("type") == "execution_error":
                            raise RuntimeError(data.get("exception_message", "ComfyUI execution error"))
                        if message.get("type") == "executing" and data.get("node"):
                            name = graph.get(str(data["node"]), {}).get("class_type", "生成処理")
                            emit("生成処理", name, 80 if str(data["node"]) in ("13", "14", "15", "16") else 10)
                        if message.get("type") == "progress":
                            value, maximum = data.get("value", 0), max(data.get("max", 1), 1)
                            emit("生成処理", f"{value} / {maximum}", 10 + 65 * value / maximum)
                except websocket.WebSocketTimeoutException:
                    pass
                if time.monotonic() - last_history < 2:
                    continue
                last_history = time.monotonic()
                entry = api("/history/" + pid).get(pid, {})
                status = entry.get("status", {})
                if status.get("status_str") in ("error", "failed"):
                    raise RuntimeError(str(status))
                if status.get("completed") and status.get("status_str") == "success":
                    path = output_video(entry, raw_dir)
                    import av
                    with av.open(str(path)) as media:
                        if not media.streams.audio or next(media.decode(video=0), None) is None:
                            raise RuntimeError("Output video/audio validation failed")
                    runtime_log = new_server_log() if server_log is not None else log_path.read_text(errors="replace")
                    uses_vsa = any(node["class_type"] == "SolAttnMiniMax" for node in graph.values())
                    if "kernel failed" in runtime_log or (uses_vsa and "VSA tiles" not in runtime_log):
                        raise RuntimeError("VSA activation/kernel check failed; inspect " + str(log_path))
                    shutil.copyfile(path, output)
                    if warm_base:
                        path.unlink(missing_ok=True)
                    emit("完了", "動画と音声を保存しました", 100)
                    return
            raise TimeoutError("FastH3 generation timed out")
    finally:
        if ws is not None:
            ws.close()
        if server_log is not None:
            log_path.write_text(new_server_log(), encoding="utf-8")
        for staged in staged_inputs:
            staged.unlink(missing_ok=True)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--first-image", type=Path)
    parser.add_argument("--last-image", type=Path)
    args = parser.parse_args()
    run(args.runtime, args.output, json.loads(args.graph), first_image=args.first_image, last_image=args.last_image)
