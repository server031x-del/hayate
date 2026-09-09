"""Owned, loopback-only ComfyUI process and API bridge for one HAYATE job."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
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


def run(runtime, output, graph, timeout=7200):
    import websocket
    output.parent.mkdir(parents=True, exist_ok=True)
    # Each job owns its server, so cancellation cannot interrupt another API client.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    def api(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as response:
            return json.load(response)
    raw_dir = output.parent / ".comfy" / output.stem
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_path = output.with_suffix(".comfy.log")
    args = [sys.executable, "-u", str(runtime / "main.py"), "--listen", "127.0.0.1", "--port", str(port),
            "--disable-auto-launch", "--output-directory", str(raw_dir),
            "--extra-model-paths-config", str(runtime / "hayate-models.yaml"),
            "--enable-dynamic-vram", "--disable-pinned-memory", "--async-offload", "2"]
    process = None
    ws = None
    def interrupted(*_):
        raise KeyboardInterrupt("HAYATE cancelled this generation")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, interrupted)
    try:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(args, cwd=runtime, stdout=log, stderr=subprocess.STDOUT)
            emit("起動準備", "ComfyUI FastH3を起動しています", 1)
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
            ws = websocket.create_connection(f"ws://127.0.0.1:{port}/ws?clientId={client}", timeout=2)
            queued = api("/prompt", {"prompt": graph, "client_id": client})
            if queued.get("node_errors") or not queued.get("prompt_id"):
                raise RuntimeError("ComfyUI rejected graph: " + json.dumps(queued))
            pid = queued["prompt_id"]
            emit("モデル読込", "FastH3モデルを読み込んでいます", 3)
            started = time.monotonic()
            last_history = 0.
            while time.monotonic() - started < timeout:
                if process.poll() is not None:
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
                    runtime_log = log_path.read_text(errors="replace")
                    if "VSA tiles" not in runtime_log or "kernel failed" in runtime_log:
                        raise RuntimeError("VSA activation not confirmed; inspect " + str(log_path))
                    shutil.copyfile(path, output)
                    emit("完了", "動画と音声を保存しました", 100)
                    return
            raise TimeoutError("FastH3 generation timed out")
    finally:
        if ws is not None:
            ws.close()
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
    args = parser.parse_args()
    run(args.runtime, args.output, json.loads(args.graph))
