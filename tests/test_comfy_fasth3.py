from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from hayate.backends.minimax_h3.comfy_fasth3 import (
    build_graph, readiness, runtime_paths, MODEL_IDS, FL2VA_MODEL_IDS, ComfyFastH3Backend,
)
from hayate.backends.minimax_h3.generation import GenerationRequest
from hayate.backends.minimax_h3 import comfy_worker
from hayate.webui.server import GenerationPayload, _profile_values


def prepared(root):
    runtime, python = runtime_paths(root)
    for path in (runtime / "main.py", python, runtime / "hayate-ready.json"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    return [{"id": key, "status": "verified"} for key in MODEL_IDS]


def test_comfy_only_needs_its_own_dit_and_shared_models(tmp_path):
    assets = prepared(tmp_path)
    assert readiness(tmp_path, assets)["ready"]
    assets[0]["status"] = "downloading"
    assert not readiness(tmp_path, assets)["ready"]
    assets[0]["status"] = "present_unverified"
    assert not readiness(tmp_path, assets)["ready"]
    assert not readiness(tmp_path, assets[1:])["ready"]


def test_graph_wires_pinned_schedule_and_optimizations():
    graph = build_graph("car", 44, 512, 288, 124, 5, 2)
    assert graph["3"]["inputs"]["selection.vsa_keep_percent"] == 5
    assert graph["3"]["inputs"]["model"] == ["17", 0]
    assert graph["17"]["inputs"]["chunks"] == 2
    assert graph["11"]["inputs"]["sigmas"] == "0.9999166, 0.9728326, 0.9230769, 0.8, 0.0"
    assert graph["7"]["inputs"]["prompt"] == "car"
    assert graph["8"]["inputs"]["noise_seed"] == 44
    assert graph["15"]["inputs"]["audio"] == ["14", 0]
    assert graph["13"]["class_type"] == "MiniMaxH3FastVAEDecode"
    with pytest.raises(ValueError):
        build_graph("car", 1, 512, 512, 124, 1)


def test_backend_rejects_image_inputs_and_preserves_unicode(tmp_path):
    backend = ComfyFastH3Backend(tmp_path, prepared(tmp_path))
    request = GenerationRequest("雨の街", tmp_path, tmp_path / "out.mp4")
    plan = backend.plan(request)
    assert plan.executable
    assert plan.backend == "comfy_fasth3"
    assert json.loads(plan.command[-1])["7"]["inputs"]["prompt"] == "雨の街"
    assert not backend.plan(replace(request, task="fl2va")).executable
    assert not backend.plan(replace(request, image_path=tmp_path / "image.png")).executable
    values = _profile_values(GenerationPayload(prompt="a", profile="comfy_fasth3"))
    assert values["steps"] == 5 and not values["easycache"] and not values["pdd"]


def test_node_inventory_fails_before_loading_weights():
    with pytest.raises(RuntimeError, match="node missing"):
        comfy_worker.validate_nodes({}, build_graph("a", 1, 512, 512, 124))
    with pytest.raises(RuntimeError, match="Unsupported choice/model"):
        comfy_worker.validate_nodes({"UNETLoader": {"input": {"required": {"unet_name": [["other"]]}}}},
                                    {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "missing"}}})


def test_web_api_routes_comfy_profile_without_starting_an_engine(tmp_path):
    from fastapi.testclient import TestClient
    from hayate.webui.server import create_app
    from hayate.webui.jobs import JobStore
    assets = prepared(tmp_path)
    submitted = []
    def submit(plan, request):
        submitted.append(plan)
        return {"id": "fake-job", "request": request, "plan": plan.to_dict()}
    manager = SimpleNamespace(store=JobStore(tmp_path / "jobs.sqlite"), submit=submit)
    service = SimpleNamespace(status=lambda: {"assets": assets})
    with TestClient(create_app(tmp_path, job_manager=manager, model_setup_service=service)) as client:
        assert client.get("/api/models/setup").json()["comfy_fasth3"]["ready"]
        assert client.get("/api/models/setup").json()["comfy_fl2va"]["missing_models"] == ["transformer_w4a8"]
        response = client.post("/api/jobs", json={"prompt": "car", "profile": "comfy_fasth3", "vsa_keep": 5},
                               headers={"X-HAYATE-UI": "1"})
        assert response.status_code == 202, response.text
        assert submitted[0].backend == "comfy_fasth3"
        assert json.loads(submitted[0].command[-1])["3"]["inputs"]["selection.vsa_keep_percent"] == 5
        assets[0]["status"] = "downloading"
        response = client.post("/api/jobs", json={"prompt": "car", "profile": "comfy_fasth3"},
                               headers={"X-HAYATE-UI": "1"})
        assert response.status_code == 422
        assert len(submitted) == 1
        response = client.post("/api/jobs", json={"prompt": "anime", "profile": "comfy_fl2va"},
                               headers={"X-HAYATE-UI": "1"})
        assert response.status_code == 422
        assert len(submitted) == 1


def test_web_api_routes_image_to_matching_fl2va_graph(tmp_path):
    import io
    from PIL import Image
    from fastapi.testclient import TestClient
    from hayate.webui.server import create_app
    from hayate.webui.jobs import JobStore

    assets = prepared(tmp_path) + [{"id": "transformer_w4a8", "status": "verified"}]
    submitted = []
    manager = SimpleNamespace(store=JobStore(tmp_path / "jobs.sqlite"),
                              submit=lambda plan, request: (submitted.append(plan) or
                                                            {"id": "image-job", "request": request,
                                                             "plan": plan.to_dict()}))
    service = SimpleNamespace(status=lambda: {"assets": assets})
    picture = io.BytesIO()
    Image.new("RGB", (512, 288), "blue").save(picture, format="PNG")
    with TestClient(create_app(tmp_path, job_manager=manager, model_setup_service=service)) as client:
        assert client.get("/api/models/setup").json()["comfy_fl2va"]["ready"]
        uploaded = client.post("/api/assets", files={"file": ("start.png", picture.getvalue(), "image/png")},
                               headers={"X-HAYATE-UI": "1"})
        assert uploaded.status_code == 200
        response = client.post("/api/jobs", json={"prompt": "same 2D anime character moving gently",
                                                  "profile": "comfy_fl2va", "image_asset_id": uploaded.json()["id"],
                                                  "width": 512, "height": 288},
                               headers={"X-HAYATE-UI": "1"})
        assert response.status_code == 202, response.text
        graph = json.loads(submitted[0].command[submitted[0].command.index("--graph") + 1])
        assert graph["7"]["inputs"]["first_frame"] == ["18", 0]
        assert graph["11"]["class_type"] == "BasicScheduler"
        assert "--first-image" in submitted[0].command


def test_output_path_cannot_escape_job_directory(tmp_path):
    root = tmp_path / "owned"
    root.mkdir()
    (tmp_path / "other.mp4").write_bytes(b"video")
    entry = {"outputs": {"16": {"videos": [{"filename": "../other.mp4"}]}}}
    with pytest.raises(RuntimeError, match="without an MP4"):
        comfy_worker.output_video(entry, root)


@pytest.mark.parametrize("vsa_active", [True, False])
def test_worker_api_to_saved_output_and_owned_cleanup(tmp_path, monkeypatch, vsa_active):
    """No server or GPU starts: exercise the API/WS/output bridge with fakes."""
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    output = tmp_path / "result.mp4"
    raw = tmp_path / ".comfy/result"
    raw.mkdir(parents=True)
    (raw / "video.mp4").write_bytes(b"mp4-test")
    graph = {"16": {"class_type": "SaveVideo", "inputs": {}},
             "3": {"class_type": "SolAttnMiniMax", "inputs": {}}}
    responses = {
        "/object_info": {kind: {"input": {"required": {}}} for kind in ("SaveVideo", "SolAttnMiniMax")},
        "/prompt": {"prompt_id": "owned"},
        "/history/owned": {"owned": {"status": {"completed": True, "status_str": "success"},
                                    "outputs": {"16": {"videos": [{"filename": "video.mp4"}]}}}},
    }
    from urllib.parse import urlparse
    import io
    monkeypatch.setattr(comfy_worker.urllib.request, "urlopen",
                        lambda req, **kw: io.BytesIO(json.dumps(responses[urlparse(req.full_url).path]).encode()))
    process = SimpleNamespace(poll=lambda: None, terminate=lambda: stopped.append(True), wait=lambda **kw: 0)
    stopped = []
    def popen(args, **kw):
        assert args[args.index("--listen") + 1] == "127.0.0.1"
        kw["stdout"].write("VSA tiles" if vsa_active else "dense fallback")
        kw["stdout"].flush()
        return process
    monkeypatch.setattr(comfy_worker.subprocess, "Popen", popen)
    monkeypatch.setattr(comfy_worker.signal, "signal", lambda *args: None)
    ws = SimpleNamespace(recv=lambda: '{"type":"status","data":{}}', close=lambda: None)
    monkeypatch.setitem(sys.modules, "websocket", SimpleNamespace(create_connection=lambda *a, **kw: ws,
                                                                WebSocketTimeoutException=TimeoutError))
    class Media:
        streams = SimpleNamespace(audio=[1])
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def decode(self, **kw): return iter([object()])
    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=lambda *a: Media()))
    if vsa_active:
        comfy_worker.run(runtime, output, graph)
        assert output.read_bytes() == b"mp4-test"
    else:
        with pytest.raises(RuntimeError, match="VSA activation"):
            comfy_worker.run(runtime, output, graph)
        assert not output.exists()
    assert stopped == [True]

def test_image_graph_and_command(tmp_path):
    from PIL import Image
    from dataclasses import replace
    from hayate.backends.minimax_h3.comfy_fasth3 import build_graph
    graph = build_graph('move', 1, 512, 288, 124, first_image=True, last_image=True, mode="fl2va")
    assert graph['7']['inputs']['first_frame'] == ['18', 0]
    assert graph['7']['inputs']['last_frame'] == ['19', 0]
    assert graph['18']['inputs']['image'] == 'first.png'
    assert graph['19']['class_type'] == 'LoadImage'
    assert graph['1']['inputs']['unet_name'] == 'minimax_h3_fl2va_pruned_w4a8_mixed.safetensors'
    assert graph['11']['class_type'] == 'BasicScheduler'
    assert graph['11']['inputs']['steps'] == 50
    assert graph['10']['inputs']['sampler_name'] == 'res_multistep'
    assert '3' not in graph and '2' not in graph
    plain = build_graph('move', 1, 512, 288, 124)
    assert '18' not in plain and 'first_frame' not in plain['7']['inputs']

    backend = ComfyFastH3Backend(tmp_path, prepared(tmp_path) + [{"id": "transformer_w4a8", "status": "verified"}], mode="fl2va")
    image = tmp_path / "frame.png"
    Image.new("RGB", (512, 288)).save(image)
    req = GenerationRequest("move", tmp_path, tmp_path / "out.mp4", width=512, height=288)
    plan = backend.plan(replace(req, image_path=image, last_image_path=image))
    assert plan.executable
    assert "--first-image" in plan.command and "--last-image" in plan.command
    assert not backend.plan(replace(req, last_image_path=image)).executable
    assert not backend.plan(replace(req, image_path=image, width=512, height=512)).executable
    assert not ComfyFastH3Backend(tmp_path, prepared(tmp_path)).plan(replace(req, image_path=image)).executable
    assert readiness(tmp_path, prepared(tmp_path), mode="fl2va")["missing_models"] == ["transformer_w4a8"]
    assert set(FL2VA_MODEL_IDS) <= {a["id"] for a in backend.assets}
    with pytest.raises(ValueError, match="text only"):
        build_graph('move', 1, 512, 288, 124, first_image=True)


@pytest.mark.parametrize("vsa_active", [True, False])
def test_warm_worker_uses_existing_server_and_separate_job_inputs(tmp_path, monkeypatch, vsa_active):
    import io
    from urllib.parse import urlparse
    from PIL import Image

    runtime = tmp_path / "runtime"
    runtime.mkdir()
    input_dir = tmp_path / "shared-input"
    output_dir = tmp_path / "shared-output"
    input_dir.mkdir()
    output_dir.mkdir()
    (output_dir / "result.mp4").write_bytes(b"mp4-test")
    server_log = tmp_path / "server.log"
    server_log.write_text("VSA tiles from prior job\n")
    first = tmp_path / "first.png"
    last = tmp_path / "last.png"
    Image.new("RGB", (32, 32)).save(first)
    Image.new("RGB", (32, 32)).save(last)
    for key, value in {
        "HAYATE_COMFY_BASE_URL": "http://127.0.0.1:8189",
        "HAYATE_COMFY_INPUT_DIR": str(input_dir),
        "HAYATE_COMFY_OUTPUT_DIR": str(output_dir),
        "HAYATE_COMFY_SERVER_LOG": str(server_log),
        "HAYATE_COMFY_REUSED": "1",
    }.items():
        monkeypatch.setenv(key, value)
    graph = {
        "16": {"class_type": "SaveVideo", "inputs": {}},
        "3": {"class_type": "SolAttnMiniMax", "inputs": {}},
        "18": {"class_type": "LoadImage", "inputs": {"image": "first.png"}},
        "19": {"class_type": "LoadImage", "inputs": {"image": "last.png"}},
    }
    seen = {}

    def urlopen(request, **kwargs):
        path = urlparse(request.full_url if hasattr(request, "full_url") else request).path
        if path == "/object_info":
            payload = {kind: {"input": {"required": {}}} for kind in ("SaveVideo", "LoadImage", "SolAttnMiniMax")}
        elif path == "/prompt":
            seen.update(json.loads(request.data))
            assert seen["prompt"]["18"]["inputs"]["image"] != "first.png"
            assert seen["prompt"]["19"]["inputs"]["image"] != "last.png"
            assert seen["prompt"]["18"]["inputs"]["image"] != seen["prompt"]["19"]["inputs"]["image"]
            assert len(list(input_dir.glob("*.png"))) == 2
            if vsa_active:
                with server_log.open("a") as log:
                    log.write("VSA tiles current job\n")
            payload = {"prompt_id": "owned"}
        else:
            payload = {"owned": {"status": {"completed": True, "status_str": "success"},
                                "outputs": {"16": {"videos": [{"filename": "result.mp4"}]}}}}
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(comfy_worker.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(comfy_worker.subprocess, "Popen", lambda *a, **kw: pytest.fail("server restarted"))
    monkeypatch.setattr(comfy_worker.signal, "signal", lambda *args: None)
    ws = SimpleNamespace(recv=lambda: '{"type":"status","data":{}}', close=lambda: None)
    monkeypatch.setitem(sys.modules, "websocket", SimpleNamespace(
        create_connection=lambda *a, **kw: ws, WebSocketTimeoutException=TimeoutError))

    class Media:
        streams = SimpleNamespace(audio=[1])
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def decode(self, **kw): return iter([object()])

    monkeypatch.setitem(sys.modules, "av", SimpleNamespace(open=lambda *a: Media()))
    output = tmp_path / "final.mp4"
    if vsa_active:
        comfy_worker.run(runtime, output, graph, first_image=first, last_image=last)
        assert output.read_bytes() == b"mp4-test"
        assert not (output_dir / "result.mp4").exists()
    else:
        with pytest.raises(RuntimeError, match="VSA activation"):
            comfy_worker.run(runtime, output, graph, first_image=first, last_image=last)
        assert not output.exists()
    assert not list(input_dir.iterdir())
    assert ("VSA tiles current job" in output.with_suffix(".comfy.log").read_text()) == vsa_active
    assert "prior job" not in output.with_suffix(".comfy.log").read_text()
