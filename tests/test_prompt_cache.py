from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from hayate.backends.minimax_h3.prompt_cache import (
    CACHE_SCHEMA_VERSION,
    install_prompt_cache_override,
)


class _Logger:
    def info(self, *_args):
        pass


def _fake_upstream():
    state = {"encodes": 0, "hits": 0}
    module = SimpleNamespace(logger=_Logger())

    def key(_args, task, plan, prompt):
        return f"{task}:{plan.height}x{plan.width}:{prompt}"

    def encode(args, task, plan, prompt, _device):
        cache = Path(args.prompt_cache)
        sidecar = Path(str(cache) + ".json")
        expected = module._prompt_cache_key(args, task, plan, prompt)
        if cache.is_file() and sidecar.is_file():
            metadata = json.loads(sidecar.read_text(encoding="utf-8"))
            if metadata.get("key") == expected:
                state["hits"] += 1
                return "cached"
        state["encodes"] += 1
        cache.write_bytes(f"embedding-{state['encodes']}".encode())
        sidecar.write_text(json.dumps({"key": expected, "block_timestamps": []}), encoding="utf-8")
        return "encoded"

    module._prompt_cache_key = key
    module.encode_prompt_stage = encode
    return module, state


def _args(cache: Path, text_encoder: Path):
    return SimpleNamespace(
        prompt_cache=str(cache),
        text_encoder=str(text_encoder),
        image_path=None,
        last_image_path=None,
        reference=[],
    )


def test_prompt_cache_is_fingerprinted_atomic_and_reused(tmp_path):
    module, state = _fake_upstream()
    install_prompt_cache_override(module, upstream_commit="audited-commit")
    model = tmp_path / "encoder.safetensors"
    model.write_bytes(b"not-a-real-model-v1")
    cache = tmp_path / "prompt.safetensors"
    args = _args(cache, model)
    plan = SimpleNamespace(height=256, width=256)

    assert module.encode_prompt_stage(args, "t2va", plan, "hello", "cuda:0") == "encoded"
    metadata = json.loads(Path(str(cache) + ".json").read_text(encoding="utf-8"))
    assert metadata["hayate"]["schema_version"] == CACHE_SCHEMA_VERSION
    assert metadata["hayate"]["upstream_commit"] == "audited-commit"
    assert state == {"encodes": 1, "hits": 0}
    assert not list(tmp_path.glob(".*.tmp*"))

    assert module.encode_prompt_stage(args, "t2va", plan, "hello", "cuda:0") == "cached"
    assert state == {"encodes": 1, "hits": 1}

    model.write_bytes(b"not-a-real-model-v2")
    assert module.encode_prompt_stage(args, "t2va", plan, "hello", "cuda:0") == "encoded"
    assert state == {"encodes": 2, "hits": 1}


def test_prompt_cache_input_content_change_invalidates_same_path(tmp_path):
    module, state = _fake_upstream()
    install_prompt_cache_override(module, upstream_commit="audited-commit")
    model = tmp_path / "encoder.safetensors"
    model.write_bytes(b"model")
    image = tmp_path / "reference.png"
    image.write_bytes(b"first")
    cache = tmp_path / "prompt.safetensors"
    args = _args(cache, model)
    args.image_path = str(image)
    plan = SimpleNamespace(height=256, width=256)

    module.encode_prompt_stage(args, "fl2va", plan, "hello", "cuda:0")
    image.write_bytes(b"second")
    module.encode_prompt_stage(args, "fl2va", plan, "hello", "cuda:0")
    assert state["encodes"] == 2

