from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from hayate.backends.minimax_h3.native_cache import install_native_transformer_cache


class _Device:
    def __init__(self, value):
        self.value = str(value)
        self.type = self.value.split(":", 1)[0]

    def __eq__(self, other):
        return isinstance(other, _Device) and self.value == other.value

    def __str__(self):
        return self.value


class _FakeCuda:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def synchronize(_device=None):
        return None

    @staticmethod
    def empty_cache():
        return None


class _FakeModel:
    def __init__(self):
        self.moves = []

    def to(self, device):
        self.moves.append(str(device))
        return self


def _args(**overrides):
    values = {
        "blocks_to_swap": 0,
        "progressive_load": False,
        "lora_weight": None,
        "offload_engine": "blockswap",
        "compile": False,
        "attn_mode": "sageattn",
        "ckpt_dir": "/models/h3",
        "dit": "/models/dit.safetensors",
        "dit_dtype": "bfloat16",
        "fp8": False,
        "fp8_scaled": False,
        "fp8_fast": False,
        "fp8_exclude_adaln": False,
        "int8_fast": True,
        "classic_block_swap": False,
        "act_chunk_rows": 32768,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_native_transformer_stays_warm_and_is_reused_between_runs(monkeypatch):
    loads = []
    clean_calls = []
    attention_calls = []
    chunk_calls = []
    model = _FakeModel()
    package = ModuleType("minimax_video")
    package.__path__ = []
    attention = SimpleNamespace(set_attention_backend=attention_calls.append)
    transformer_module = SimpleNamespace(set_act_chunk_rows=chunk_calls.append)
    monkeypatch.setitem(sys.modules, "minimax_video", package)
    monkeypatch.setitem(sys.modules, "minimax_video.attention", attention)
    monkeypatch.setitem(sys.modules, "minimax_video.transformer", transformer_module)
    module = SimpleNamespace(
        torch=SimpleNamespace(device=_Device, cuda=_FakeCuda),
        load_transformer_stage=lambda *_args: (loads.append(model) or model, None),
        clean_memory_on_device=lambda device: clean_calls.append(str(device)),
    )

    def run_one(args, task, device):
        transformer, _loader = module.load_transformer_stage(args, task, device)
        module.clean_memory_on_device(device)
        return transformer

    module.run_one = run_one
    install_native_transformer_cache(module)

    first = module.run_one(_args(), "t2va", "cuda:0")
    second = module.run_one(_args(), "t2va", "cuda:0")

    assert first is second is model
    assert len(loads) == 1
    assert model.moves == ["cpu", "cuda:0", "cpu", "cuda:0", "cpu", "cuda:0"]
    assert clean_calls == ["cuda:0", "cuda:0"]
    assert attention_calls == ["sageattn"]
    assert chunk_calls == [32768]


def test_native_transformer_cache_reloads_when_model_contract_changes():
    models = []
    module = SimpleNamespace(
        torch=SimpleNamespace(device=_Device, cuda=_FakeCuda),
        clean_memory_on_device=lambda _device: None,
    )

    def load(*_args):
        model = _FakeModel()
        models.append(model)
        return model, None

    module.load_transformer_stage = load

    def run_one(args, task, device):
        transformer, _loader = module.load_transformer_stage(args, task, device)
        module.clean_memory_on_device(device)
        return transformer

    module.run_one = run_one
    install_native_transformer_cache(module)

    first = module.run_one(_args(), "t2va", "cuda:0")
    second = module.run_one(_args(), "fl2va", "cuda:0")

    assert first is not second
    assert len(models) == 2


def test_native_transformer_restore_failure_falls_back_to_cold_load(monkeypatch):
    models = []

    class _RestoreFailureModel(_FakeModel):
        def __init__(self):
            super().__init__()
            self.cuda_moves = 0

        def to(self, device):
            if str(device).startswith("cuda"):
                self.cuda_moves += 1
                # The post-job warm placement succeeds; fail only when the
                # next generation tries to restore the cached object.
                if self.cuda_moves == 2:
                    raise RuntimeError("unsupported quantized transfer")
            return super().to(device)

    def load(*_args):
        model = _RestoreFailureModel() if not models else _FakeModel()
        models.append(model)
        return model, None

    module = SimpleNamespace(
        torch=SimpleNamespace(device=_Device, cuda=_FakeCuda),
        load_transformer_stage=load,
        clean_memory_on_device=lambda _device: None,
    )

    def run_one(args, task, device):
        transformer, _loader = module.load_transformer_stage(args, task, device)
        module.clean_memory_on_device(device)
        return transformer

    module.run_one = run_one
    package = ModuleType("minimax_video")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "minimax_video", package)
    monkeypatch.setitem(
        sys.modules,
        "minimax_video.attention",
        SimpleNamespace(set_attention_backend=lambda _mode: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "minimax_video.transformer",
        SimpleNamespace(set_act_chunk_rows=lambda _rows: None),
    )
    install_native_transformer_cache(module)

    first = module.run_one(_args(), "t2va", "cuda:0")
    second = module.run_one(_args(), "t2va", "cuda:0")

    assert first is models[0]
    assert second is models[1]
    assert len(models) == 2


def test_native_transformer_cache_is_disabled_for_unsupported_placement():
    models = []
    module = SimpleNamespace(
        torch=SimpleNamespace(device=_Device, cuda=_FakeCuda),
        clean_memory_on_device=lambda _device: None,
    )

    def load(*_args):
        model = _FakeModel()
        models.append(model)
        return model, None

    module.load_transformer_stage = load

    def run_one(args, task, device):
        transformer, _loader = module.load_transformer_stage(args, task, device)
        module.clean_memory_on_device(device)
        return transformer

    module.run_one = run_one
    install_native_transformer_cache(module)

    module.run_one(_args(blocks_to_swap=4), "t2va", "cuda:0")
    module.run_one(_args(blocks_to_swap=4), "t2va", "cuda:0")

    assert len(models) == 2
