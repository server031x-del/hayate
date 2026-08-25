from __future__ import annotations

import torch

from hayate.loaders.tensor_source import CheckpointTensorSource
from safetensors.torch import save_file


def test_pread_tensor_source_drains_owned_tensors(tmp_path):
    path = tmp_path / "model.safetensors"
    save_file({"first": torch.arange(4), "second": torch.ones(2)}, path)

    with CheckpointTensorSource(path, backend="pread") as source:
        assert source.backend == "pread"
        assert set(source.keys()) == {"first", "second"}
        first = source.take_tensor("first")
        assert first.tolist() == [0, 1, 2, 3]
        assert source.remaining == 1


def test_mmap_tensor_source_can_detach_before_close(tmp_path):
    path = tmp_path / "model.safetensors"
    save_file({"weight": torch.arange(4)}, path)

    with CheckpointTensorSource(path, backend="mmap") as source:
        value = source.take_tensor("weight", copy_mmap=True)
    assert value.tolist() == [0, 1, 2, 3]
