from __future__ import annotations

import torch
from safetensors.torch import save_file

from hayate.backends.minimax_h3.pdd import (
    EXPECTED_TARGETS,
    PDDCheckpointConfig,
    PDDLoRALinear,
    PDDParallelHead,
    pdd_time_grid,
    project_adaln_adapter,
)


def test_pdd_checkpoint_contract_and_nfe(tmp_path):
    tensors = {}
    for index in range(362):
        suffix = "adaln_proj.linear" if index < 50 else "branch.to_q"
        prefix = f"transformer_blocks.{index}.{suffix}"
        tensors[prefix + ".lora_down"] = torch.ones(1, 1, dtype=torch.bfloat16)
        tensors[prefix + ".lora_up"] = torch.ones(1, 1, dtype=torch.bfloat16)
    tensors.update(
        {
            "proj_out.weight": torch.ones(4, 1, 1, dtype=torch.bfloat16),
            "proj_out.bias": torch.zeros(4, 1, dtype=torch.bfloat16),
            "audio_proj_out.weight": torch.ones(4, 1, 1, dtype=torch.bfloat16),
            "audio_proj_out.bias": torch.zeros(4, 1, dtype=torch.bfloat16),
        }
    )
    path = tmp_path / "pdd.safetensors"
    save_file(
        tensors,
        path,
        metadata={
            "pdd_num_steps": "4",
            "pdd_block_size": "2",
            "lora_rank": "1",
            "lora_alpha": "1.0",
            "lora_targets": ",".join(EXPECTED_TARGETS),
        },
    )
    config = PDDCheckpointConfig.inspect(path)
    assert config.nfe == 2
    assert config.tensor_count == 728


def test_pdd_lora_keeps_base_separate_and_applies_constant_offset():
    base = torch.nn.Linear(2, 1, bias=False)
    base.weight.data.copy_(torch.tensor([[1.0, 1.0]]))
    layer = PDDLoRALinear.create(
        base,
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[2.0]]),
        0.5,
        offset=torch.tensor([4.0]),
    )
    result = layer(torch.tensor([[3.0, 5.0]]))
    assert torch.allclose(result, torch.tensor([[13.0]]))
    assert torch.equal(layer.base.weight, torch.tensor([[1.0, 1.0]]))


def test_pdd_parallel_head_fuses_only_the_active_interval():
    source = torch.nn.Linear(1, 1)
    weight = torch.tensor([[[1.0]], [[3.0]], [[100.0]], [[200.0]]])
    bias = torch.zeros(4, 1)
    head = PDDParallelHead.create(source, weight, bias, 4)
    head.set_interval(0, torch.tensor([0.25, 0.75]))
    assert torch.allclose(head(torch.tensor([[2.0]])), torch.tensor([[5.0]]))


def test_pdd_time_grid_is_monotonic_and_bounded():
    grid = pdd_time_grid(12.0, 32)
    assert grid.shape == (33,)
    assert grid[0] == 0
    assert grid[-1] == 1
    assert torch.all(grid[1:] > grid[:-1])


def test_released_adaln_projection_preserves_the_affine_lora_function():
    generator = torch.Generator().manual_seed(7)
    down = torch.randn(3, 5, generator=generator)
    up = torch.randn(4, 3, generator=generator)
    basis = torch.randn(2, 5, generator=generator)
    mean = torch.randn(5, generator=generator)
    coordinates = torch.randn(6, 2, generator=generator)
    projected, offset = project_adaln_adapter(down, up, basis, mean)
    released_input = mean + coordinates @ basis
    released = (released_input @ down.T) @ up.T
    pruned = (coordinates @ projected.T) @ up.T + offset
    assert torch.allclose(released, pruned, atol=2e-5, rtol=2e-5)
