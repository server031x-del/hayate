from __future__ import annotations

from hayate.runtime.gpu_lease import GPULease


def test_gpu_lease_blocks_a_second_owner_and_releases(tmp_path):
    path = tmp_path / "gpu.lock"
    first = GPULease(path, owner={"pid": 100, "kind": "first"})
    second = GPULease(path, owner={"pid": 200, "kind": "second"})

    assert first.acquire()
    assert not second.acquire()
    assert second.busy_owner()["pid"] == 100

    first.release()
    assert second.acquire()
    second.release()
