from __future__ import annotations

import json
import subprocess
import sys

from hayate.kernels.comfy_kitchen import probe_w4a8_kernel


def test_kernel_probe_parses_isolated_json_result():
    def runner(*args, **kwargs):
        return subprocess.CompletedProcess(
            args[0],
            0,
            "diagnostic\n" + json.dumps({"ok": True, "operation_executed": True}),
            "",
        )

    result = probe_w4a8_kernel(sys.executable, runner=runner)
    assert result.ok
    assert result.operation_executed
