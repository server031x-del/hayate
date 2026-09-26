from pathlib import Path

from colab.setup_native_h3 import SAGEATTENTION_SOURCE, _sageattention_install_command


def test_sageattention_installer_uses_visible_logs_and_pinned_upstream_source(tmp_path: Path):
    command = _sageattention_install_command("/usr/bin/python3", tmp_path / "torch-constraints.txt")

    assert command[:4] == ["/usr/bin/python3", "-m", "pip", "install"]
    assert "--no-build-isolation" in command
    assert "--no-deps" in command
    assert "-q" not in command
    assert command[-1] == SAGEATTENTION_SOURCE
    assert SAGEATTENTION_SOURCE.startswith("git+https://github.com/thu-ml/SageAttention.git@")
    assert len(SAGEATTENTION_SOURCE.rsplit("@", maxsplit=1)[-1]) == 40
