"""Create the source bundle used to bootstrap HAYATE Studio in Colab.

The GitHub repository is private, so the Colab notebook cannot clone it without
credentials.  This helper creates a small, reproducible source archive that can
be uploaded to a Colab VM for a single session.  Model weights are deliberately
excluded; they are downloaded or linked by a separate runtime step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


ROOT_FILES = ("pyproject.toml", "README.md", "LICENSE")
ROOT_DIRECTORIES = ("configs", "hayate")


def iter_files(root: Path):
    for relative in ROOT_FILES:
        path = root / relative
        if path.is_file():
            yield path
    for directory in ROOT_DIRECTORIES:
        base = root / directory
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and not path.name.endswith(".pyc"):
                yield path


def build_bundle(root: Path, output: Path) -> tuple[int, int, str]:
    files = sorted(iter_files(root), key=lambda path: path.relative_to(root).as_posix())
    if not files:
        raise RuntimeError(f"no source files found under {root}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    return len(files), output.stat().st_size, digest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("colab-webui-bundle.zip"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    files, size, digest = build_bundle(root, (root / args.output).resolve() if not args.output.is_absolute() else args.output)
    print(json.dumps({"files": files, "bytes": size, "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
