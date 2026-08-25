from __future__ import annotations

import argparse
import json
import struct
import urllib.request
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--header", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()
    with args.header.open("rb") as handle:
        header_length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_length))
    origin = 8 + header_length
    names = [name for name in header if name.endswith(".comfy_quant")][: args.limit]
    for name in names:
        start, end = header[name]["data_offsets"]
        request = urllib.request.Request(
            args.url,
            headers={"Range": f"bytes={origin + start}-{origin + end - 1}"},
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status != 206:
                raise RuntimeError(f"server ignored Range request: HTTP {response.status}")
            raw = response.read(end - start + 1)
        if len(raw) != end - start:
            raise RuntimeError(f"short marker payload for {name}")
        try:
            decoded = raw.rstrip(b"\0").decode("utf-8")
        except UnicodeDecodeError:
            decoded = raw.hex()
        print(json.dumps({"name": name, "bytes": len(raw), "value": decoded}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
