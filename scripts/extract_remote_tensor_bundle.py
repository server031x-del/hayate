from __future__ import annotations

import argparse
import json
import struct
import urllib.request
from pathlib import Path


def fetch(url: str, start: int, end: int) -> bytes:
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end - 1}"})
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.status != 206:
            raise RuntimeError(f"server did not honor Range request: HTTP {response.status}")
        payload = response.read(end - start + 1)
    if len(payload) != end - start:
        raise RuntimeError(f"short response: expected {end - start}, got {len(payload)}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--header", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--tensor", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.header.open("rb") as handle:
        source_header_length = struct.unpack("<Q", handle.read(8))[0]
        source = json.loads(handle.read(source_header_length))
    origin = 8 + source_header_length
    output_header = {}
    payloads = []
    cursor = 0
    for name in args.tensor:
        descriptor = source[name]
        start, end = descriptor["data_offsets"]
        payload = fetch(args.url, origin + start, origin + end)
        output_header[name] = {
            "dtype": descriptor["dtype"],
            "shape": descriptor["shape"],
            "data_offsets": [cursor, cursor + len(payload)],
        }
        payloads.append(payload)
        cursor += len(payload)
    encoded = json.dumps(output_header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        for payload in payloads:
            handle.write(payload)
    print(f"saved {args.output} ({args.output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
