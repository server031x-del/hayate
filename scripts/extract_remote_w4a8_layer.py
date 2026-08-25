from __future__ import annotations

import argparse
import json
import struct
import urllib.request
from pathlib import Path


SUFFIXES = (".weight", ".weight_s_rel", ".weight_s_channel", ".weight_codebook")


def fetch_range(url: str, start: int, end: int) -> bytes:
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end - 1}"})
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.status != 206:
            raise RuntimeError(f"server did not honor Range request: HTTP {response.status}")
        data = response.read(end - start + 1)
    expected = end - start
    if len(data) != expected:
        raise RuntimeError(f"short Range response: expected {expected}, got {len(data)}")
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--header", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--layer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.header.open("rb") as handle:
        header_length = struct.unpack("<Q", handle.read(8))[0]
        source = json.loads(handle.read(header_length))
    quantization = json.loads(source["__metadata__"]["_quantization_metadata"])
    if args.layer not in quantization["layers"]:
        raise KeyError(f"layer is not declared in source metadata: {args.layer}")

    output_header = {}
    payloads = []
    cursor = 0
    payload_origin = 8 + header_length
    for suffix in SUFFIXES:
        name = args.layer + suffix
        descriptor = source[name]
        source_start, source_end = descriptor["data_offsets"]
        payload = fetch_range(
            args.url,
            payload_origin + source_start,
            payload_origin + source_end,
        )
        output_header[name] = {
            "dtype": descriptor["dtype"],
            "shape": descriptor["shape"],
            "data_offsets": [cursor, cursor + len(payload)],
        }
        payloads.append(payload)
        cursor += len(payload)
    output_header["__metadata__"] = {
        "_quantization_metadata": json.dumps(
            {"layers": {args.layer: quantization["layers"][args.layer]}},
            separators=(",", ":"),
        )
    }
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
