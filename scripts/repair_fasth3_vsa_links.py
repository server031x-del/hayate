from __future__ import annotations

import json
from pathlib import Path


p = Path(r"M:\Project\HAYATE\data\comfyui-user\default\workflows\MiniMaxH3 I2V.json")
d = json.loads(p.read_text(encoding="utf-8"))
sg = d["definitions"]["subgraphs"][0]
nodes = {n["id"]: n for n in sg["nodes"]}
if nodes[144]["type"] != "MiniMaxH3SigmaShift" or nodes[143]["type"] != "H3VSA":
    raise SystemExit("unexpected FastH3 node types")
nodes[144]["outputs"][0]["links"] = [252]
nodes[143]["outputs"][0]["links"] = [253]
p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("repaired FastH3 link output metadata")
