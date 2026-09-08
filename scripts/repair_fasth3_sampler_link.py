from __future__ import annotations

import json
from pathlib import Path

p = Path(r"M:\Project\HAYATE\data\comfyui-user\default\workflows\MiniMaxH3 I2V.json")
d = json.loads(p.read_text(encoding="utf-8"))
sg = d["definitions"]["subgraphs"][0]
if d.get("last_node_id") != 150 or d.get("last_link_id") != 264:
    raise SystemExit("unexpected FastH3 sampler workflow version")
for existing in sg["links"]:
    if existing.get("id") == 16:
        existing.update({"origin_id": 123, "origin_slot": 0, "target_id": 150, "target_slot": 0, "type": "SAMPLER"})
        break
else:
    raise SystemExit("original sampler link 16 missing")
p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("repaired link 16: normal res_multistep -> FastH3 sampler switch")
