"""Add a FastH3-only Euler sampler switch to the patched I2V workflow."""

from __future__ import annotations

import json
from pathlib import Path


WORKFLOW = Path(r"M:\Project\HAYATE\data\comfyui-user\default\workflows\MiniMaxH3 I2V.json")


def node(node_id, node_type, pos, size, order, inputs, outputs, title, widgets, named=None):
    item = {
        "id": node_id, "type": node_type, "pos": pos, "size": size,
        "flags": {}, "order": order, "mode": 0, "inputs": inputs,
        "outputs": outputs,
        "properties": {"Node name for S&R": node_type, "cnr_id": "comfy-core", "ver": "0.34.0"},
        "title": title, "widgets_values": widgets,
    }
    if named is not None:
        item["widgets_values_named"] = named
    return item


def link(i, origin, oslot, target, tslot, typ):
    return {"id": i, "origin_id": origin, "origin_slot": oslot, "target_id": target, "target_slot": tslot, "type": typ}


p = WORKFLOW
d = json.loads(p.read_text(encoding="utf-8"))
if d.get("last_node_id") != 148 or d.get("last_link_id") != 261:
    raise SystemExit(f"unexpected workflow version: {d.get('last_node_id')},{d.get('last_link_id')}")
sg = d["definitions"]["subgraphs"][0]
nodes = {n["id"]: n for n in sg["nodes"]}
if 145 not in nodes or nodes[145].get("type") != "PrimitiveBoolean":
    raise SystemExit("FastH3 mode switch not found")
if any(n["id"] in {149, 150} for n in sg["nodes"]):
    raise SystemExit("FastH3 sampler switch is already present")

sg["nodes"].extend([
    node(149, "KSamplerSelect", [1030, 4630], [280, 90], 27,
         [{"name": "sampler_name", "type": "COMBO", "widget": {"name": "sampler_name"}, "link": None}],
         [{"name": "SAMPLER", "type": "SAMPLER", "links": [262]}],
         "FastH3 sampler (Euler)", ["euler"], {"sampler_name": "euler"}),
    node(150, "ComfySwitchNode", [1370, 4700], [330, 110], 28,
         [{"name": "on_false", "type": "SAMPLER", "link": 16},
          {"name": "on_true", "type": "SAMPLER", "link": 262},
          {"name": "switch", "type": "BOOLEAN", "widget": {"name": "switch"}, "link": 263}],
         [{"name": "SAMPLER", "type": "SAMPLER", "links": [264]}],
         "If/Else Switch (FastH3 Euler / normal-Turbo)", [True], {"switch": True}),
])

for item in nodes[125]["inputs"]:
    if item.get("name") == "sampler":
        item["link"] = 264
        break
else:
    raise SystemExit("SamplerCustomAdvanced sampler input missing")
nodes[123]["outputs"][0]["links"] = [16]
nodes[145]["outputs"][0]["links"] = [255, 258, 263]
# Link 16 originally fed SamplerCustomAdvanced; it now feeds the switch's
# normal/Turbo branch and keeps the original res_multistep path intact.
for existing in sg["links"]:
    if existing.get("id") == 16:
        existing.update({"origin_id": 123, "origin_slot": 0, "target_id": 150, "target_slot": 0, "type": "SAMPLER"})
        break
else:
    raise SystemExit("original KSamplerSelect link 16 missing")
sg["links"].extend([
    link(262, 149, 0, 150, 1, "SAMPLER"),
    link(263, 145, 0, 150, 2, "BOOLEAN"),
    link(264, 150, 0, 125, 2, "SAMPLER"),
])
sg["state"]["lastNodeId"] = 150
sg["state"]["lastLinkId"] = 264
d["last_node_id"] = 150
d["last_link_id"] = 264
p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"patched {p}: FastH3 selects Euler; normal/Turbo keeps res_multistep")
