"""Patch the existing MiniMax H3 I2V subgraph for FastH3 VSA.

This is deliberately a guarded, one-shot migration.  It refuses to touch a
workflow whose expected node/link ids have changed, so an upstream/user edit
cannot be silently overwritten.
"""

from __future__ import annotations

import json
from pathlib import Path


WORKFLOW = Path(r"M:\Project\HAYATE\data\comfyui-user\default\workflows\MiniMaxH3 I2V.json")


def _node(node_id: int, node_type: str, pos: list[float], size: list[float],
          order: int, inputs: list[dict], outputs: list[dict], *, title: str | None = None,
          widgets_values=None, widgets_values_named=None, properties=None) -> dict:
    result = {
        "id": node_id,
        "type": node_type,
        "pos": pos,
        "size": size,
        "flags": {},
        "order": order,
        "mode": 0,
        "inputs": inputs,
        "outputs": outputs,
        "properties": properties or {"Node name for S&R": node_type, "cnr_id": "comfy-core", "ver": "0.34.0"},
    }
    if title:
        result["title"] = title
    if widgets_values is not None:
        result["widgets_values"] = widgets_values
    if widgets_values_named is not None:
        result["widgets_values_named"] = widgets_values_named
    return result


def _link(link_id: int, origin_id: int, origin_slot: int, target_id: int,
          target_slot: int, link_type: str) -> dict:
    return {
        "id": link_id,
        "origin_id": origin_id,
        "origin_slot": origin_slot,
        "target_id": target_id,
        "target_slot": target_slot,
        "type": link_type,
    }


def _set_input(node: dict, name: str, link_id: int | None) -> None:
    for item in node.get("inputs", []):
        if item.get("name") == name:
            item["link"] = link_id
            return
    raise KeyError(f"input {name!r} missing on node {node.get('id')}")


def _set_output_links(node: dict, links: list[int]) -> None:
    if not node.get("outputs"):
        raise KeyError(f"node {node.get('id')} has no output")
    node["outputs"][0]["links"] = links or None


def main() -> None:
    data = json.loads(WORKFLOW.read_text(encoding="utf-8"))
    if data.get("last_node_id") != 142 or data.get("last_link_id") != 250:
        raise SystemExit(
            "Refusing migration: expected last_node_id=142,last_link_id=250; "
            f"got {data.get('last_node_id')},{data.get('last_link_id')}")
    if any(n.get("type") == "H3VSA" for n in data.get("nodes", [])):
        raise SystemExit("Workflow is already patched (H3VSA found in outer graph)")

    subgraphs = data.get("definitions", {}).get("subgraphs", [])
    if len(subgraphs) != 1 or subgraphs[0].get("name") != "Image to Video (MiniMax H3)":
        raise SystemExit("Refusing migration: MiniMax H3 I2V subgraph not found")
    subgraph = subgraphs[0]
    nodes = {n["id"]: n for n in subgraph.get("nodes", [])}
    required_nodes = {123, 124, 126, 127, 134, 135, 136, 137, 138, 139}
    if not required_nodes.issubset(nodes):
        raise SystemExit(f"Refusing migration: missing expected nodes {required_nodes - set(nodes)}")
    if any(n["id"] in {143, 144, 145, 146, 147, 148} for n in subgraph.get("nodes", [])):
        raise SystemExit("Workflow is already patched (new ids found)")

    # Keep the embedded subgraph fallback aligned with the FastH3 model used
    # by the outer MiniMax H3 node.  The outer input is normally connected,
    # but an explicit valid default avoids a stale/missing UNET selection when
    # the subgraph is opened or reused independently.
    nodes[127]["widgets_values"] = [
        "minimax_h3_fastvideo_vsa_datafree_1300step_4step_int8_convrot.safetensors",
        "default",
    ]
    nodes[127]["widgets_values_named"] = {
        "unet_name": "minimax_h3_fastvideo_vsa_datafree_1300step_4step_int8_convrot.safetensors",
        "weight_dtype": "default",
    }

    # Existing normal/Turbo branch remains intact.  FastH3 uses the base model
    # directly, then applies the official MiniMax shifts and VSA patch.
    new_nodes = [
        _node(
            144, "MiniMaxH3SigmaShift", [-40, 4600], [300, 130], 21,
            [
                {"name": "model", "type": "MODEL", "link": 251},
                {"name": "shift_video", "type": "FLOAT", "widget": {"name": "shift_video"}, "link": None},
                {"name": "shift_audio", "type": "FLOAT", "widget": {"name": "shift_audio"}, "link": None},
            ],
            [{"name": "MODEL", "type": "MODEL", "links": [252]}],
            title="FastH3 Sigma Shift (video 12 / audio 3)",
            widgets_values=[12.0, 3.0],
            widgets_values_named={"shift_video": 12.0, "shift_audio": 3.0},
            properties={"Node name for S&R": "MiniMaxH3SigmaShift", "cnr_id": "comfy-core", "ver": "0.34.0"},
        ),
        _node(
            143, "H3VSA", [300, 4580], [330, 180], 22,
            [
                {"name": "model", "type": "MODEL", "link": 252},
                {"name": "gate_file", "type": "COMBO", "widget": {"name": "gate_file"}, "link": None},
                {"name": "topk_ratio", "type": "FLOAT", "widget": {"name": "topk_ratio"}, "link": None},
                {"name": "min_tokens", "type": "INT", "widget": {"name": "min_tokens"}, "link": None},
            ],
            [{"name": "MODEL", "type": "MODEL", "links": [253]}],
            title="FastH3 VSA (10% attention / embedded gates)",
            widgets_values=["<model-embedded-gates>", 0.10, 4096],
            widgets_values_named={"gate_file": "<model-embedded-gates>", "topk_ratio": 0.10, "min_tokens": 4096},
            properties={"Node name for S&R": "H3VSA", "ver": "0.1.0"},
        ),
        _node(
            145, "PrimitiveBoolean", [300, 4840], [300, 90], 23,
            [{"name": "value", "type": "BOOLEAN", "widget": {"name": "value"}, "link": None}],
            [{"name": "BOOLEAN", "type": "BOOLEAN", "links": [255, 258]}],
            title="FastH3 VSA mode (Turbo LoRA OFF)",
            widgets_values=[True],
            widgets_values_named={"value": True},
            properties={"Node name for S&R": "PrimitiveBoolean", "cnr_id": "comfy-core", "ver": "0.34.0"},
        ),
        _node(
            146, "ComfySwitchNode", [680, 4700], [310, 110], 24,
            [
                {"name": "on_false", "type": "MODEL", "link": 254},
                {"name": "on_true", "type": "MODEL", "link": 253},
                {"name": "switch", "type": "BOOLEAN", "widget": {"name": "switch"}, "link": 255},
            ],
            [{"name": "MODEL", "type": "MODEL", "links": [256, 257]}],
            title="If/Else Switch (FastH3 / normal-Turbo)",
            widgets_values=[True],
            widgets_values_named={"switch": True},
            properties={"Node name for S&R": "ComfySwitchNode", "cnr_id": "comfy-core", "ver": "0.34.0"},
        ),
        _node(
            147, "PrimitiveInt", [680, 4900], [300, 90], 25,
            [{"name": "value", "type": "INT", "widget": {"name": "value"}, "link": None}],
            [{"name": "INT", "type": "INT", "links": [259]}],
            title="FastH3 steps (fixed 4)",
            widgets_values=[4, "fixed"],
            widgets_values_named={"value": 4, "fixed": "fixed"},
            properties={"Node name for S&R": "PrimitiveInt", "cnr_id": "comfy-core", "ver": "0.34.0"},
        ),
        _node(
            148, "ComfySwitchNode", [1030, 4820], [310, 110], 26,
            [
                {"name": "on_false", "type": "INT", "link": 260},
                {"name": "on_true", "type": "INT", "link": 259},
                {"name": "switch", "type": "BOOLEAN", "widget": {"name": "switch"}, "link": 258},
            ],
            [{"name": "INT", "type": "INT", "links": [261]}],
            title="If/Else Switch (FastH3 4-step)",
            widgets_values=[True],
            widgets_values_named={"switch": True},
            properties={"Node name for S&R": "ComfySwitchNode", "cnr_id": "comfy-core", "ver": "0.34.0"},
        ),
    ]
    subgraph["nodes"].extend(new_nodes)

    # The old links are intentionally removed from their consumers and then
    # recreated through the FastH3 switches.  This prevents an accidental
    # second Turbo-LoRA evaluation when FastH3 is selected.
    _set_input(nodes[124], "model", 256)
    _set_input(nodes[124], "steps", 261)
    _set_input(nodes[126], "model", 257)
    _set_output_links(nodes[135], [254])
    _set_output_links(nodes[136], [260])
    _set_output_links(nodes[127], [229, 232, 251])

    remove_ids = {234, 235, 241}
    subgraph["links"] = [l for l in subgraph.get("links", []) if l.get("id") not in remove_ids]
    subgraph["links"].extend([
        _link(251, 127, 0, 144, 0, "MODEL"),
        _link(252, 144, 0, 143, 0, "MODEL"),
        _link(253, 143, 0, 146, 1, "MODEL"),
        _link(254, 135, 0, 146, 0, "MODEL"),
        _link(255, 145, 0, 146, 2, "BOOLEAN"),
        _link(256, 146, 0, 124, 0, "MODEL"),
        _link(257, 146, 0, 126, 0, "MODEL"),
        _link(258, 145, 0, 148, 2, "BOOLEAN"),
        _link(259, 147, 0, 148, 1, "INT"),
        _link(260, 136, 0, 148, 0, "INT"),
        _link(261, 148, 0, 124, 2, "INT"),
    ])
    subgraph["state"]["lastNodeId"] = 148
    subgraph["state"]["lastLinkId"] = 261
    data["last_node_id"] = 148
    data["last_link_id"] = 261

    WORKFLOW.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"patched {WORKFLOW}")
    print("FastH3 graph: 127 -> 144 SigmaShift(12/3) -> 143 H3VSA(0.10/4096)")
    print("FastH3 switches: model and steps(4); existing normal/Turbo branch preserved")


if __name__ == "__main__":
    main()
