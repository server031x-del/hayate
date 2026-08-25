from __future__ import annotations

from hayate.backends.minimax_h3.upstream import H3UpstreamAdapter, REQUIRED_COMPONENTS


def test_upstream_adapter_validates_contract_without_importing_models(tmp_path):
    for relative in REQUIRED_COMPONENTS.values():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n", encoding="utf-8")
    validation = H3UpstreamAdapter(tmp_path).validate()
    assert validation.valid
    assert validation.commit is None

