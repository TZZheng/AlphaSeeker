from __future__ import annotations

import pytest

from src.shared.node_contracts import NodeResult, contract_node, partial_patch

pytestmark = pytest.mark.unit


def test_contract_node_marks_successful_patch() -> None:
    @contract_node("sample")
    def _node(_state):
        return {"value": 42}

    out = _node({})

    assert out["value"] == 42
    assert isinstance(out["last_node_result"], NodeResult)
    assert out["last_node_result"].status == "ok"
    assert out["node_results"]["sample"].status == "ok"


def test_contract_node_marks_partial_patch() -> None:
    @contract_node("sample")
    def _node(_state):
        return partial_patch({"value": None}, error="dependency unavailable")

    out = _node({})

    assert out["last_node_result"].status == "partial"
    assert out["last_node_result"].error == "dependency unavailable"


def test_contract_node_marks_exception_as_failed() -> None:
    @contract_node("sample")
    def _node(_state):
        raise RuntimeError("boom")

    out = _node({})

    assert out["error"] == "boom"
    assert out["last_node_result"].status == "failed"
    assert "RuntimeError" in out["last_node_result"].error
