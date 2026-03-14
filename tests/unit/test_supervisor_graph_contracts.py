from __future__ import annotations

from pathlib import Path

import pytest
from langgraph.types import Send

from src.shared.schemas import SubAgentRequest
from src.supervisor.graph import (
    _as_classified_state,
    _extract_report,
    route_to_agents,
    synthesize_results,
    validate_routing_state,
)
from src.supervisor.synthesizer import SynthesisOutput

pytestmark = pytest.mark.unit


class _SectionObj:
    def __init__(self, title: str, content: str) -> None:
        self.title = title
        self.content = content


class _ReportModelLike:
    def model_dump(self) -> dict:
        return {
            "summary": {"title": "Investment Summary", "content": "Summary content."},
            "recommendation": "BUY",
        }


def _state_with_routes() -> dict:
    return {
        "user_prompt": "Analyze AAPL",
        "intent": "equity",
        "sub_agents_needed": ["equity", "macro"],
        "classified_entities": {
            "equity": SubAgentRequest(user_prompt="Analyze AAPL", ticker="AAPL"),
            "macro": SubAgentRequest(user_prompt="Analyze AAPL", topic="rates"),
        },
    }


def test_validate_routing_state_requires_sub_agents() -> None:
    state = {"user_prompt": "x", "classified_entities": {}}

    out = validate_routing_state(state)

    assert out["error"] == "No sub-agents selected by classifier"


def test_validate_routing_state_requires_entities() -> None:
    state = {"user_prompt": "x", "sub_agents_needed": ["equity"]}

    out = validate_routing_state(state)

    assert out["error"] == "Classifier did not produce classified_entities"


def test_validate_routing_state_requires_matching_requests() -> None:
    state = {
        "user_prompt": "x",
        "sub_agents_needed": ["equity", "macro"],
        "classified_entities": {
            "equity": SubAgentRequest(user_prompt="Analyze AAPL", ticker="AAPL"),
        },
    }

    out = validate_routing_state(state)

    assert "Missing SubAgentRequest for: macro" == out["error"]


def test_validate_routing_state_success() -> None:
    out = validate_routing_state(_state_with_routes())
    assert out == {}


def test_as_classified_state_success() -> None:
    classified = _as_classified_state(_state_with_routes())
    assert classified["intent"] == "equity"
    assert "equity" in classified["classified_entities"]


def test_as_classified_state_raises_when_missing_required_fields() -> None:
    with pytest.raises(ValueError, match="routing state is incomplete"):
        _as_classified_state({"user_prompt": "x"})


def test_route_to_agents_returns_send_fanout() -> None:
    sends = route_to_agents(_state_with_routes())

    assert isinstance(sends, list)
    assert all(isinstance(s, Send) for s in sends)
    assert [s.node for s in sends] == ["run_equity_agent", "run_macro_agent"]


def test_route_to_agents_returns_handle_error_on_state_error() -> None:
    out = route_to_agents({"user_prompt": "x", "error": "bad"})
    assert out == "handle_error"


def test_route_to_agents_returns_handle_error_when_no_valid_agents() -> None:
    out = route_to_agents(
        {
            "user_prompt": "x",
            "sub_agents_needed": ["unknown-agent"],
        }
    )
    assert out == "handle_error"


def test_extract_report_prefers_file_output(tmp_path: Path) -> None:
    report = tmp_path / "report.md"
    report.write_text("# File report", encoding="utf-8")

    out = _extract_report({"report_path": str(report)}, agent_type="equity")

    assert out == "# File report"


def test_extract_report_uses_report_content_fallback() -> None:
    out = _extract_report({"report_content": _ReportModelLike()}, agent_type="equity")

    assert out.startswith("# Partial Equity Report")
    assert "## Investment Summary" in out
    assert "**recommendation:** BUY" in out


def test_extract_report_uses_sections_fallback() -> None:
    out = _extract_report(
        {
            "sections": {
                "risk": _SectionObj(title="Risks", content="Main risks."),
                "misc": "Extra text",
            }
        },
        agent_type="macro",
    )

    assert out.startswith("# Partial Macro Report (incomplete)")
    assert "## Risks" in out
    assert "## misc" in out


def test_extract_report_uses_error_fallback() -> None:
    out = _extract_report({"error": "upstream failed"}, agent_type="commodity")

    assert out == "**Commodity Agent:** Pipeline failed — upstream failed"


def test_synthesize_results_success_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_run_synthesis(_):
        return SynthesisOutput(
            final_response="Unified answer",
            mode="synthesis",
            agents_used=["equity", "macro"],
        )

    monkeypatch.setattr("src.supervisor.graph.run_synthesis", _fake_run_synthesis)

    out = synthesize_results(
        {
            "user_prompt": "Analyze AAPL with rates context",
            "intent": "equity",
            "agent_results": {"equity": "eq report", "macro": "macro report"},
        }
    )

    assert out == {"final_response": "Unified answer"}


def test_synthesize_results_returns_error_contract_on_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(_):
        raise RuntimeError("llm down")

    monkeypatch.setattr("src.supervisor.graph.run_synthesis", _boom)

    out = synthesize_results(
        {
            "user_prompt": "Analyze AAPL",
            "intent": "equity",
            "agent_results": {"equity": "eq report"},
        }
    )

    assert out["final_response"].startswith("Synthesis failed:")
    assert out["error"] == "llm down"


def test_synthesize_results_empty_agent_results_contract() -> None:
    out = synthesize_results({"user_prompt": "Analyze AAPL"})

    assert out == {
        "final_response": "No agent produced results.",
        "error": "Empty agent_results",
    }
