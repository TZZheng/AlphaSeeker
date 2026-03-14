from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from src.shared.schemas import SubAgentRequest
from src.supervisor import graph as supervisor_graph
from src.supervisor.synthesizer import SynthesisOutput

pytestmark = pytest.mark.component


class _FakeApp:
    def __init__(self, result: dict[str, Any] | None = None, error: Exception | None = None) -> None:
        self._result = result or {}
        self._error = error

    def invoke(self, _input: dict[str, Any]) -> dict[str, Any]:
        if self._error is not None:
            raise self._error
        return self._result


def _make_classified_state() -> dict[str, Any]:
    return {
        "user_prompt": "Analyze AAPL with macro context",
        "intent": "equity",
        "sub_agents_needed": ["equity", "macro"],
        "classified_entities": {
            "equity": SubAgentRequest(user_prompt="Analyze AAPL", ticker="AAPL"),
            "macro": SubAgentRequest(user_prompt="Analyze AAPL", topic="US rates"),
        },
    }


def test_supervisor_component_happy_path_with_stubbed_agents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eq_module = types.SimpleNamespace(
        app=_FakeApp(
            result={
                "report_content": {
                    "summary": {"title": "Equity Summary", "content": "Equity details"}
                }
            }
        )
    )
    macro_module = types.SimpleNamespace(
        app=_FakeApp(
            result={
                "report_content": {
                    "overview": {"title": "Macro Overview", "content": "Macro details"}
                }
            }
        )
    )

    monkeypatch.setitem(sys.modules, "src.agents.equity.graph", eq_module)
    monkeypatch.setitem(sys.modules, "src.agents.macro.graph", macro_module)

    state = _make_classified_state()
    eq_out = supervisor_graph.run_equity_agent(state)
    macro_out = supervisor_graph.run_macro_agent(state)

    agent_results = {}
    agent_results.update(eq_out["agent_results"])
    agent_results.update(macro_out["agent_results"])

    def _fake_run_synthesis(_input):
        return SynthesisOutput(
            final_response="Unified supervisor output",
            mode="synthesis",
            agents_used=["equity", "macro"],
        )

    monkeypatch.setattr(supervisor_graph, "run_synthesis", _fake_run_synthesis)

    final = supervisor_graph.synthesize_results(
        {
            "user_prompt": state["user_prompt"],
            "intent": state["intent"],
            "agent_results": agent_results,
        }
    )

    assert "equity" in agent_results
    assert "macro" in agent_results
    assert final == {"final_response": "Unified supervisor output"}


def test_supervisor_component_partial_failure_still_synthesizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eq_module = types.SimpleNamespace(app=_FakeApp(error=RuntimeError("equity failure")))
    macro_module = types.SimpleNamespace(
        app=_FakeApp(result={"sections": {"policy": "Macro section content"}})
    )

    monkeypatch.setitem(sys.modules, "src.agents.equity.graph", eq_module)
    monkeypatch.setitem(sys.modules, "src.agents.macro.graph", macro_module)

    state = _make_classified_state()
    eq_out = supervisor_graph.run_equity_agent(state)
    macro_out = supervisor_graph.run_macro_agent(state)

    agent_results = {}
    agent_results.update(eq_out["agent_results"])
    agent_results.update(macro_out["agent_results"])

    def _fake_run_synthesis(_input):
        return SynthesisOutput(
            final_response="Final answer with partial data",
            mode="synthesis",
            agents_used=["equity", "macro"],
        )

    monkeypatch.setattr(supervisor_graph, "run_synthesis", _fake_run_synthesis)

    final = supervisor_graph.synthesize_results(
        {
            "user_prompt": state["user_prompt"],
            "intent": state["intent"],
            "agent_results": agent_results,
        }
    )

    assert agent_results["equity"].startswith("**Equity Agent Error:**")
    assert "Partial Macro Report" in agent_results["macro"]
    assert final["final_response"] == "Final answer with partial data"


def test_classification_validation_and_routing_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_classification = supervisor_graph.ClassificationResult(
        primary_intent="equity",
        tasks=[
            supervisor_graph.AgentTask(agent_type="equity", ticker="AAPL"),
            supervisor_graph.AgentTask(agent_type="macro", topic="US rates"),
        ],
        reasoning="cross-domain",
    )

    monkeypatch.setattr("src.supervisor.router.classify_user_prompt", lambda _prompt: fake_classification)

    classified = supervisor_graph.classify_intent({"user_prompt": "Analyze AAPL and rates"})
    valid = supervisor_graph.validate_routing_state({"user_prompt": "x", **classified})
    sends = supervisor_graph.route_to_agents({"user_prompt": "x", **classified})

    assert valid == {}
    assert isinstance(sends, list)
    assert [send.node for send in sends] == ["run_equity_agent", "run_macro_agent"]
