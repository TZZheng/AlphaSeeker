from __future__ import annotations

import pytest

from src.shared.model_config import get_missing_provider_env_vars
from src.supervisor.graph import app as supervisor_app

pytestmark = [pytest.mark.live, pytest.mark.network]


def _assert_live_model_env_ready() -> None:
    missing = get_missing_provider_env_vars()
    if missing:
        formatted = ", ".join(f"{provider}: {env_req}" for provider, env_req in missing.items())
        pytest.fail(
            "Live pipeline test cannot run because required model-provider keys are missing: "
            f"{formatted}"
        )


def _assert_agent_result_contract(agent_results: dict[str, str], required_agents: list[str]) -> None:
    assert isinstance(agent_results, dict)
    for agent in required_agents:
        assert agent in agent_results, f"Expected agent '{agent}' in agent_results, got {list(agent_results.keys())}"
        content = agent_results[agent]
        assert isinstance(content, str) and content.strip(), f"Agent '{agent}' output is empty"
        assert not content.startswith(f"**{agent.title()} Agent Error:**"), (
            f"Agent '{agent}' returned an explicit error payload: {content[:200]}"
        )


def test_live_full_pipeline_single_domain_passthrough() -> None:
    """
    End-to-end live run through the supervisor graph for a single-domain request.
    Verifies at least the equity sub-agent pipeline executes and final response is non-empty.
    """
    _assert_live_model_env_ready()

    prompt = (
        "Analyze AAPL equity valuation, business quality, and company-specific risks only. "
        "Do not include macroeconomic or commodity analysis."
    )
    final_state = supervisor_app.invoke({"user_prompt": prompt})

    assert not final_state.get("error"), f"Supervisor returned error: {final_state.get('error')}"
    final_response = final_state.get("final_response", "")
    assert isinstance(final_response, str) and final_response.strip()

    agent_results = final_state.get("agent_results", {})
    _assert_agent_result_contract(agent_results, required_agents=["equity"])


def test_live_full_pipeline_multi_domain_synthesis() -> None:
    """
    End-to-end live run through the full supervisor pipeline with all three domains.
    Verifies all sub-agents run and multi-agent synthesis produces a non-empty response.
    """
    _assert_live_model_env_ready()

    prompt = (
        "Create one integrated investment brief that combines: "
        "(1) AAPL equity view, (2) US interest-rate macro outlook, and "
        "(3) crude-oil commodity outlook for the next 12 months."
    )
    final_state = supervisor_app.invoke({"user_prompt": prompt})

    assert not final_state.get("error"), f"Supervisor returned error: {final_state.get('error')}"
    final_response = final_state.get("final_response", "")
    assert isinstance(final_response, str) and final_response.strip()

    agent_results = final_state.get("agent_results", {})
    _assert_agent_result_contract(
        agent_results,
        required_agents=["equity", "macro", "commodity"],
    )

    # Multi-domain prompt should route to at least 2 agents, triggering synthesis mode.
    assert len(agent_results) >= 2
    assert not final_response.startswith("# Response to:"), (
        "Expected synthesized multi-agent output, got single-agent passthrough format."
    )
