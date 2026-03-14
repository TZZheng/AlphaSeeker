from __future__ import annotations

import os

import pytest

from src.supervisor.graph import classify_intent, synthesize_results

pytestmark = [pytest.mark.live, pytest.mark.network]


def _require_env(env_name: str) -> None:
    if not os.getenv(env_name):
        pytest.skip(f"Missing required env var for live test: {env_name}")


@pytest.mark.provider_kimi
def test_live_supervisor_smoke_kimi(monkeypatch: pytest.MonkeyPatch) -> None:
    _require_env("KIMI_API_KEY")

    monkeypatch.setenv("ALPHASEEKER_MODEL_SUPERVISOR_CLASSIFY", "kimi-k2.5")
    monkeypatch.setenv("ALPHASEEKER_MODEL_SUPERVISOR_SYNTHESIZE", "kimi-k2.5")

    classification = classify_intent({"user_prompt": "Analyze AAPL and interest rates."})

    assert "error" not in classification
    assert classification["sub_agents_needed"]

    synthesis = synthesize_results(
        {
            "user_prompt": "Analyze AAPL and interest rates.",
            "intent": classification.get("intent", "equity"),
            "agent_results": {
                "equity": "## Equity\nRevenue growth is improving.",
                "macro": "## Macro\nRates remain restrictive.",
            },
        }
    )

    assert "error" not in synthesis
    assert isinstance(synthesis.get("final_response"), str)
    assert synthesis["final_response"].strip() != ""


@pytest.mark.provider_sf
def test_live_supervisor_smoke_siliconflow(monkeypatch: pytest.MonkeyPatch) -> None:
    _require_env("SILICONFLOW_API_KEY")

    monkeypatch.setenv("ALPHASEEKER_MODEL_SUPERVISOR_CLASSIFY", "sf/Qwen/Qwen3-8B")
    monkeypatch.setenv("ALPHASEEKER_MODEL_SUPERVISOR_SYNTHESIZE", "sf/Qwen/Qwen3-8B")

    classification = classify_intent({"user_prompt": "Crude oil outlook next 12 months."})

    assert "error" not in classification
    assert classification["sub_agents_needed"]

    synthesis = synthesize_results(
        {
            "user_prompt": "Crude oil outlook next 12 months.",
            "intent": classification.get("intent", "commodity"),
            "agent_results": {
                "commodity": "## Commodity\nInventory draws support prices.",
                "macro": "## Macro\nGlobal growth is slowing.",
            },
        }
    )

    assert "error" not in synthesis
    assert isinstance(synthesis.get("final_response"), str)
    assert synthesis["final_response"].strip() != ""
