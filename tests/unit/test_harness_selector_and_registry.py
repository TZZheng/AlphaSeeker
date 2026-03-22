from __future__ import annotations

import pytest

from src.harness.registry import build_skill_registry, get_skills_for_packs
from src.harness.selector import select_packs
from src.supervisor.router import AgentTask, ClassificationResult

pytestmark = pytest.mark.unit


def test_select_packs_uses_supervisor_classifier() -> None:
    classification = ClassificationResult(
        primary_intent="equity",
        tasks=[
            AgentTask(agent_type="equity", ticker="AAPL"),
            AgentTask(agent_type="macro", topic="US rates"),
        ],
        reasoning="cross-domain",
    )

    packs = select_packs("Analyze AAPL with macro context", classify_fn=lambda _prompt: classification)

    assert packs == ["core", "equity", "macro"]


def test_select_packs_falls_back_to_all_domains() -> None:
    packs = select_packs("Fallback", classify_fn=lambda _prompt: (_ for _ in ()).throw(RuntimeError("boom")))

    assert packs == ["core", "equity", "macro", "commodity"]


def test_registry_exposes_core_and_domain_skills() -> None:
    registry = build_skill_registry()

    assert "search_and_read" in registry
    assert registry["search_and_read"].pack == "core"
    assert "fetch_company_profile" in registry
    assert registry["fetch_company_profile"].pack == "equity"
    assert "fetch_macro_indicators" in registry
    assert registry["fetch_macro_indicators"].pack == "macro"
    assert "fetch_eia_inventory" in registry
    assert registry["fetch_eia_inventory"].pack == "commodity"


def test_get_skills_for_packs_filters_to_enabled_packs() -> None:
    registry = build_skill_registry()

    enabled = get_skills_for_packs(registry, ["core", "macro"])
    enabled_names = {spec.name for spec in enabled}

    assert "search_and_read" in enabled_names
    assert "fetch_macro_indicators" in enabled_names
    assert "fetch_company_profile" not in enabled_names
    assert "fetch_eia_inventory" not in enabled_names
