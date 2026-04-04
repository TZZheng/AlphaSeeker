from __future__ import annotations

from datetime import datetime

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

    assert "read_file" in registry
    assert registry["read_file"].pack == "core"
    assert "get_current_datetime" in registry
    assert registry["get_current_datetime"].pack == "core"
    assert "read_web_pages" in registry
    assert registry["read_web_pages"].pack == "core"
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

    assert "read_file" in enabled_names
    assert "get_current_datetime" in enabled_names
    assert "read_web_pages" in enabled_names
    assert "fetch_macro_indicators" in enabled_names
    assert "fetch_company_profile" not in enabled_names
    assert "fetch_eia_inventory" not in enabled_names


def test_get_current_datetime_skill_returns_absolute_date_fields() -> None:
    registry = build_skill_registry()
    spec = registry["get_current_datetime"]

    result = spec.executor({}, None)

    assert result.status == "ok"
    assert result.details["local_date"]
    assert result.details["utc_date"]
    assert result.details["timezone"]
    assert isinstance(result.details["unix_timestamp"], int)
    datetime.fromisoformat(result.details["local_iso"])
    datetime.fromisoformat(result.details["utc_iso"])
