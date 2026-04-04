"""Macro-specific harness skill adapters."""

from __future__ import annotations

from typing import Any

from src.agents.macro.tools.fred import fetch_macro_indicators
from src.agents.macro.tools.world_bank import fetch_world_bank_indicators
from src.harness.skills.common import artifact_evidence, make_result, safe_read
from src.harness.types import HarnessState, SkillMetrics, SkillResult, SkillSpec


def fetch_macro_indicators_skill(arguments: dict[str, Any], _state: HarnessState) -> SkillResult:
    topic = str(arguments.get("topic") or "").strip()
    countries = arguments.get("countries") or []
    if not topic:
        return make_result(
            "fetch_macro_indicators",
            arguments,
            status="failed",
            summary="fetch_macro_indicators requires a topic.",
            error="Missing topic.",
        )

    path, metadata = fetch_macro_indicators(topic=topic, countries=list(countries))
    if not path:
        return make_result(
            "fetch_macro_indicators",
            arguments,
            status="partial",
            summary=f"No macro-indicator artifact was produced for '{topic}'.",
            details={"topic": topic, "countries": list(countries), "metadata": metadata},
            error="Indicator source unavailable.",
        )
    text = safe_read(path, max_chars=5000)
    return make_result(
        "fetch_macro_indicators",
        arguments,
        status="ok",
        summary=f"Fetched macro indicators for '{topic}'.",
        details={"topic": topic, "countries": list(countries), "metadata": metadata, "path": path},
        metrics=SkillMetrics(
            evidence_count=1,
            artifact_count=1,
            sections_touched=["Macro Transmission", "Scenarios"],
        ),
        output_text=text,
        artifacts=[path],
        evidence=[artifact_evidence("fetch_macro_indicators", f"Macro indicators for {topic}.", path, content=text, metadata=metadata)],
    )


def fetch_world_bank_indicators_skill(arguments: dict[str, Any], _state: HarnessState) -> SkillResult:
    countries = list(arguments.get("countries") or [])
    indicator_codes = arguments.get("indicator_codes")
    date_range = str(arguments.get("date_range") or "2019:2025")
    if not countries:
        return make_result(
            "fetch_world_bank_indicators",
            arguments,
            status="failed",
            summary="fetch_world_bank_indicators requires at least one country.",
            error="Missing countries.",
        )

    path, metadata = fetch_world_bank_indicators(
        countries=countries,
        indicator_codes=indicator_codes,
        date_range=date_range,
    )
    if not path:
        return make_result(
            "fetch_world_bank_indicators",
            arguments,
            status="partial",
            summary="World Bank indicators returned no artifact.",
            details={"countries": countries, "metadata": metadata},
            error="World Bank source unavailable.",
        )
    text = safe_read(path, max_chars=5000)
    return make_result(
        "fetch_world_bank_indicators",
        arguments,
        status="ok",
        summary=f"Fetched World Bank indicators for {', '.join(countries)}.",
        details={"countries": countries, "metadata": metadata, "path": path},
        metrics=SkillMetrics(
            evidence_count=1,
            artifact_count=1,
            sections_touched=["Macro Transmission", "Scenarios"],
        ),
        output_text=text,
        artifacts=[path],
        evidence=[artifact_evidence("fetch_world_bank_indicators", f"World Bank indicators for {', '.join(countries)}.", path, content=text, metadata=metadata)],
    )


MACRO_SKILLS = [
    SkillSpec(
        name="fetch_macro_indicators",
        description="Fetch FRED-based macro indicators for a topic and country set.",
        pack="macro",
        input_schema={"topic": "string", "countries": "string[]"},
        produces_artifacts=True,
        executor=fetch_macro_indicators_skill,
    ),
    SkillSpec(
        name="fetch_world_bank_indicators",
        description="Fetch cross-country World Bank indicator tables.",
        pack="macro",
        input_schema={
            "countries": "string[]",
            "indicator_codes": "string[]",
            "date_range": "string",
        },
        produces_artifacts=True,
        executor=fetch_world_bank_indicators_skill,
    ),
]
