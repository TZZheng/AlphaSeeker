"""Commodity-specific harness skill adapters."""

from __future__ import annotations

from typing import Any

from src.tools.commodity.cftc import fetch_cot_report
from src.tools.commodity.eia import fetch_eia_inventory
from src.tools.commodity.futures import fetch_futures_curve
from src.harness.skills.common import artifact_evidence, make_result, safe_read, skill_artifact_dir
from src.harness.types import HarnessState, SkillMetrics, SkillResult, SkillSpec


def fetch_eia_inventory_skill(arguments: dict[str, Any], _state: HarnessState) -> SkillResult:
    asset = str(arguments.get("asset") or "").strip()
    if not asset:
        return make_result(
            "fetch_eia_inventory",
            arguments,
            status="failed",
            summary="fetch_eia_inventory requires an asset.",
            error="Missing asset.",
        )

    path, metadata = fetch_eia_inventory(asset, output_dir=skill_artifact_dir(_state, "commodity"))
    if not path:
        return make_result(
            "fetch_eia_inventory",
            arguments,
            status="partial",
            summary=f"EIA has no artifact for asset '{asset}'.",
            details={"asset": asset, "metadata": metadata},
            error="No EIA artifact generated.",
        )
    text = safe_read(path, max_chars=5000)
    return make_result(
        "fetch_eia_inventory",
        arguments,
        status="ok",
        summary=f"Fetched EIA inventory data for {asset}.",
        details={"asset": asset, "metadata": metadata, "path": path},
        metrics=SkillMetrics(
            evidence_count=1,
            artifact_count=1,
            sections_touched=["Commodity Balance"],
        ),
        output_text=text,
        artifacts=[path],
        evidence=[artifact_evidence("fetch_eia_inventory", f"EIA inventory data for {asset}.", path, content=text, metadata=metadata)],
    )


def fetch_cot_report_skill(arguments: dict[str, Any], _state: HarnessState) -> SkillResult:
    asset = str(arguments.get("asset") or "").strip()
    num_weeks = int(arguments.get("num_weeks", 12))
    if not asset:
        return make_result(
            "fetch_cot_report",
            arguments,
            status="failed",
            summary="fetch_cot_report requires an asset.",
            error="Missing asset.",
        )

    path, metadata = fetch_cot_report(
        asset,
        num_weeks=num_weeks,
        output_dir=skill_artifact_dir(_state, "commodity"),
    )
    if not path:
        return make_result(
            "fetch_cot_report",
            arguments,
            status="partial",
            summary=f"No COT artifact was generated for {asset}.",
            details={"asset": asset, "metadata": metadata},
            error="No COT artifact generated.",
        )
    text = safe_read(path, max_chars=5000)
    return make_result(
        "fetch_cot_report",
        arguments,
        status="ok",
        summary=f"Fetched CFTC COT positioning for {asset}.",
        details={"asset": asset, "metadata": metadata, "path": path},
        metrics=SkillMetrics(
            evidence_count=1,
            artifact_count=1,
            sections_touched=["Curve and Positioning", "Risks and Counterevidence"],
        ),
        output_text=text,
        artifacts=[path],
        evidence=[artifact_evidence("fetch_cot_report", f"COT positioning for {asset}.", path, content=text, metadata=metadata)],
    )


def fetch_futures_curve_skill(arguments: dict[str, Any], _state: HarnessState) -> SkillResult:
    asset = str(arguments.get("asset") or "").strip()
    num_contracts = int(arguments.get("num_contracts", 12))
    if not asset:
        return make_result(
            "fetch_futures_curve",
            arguments,
            status="failed",
            summary="fetch_futures_curve requires an asset.",
            error="Missing asset.",
        )

    path, metadata = fetch_futures_curve(
        asset,
        num_contracts=num_contracts,
        output_dir=skill_artifact_dir(_state, "commodity"),
    )
    if not path:
        return make_result(
            "fetch_futures_curve",
            arguments,
            status="partial",
            summary=f"No futures-curve artifact was generated for {asset}.",
            details={"asset": asset, "metadata": metadata},
            error="No futures-curve artifact generated.",
        )
    text = safe_read(path, max_chars=5000)
    return make_result(
        "fetch_futures_curve",
        arguments,
        status="ok",
        summary=f"Fetched futures curve data for {asset}.",
        details={"asset": asset, "metadata": metadata, "path": path},
        metrics=SkillMetrics(
            evidence_count=1,
            artifact_count=1,
            sections_touched=["Curve and Positioning"],
        ),
        output_text=text,
        artifacts=[path],
        evidence=[artifact_evidence("fetch_futures_curve", f"Futures curve for {asset}.", path, content=text, metadata=metadata)],
    )


COMMODITY_SKILLS = [
    SkillSpec(
        name="fetch_eia_inventory",
        description="Fetch EIA inventory, production, import, and spot-price data for supported energy commodities.",
        pack="commodity",
        input_schema={
            "type": "object",
            "properties": {"asset": {"type": "string", "description": "Energy asset such as crude oil or natural gas."}},
            "required": ["asset"],
        },
        produces_artifacts=True,
        executor=fetch_eia_inventory_skill,
    ),
    SkillSpec(
        name="fetch_cot_report",
        description="Fetch CFTC Commitments of Traders positioning for futures-market sentiment and crowding.",
        pack="commodity",
        input_schema={
            "type": "object",
            "properties": {
                "asset": {"type": "string", "description": "Commodity such as crude oil, gold, copper, corn."},
                "num_weeks": {"type": "integer", "default": 12, "minimum": 2},
            },
            "required": ["asset"],
        },
        produces_artifacts=True,
        executor=fetch_cot_report_skill,
    ),
    SkillSpec(
        name="fetch_futures_curve",
        description="Fetch futures-curve prices and contango/backwardation structure for supported commodities.",
        pack="commodity",
        input_schema={
            "type": "object",
            "properties": {
                "asset": {"type": "string", "description": "Commodity such as crude oil, gold, or natural gas."},
                "num_contracts": {"type": "integer", "default": 12, "minimum": 2},
            },
            "required": ["asset"],
        },
        produces_artifacts=True,
        executor=fetch_futures_curve_skill,
    ),
]
