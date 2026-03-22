"""Commodity-specific harness skill adapters."""

from __future__ import annotations

from typing import Any

from src.agents.commodity.tools.cftc import fetch_cot_report
from src.agents.commodity.tools.eia import fetch_eia_inventory
from src.agents.commodity.tools.futures import fetch_futures_curve
from src.harness.skills.common import artifact_evidence, make_result, safe_read
from src.harness.types import HarnessState, SkillResult, SkillSpec


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

    path, metadata = fetch_eia_inventory(asset)
    if not path:
        return make_result(
            "fetch_eia_inventory",
            arguments,
            status="partial",
            summary=f"EIA has no artifact for asset '{asset}'.",
            structured_data={"asset": asset, "metadata": metadata},
            error="No EIA artifact generated.",
        )
    text = safe_read(path, max_chars=5000)
    return make_result(
        "fetch_eia_inventory",
        arguments,
        status="ok",
        summary=f"Fetched EIA inventory data for {asset}.",
        structured_data={"asset": asset, "metadata": metadata, "path": path},
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

    path, metadata = fetch_cot_report(asset, num_weeks=num_weeks)
    if not path:
        return make_result(
            "fetch_cot_report",
            arguments,
            status="partial",
            summary=f"No COT artifact was generated for {asset}.",
            structured_data={"asset": asset, "metadata": metadata},
            error="No COT artifact generated.",
        )
    text = safe_read(path, max_chars=5000)
    return make_result(
        "fetch_cot_report",
        arguments,
        status="ok",
        summary=f"Fetched CFTC COT positioning for {asset}.",
        structured_data={"asset": asset, "metadata": metadata, "path": path},
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

    path, metadata = fetch_futures_curve(asset, num_contracts=num_contracts)
    if not path:
        return make_result(
            "fetch_futures_curve",
            arguments,
            status="partial",
            summary=f"No futures-curve artifact was generated for {asset}.",
            structured_data={"asset": asset, "metadata": metadata},
            error="No futures-curve artifact generated.",
        )
    text = safe_read(path, max_chars=5000)
    return make_result(
        "fetch_futures_curve",
        arguments,
        status="ok",
        summary=f"Fetched futures curve data for {asset}.",
        structured_data={"asset": asset, "metadata": metadata, "path": path},
        output_text=text,
        artifacts=[path],
        evidence=[artifact_evidence("fetch_futures_curve", f"Futures curve for {asset}.", path, content=text, metadata=metadata)],
    )


COMMODITY_SKILLS = [
    SkillSpec(
        name="fetch_eia_inventory",
        description="Fetch EIA inventory and production data for an energy commodity.",
        pack="commodity",
        input_schema={"asset": "string"},
        produces_artifacts=True,
        executor=fetch_eia_inventory_skill,
    ),
    SkillSpec(
        name="fetch_cot_report",
        description="Fetch CFTC COT positioning for a commodity futures market.",
        pack="commodity",
        input_schema={"asset": "string", "num_weeks": "integer"},
        produces_artifacts=True,
        executor=fetch_cot_report_skill,
    ),
    SkillSpec(
        name="fetch_futures_curve",
        description="Fetch futures-curve structure for a commodity.",
        pack="commodity",
        input_schema={"asset": "string", "num_contracts": "integer"},
        produces_artifacts=True,
        executor=fetch_futures_curve_skill,
    ),
]
