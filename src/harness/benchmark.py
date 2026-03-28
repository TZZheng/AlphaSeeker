"""Benchmark set and metric extraction helpers for harness review."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, ConfigDict, Field
import yaml

from src.harness import HarnessRequest, run_harness
from src.harness.types import HarnessResponse


class BenchmarkCase(BaseModel):
    """One fixed benchmark prompt used for regression tracking."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    prompt: str
    expected_packs: list[str] = Field(default_factory=list)


class BenchmarkLane(BaseModel):
    """A model-lane configuration for the benchmark runner."""

    model_config = ConfigDict(extra="forbid")

    lane_id: str
    request_overrides: dict[str, object] = Field(default_factory=dict)


class BenchmarkMetrics(BaseModel):
    """Extracted metrics from one harness run."""

    model_config = ConfigDict(extra="forbid")

    artifact_creation_success: bool
    contract_satisfaction: str
    citation_coverage: str
    missing_section_count: int
    freshness_failures: int
    numeric_inconsistency_count: int
    counterevidence_gap_count: int
    runtime_status: str


class BenchmarkResult(BaseModel):
    """One benchmark result row."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    lane_id: str
    report_path: str | None = None
    trace_path: str | None = None
    metrics: BenchmarkMetrics


DEFAULT_BENCHMARK_CASES = [
    BenchmarkCase(
        case_id="equity_single_name",
        prompt="Analyze AAPL valuation and risk using current evidence.",
        expected_packs=["core", "equity"],
    ),
    BenchmarkCase(
        case_id="equity_macro_cross_domain",
        prompt="How do higher rates affect JPM and bank margins?",
        expected_packs=["core", "equity", "macro"],
    ),
    BenchmarkCase(
        case_id="macro_outlook",
        prompt="US macro outlook for the next 12 months.",
        expected_packs=["core", "macro"],
    ),
    BenchmarkCase(
        case_id="commodity_outlook",
        prompt="Crude oil supply-demand and futures curve outlook.",
        expected_packs=["core", "commodity"],
    ),
    BenchmarkCase(
        case_id="commodity_latest",
        prompt="What is the latest evidence on copper demand and supply risks?",
        expected_packs=["core", "commodity"],
    ),
]


DEFAULT_BENCHMARK_LANES = [
    BenchmarkLane(lane_id="weak", request_overrides={"max_steps": 6, "max_revision_rounds": 1}),
    BenchmarkLane(lane_id="strong", request_overrides={"max_steps": 8, "max_revision_rounds": 2}),
]


def extract_metrics(response: HarnessResponse) -> BenchmarkMetrics:
    """Summarize the saved QA report into benchmark-friendly metrics."""

    qa_payload: dict[str, object] = {}
    if response.dossier_paths.get("qa_report"):
        qa_path = Path(response.dossier_paths["qa_report"])
        if qa_path.exists():
            qa_payload = json.loads(qa_path.read_text(encoding="utf-8"))

    return BenchmarkMetrics(
        artifact_creation_success=all(
            [
                bool(response.report_path and Path(response.report_path).exists()),
                bool(response.trace_path and Path(response.trace_path).exists()),
                bool(response.dossier_paths.get("mission")),
                bool(response.dossier_paths.get("progress")),
                bool(response.dossier_paths.get("research_plan")),
                bool(response.dossier_paths.get("research_contract")),
                bool(response.dossier_paths.get("qa_report")),
            ]
        ),
        contract_satisfaction=str(qa_payload.get("decision", "unknown")),
        citation_coverage=str(qa_payload.get("citation_coverage", "unknown")),
        missing_section_count=len(qa_payload.get("missing_sections", [])),
        freshness_failures=len(qa_payload.get("freshness_warnings", [])),
        numeric_inconsistency_count=len(qa_payload.get("numeric_inconsistencies", [])),
        counterevidence_gap_count=len(qa_payload.get("counterevidence_gaps", [])),
        runtime_status=response.status,
    )


def run_benchmark_suite(
    *,
    cases: list[BenchmarkCase] | None = None,
    lanes: list[BenchmarkLane] | None = None,
    run_fn: Callable[[HarnessRequest], HarnessResponse] = run_harness,
) -> list[BenchmarkResult]:
    """Run the fixed benchmark set across one or more lanes."""

    benchmark_cases = cases or DEFAULT_BENCHMARK_CASES
    benchmark_lanes = lanes or DEFAULT_BENCHMARK_LANES
    results: list[BenchmarkResult] = []
    for lane in benchmark_lanes:
        for case in benchmark_cases:
            request = HarnessRequest(
                user_prompt=case.prompt,
                **lane.request_overrides,
            )
            response = run_fn(request)
            results.append(
                BenchmarkResult(
                    case_id=case.case_id,
                    lane_id=lane.lane_id,
                    report_path=response.report_path,
                    trace_path=response.trace_path,
                    metrics=extract_metrics(response),
                )
            )
    return results


def compare_benchmark_results(
    current_results: list[BenchmarkResult],
    baseline_results: list[BenchmarkResult],
) -> dict[str, dict[str, int]]:
    """Compare missing-section and counterevidence metrics against a baseline."""

    baseline_index = {(item.case_id, item.lane_id): item for item in baseline_results}
    comparison: dict[str, dict[str, int]] = {}
    for item in current_results:
        baseline_item = baseline_index.get((item.case_id, item.lane_id))
        if baseline_item is None:
            continue
        comparison[f"{item.case_id}:{item.lane_id}"] = {
            "missing_section_delta": item.metrics.missing_section_count
            - baseline_item.metrics.missing_section_count,
            "counterevidence_gap_delta": item.metrics.counterevidence_gap_count
            - baseline_item.metrics.counterevidence_gap_count,
            "freshness_failure_delta": item.metrics.freshness_failures
            - baseline_item.metrics.freshness_failures,
        }
    return comparison


def detect_model_role_changes(current_config_path: str, baseline_config_path: str) -> dict[str, tuple[str, str]]:
    """Return changed harness model-role assignments between two config files."""

    current = yaml.safe_load(Path(current_config_path).read_text(encoding="utf-8")) or {}
    baseline = yaml.safe_load(Path(baseline_config_path).read_text(encoding="utf-8")) or {}
    current_harness = current.get("harness", {})
    baseline_harness = baseline.get("harness", {})

    changed: dict[str, tuple[str, str]] = {}
    all_roles = set(current_harness) | set(baseline_harness)
    for role in sorted(all_roles):
        current_model = str(current_harness.get(role, ""))
        baseline_model = str(baseline_harness.get(role, ""))
        if current_model != baseline_model:
            changed[role] = (baseline_model, current_model)
    return changed
