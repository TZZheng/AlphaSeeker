"""Benchmark helpers for the harness runtime."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

from src.harness.runtime import run_harness
from src.harness.types import HarnessRequest, HarnessResponse


@dataclass(frozen=True)
class BenchmarkCase:
    """One benchmark prompt for the harness runtime."""

    name: str
    prompt: str


@dataclass(frozen=True)
class BenchmarkLane:
    """One model-lane configuration for a benchmark suite."""

    name: str
    model_name: str


@dataclass(frozen=True)
class BenchmarkMetrics:
    """Summary metrics derived from one harness run."""

    artifact_creation_success: bool
    contract_satisfaction: bool
    citation_coverage: str
    missing_section_count: int
    freshness_failure_count: int
    numeric_inconsistency_count: int
    counterevidence_gap_count: int


@dataclass(frozen=True)
class BenchmarkResult:
    """One benchmark result row."""

    case: BenchmarkCase
    lane: BenchmarkLane
    response: HarnessResponse
    metrics: BenchmarkMetrics


DEFAULT_BENCHMARK_CASES: tuple[BenchmarkCase, ...] = (
    BenchmarkCase("equity_aapl", "Analyze AAPL valuation and risk using current evidence."),
    BenchmarkCase("equity_macro_banks", "How do higher rates affect JPM and bank margins?"),
    BenchmarkCase("macro_outlook", "US macro outlook for the next 12 months."),
    BenchmarkCase("commodity_crude", "Crude oil supply-demand and futures curve outlook."),
    BenchmarkCase("latest_copper", "What is the latest evidence on copper demand and supply risks?"),
)

WEAK_MODEL_LANE = BenchmarkLane("weak", "sf/Qwen/Qwen3-8B")
STRONG_MODEL_LANE = BenchmarkLane("strong", "kimi-k2.5")


@contextmanager
def harness_model_lane(lane: BenchmarkLane) -> Iterator[None]:
    """Temporarily point harness model roles at a single configured model."""

    role_names = [
        "PLANNER",
        "PLANNER_COMPILE",
        "CONTRACT_BUILDER",
        "CONTROLLER",
        "WRITER",
        "VERIFY",
    ]
    env_keys = [f"ALPHASEEKER_MODEL_HARNESS_{role}" for role in role_names]
    previous = {key: os.environ.get(key) for key in env_keys}
    try:
        for key in env_keys:
            os.environ[key] = lane.model_name
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _load_trace_metrics(response: HarnessResponse) -> BenchmarkMetrics:
    if not response.trace_path or not Path(response.trace_path).exists():
        return BenchmarkMetrics(
            artifact_creation_success=False,
            contract_satisfaction=False,
            citation_coverage="fail",
            missing_section_count=0,
            freshness_failure_count=0,
            numeric_inconsistency_count=0,
            counterevidence_gap_count=0,
        )
    payload = json.loads(Path(response.trace_path).read_text(encoding="utf-8"))
    verification = payload.get("verification_reports", [])
    latest = verification[-1] if verification else {}
    return BenchmarkMetrics(
        artifact_creation_success=all(
            bool(payload.get(path_key))
            for path_key in ("mission_path", "progress_path", "plan_path", "contract_path", "trace_path", "report_path")
        ),
        contract_satisfaction=latest.get("decision") == "pass",
        citation_coverage=str(latest.get("citation_coverage", "fail")),
        missing_section_count=len(latest.get("missing_sections", [])),
        freshness_failure_count=len(latest.get("freshness_warnings", [])),
        numeric_inconsistency_count=len(latest.get("numeric_inconsistencies", [])),
        counterevidence_gap_count=len(latest.get("counterevidence_gaps", [])),
    )


def run_benchmark_suite(
    cases: tuple[BenchmarkCase, ...] = DEFAULT_BENCHMARK_CASES,
    lanes: tuple[BenchmarkLane, ...] = (WEAK_MODEL_LANE, STRONG_MODEL_LANE),
    run_case_fn: Callable[[HarnessRequest], HarnessResponse] = run_harness,
) -> list[BenchmarkResult]:
    """Run a benchmark suite across one or more model lanes."""

    results: list[BenchmarkResult] = []
    for lane in lanes:
        with harness_model_lane(lane):
            for case in cases:
                response = run_case_fn(HarnessRequest(user_prompt=case.prompt, benchmark_label=lane.name))
                metrics = _load_trace_metrics(response)
                results.append(BenchmarkResult(case=case, lane=lane, response=response, metrics=metrics))
    return results
