"""Benchmark helpers for the file-based harness kernel."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, ConfigDict, Field

from src.harness import HarnessRequest, run_harness
from src.harness.artifacts import registry_paths
from src.harness.types import HarnessResponse


class BenchmarkCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    prompt: str
    requested_packs: list[str] = Field(default_factory=list)


class BenchmarkLane(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lane_id: str
    request_overrides: dict[str, object] = Field(default_factory=dict)


class BenchmarkMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runtime_status: str
    stop_reason: str = "unknown"
    agent_count: int = 0
    done_count: int = 0
    failed_count: int = 0
    stale_count: int = 0
    event_count: int = 0
    object_count: int = 0
    root_final_exists: bool = False


class BenchmarkResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str
    lane_id: str
    run_root: str | None = None
    final_report_path: str | None = None
    metrics: BenchmarkMetrics


DEFAULT_BENCHMARK_CASES = [
    BenchmarkCase(case_id="equity_single_name", prompt="Analyze AAPL valuation and risk using current evidence.", requested_packs=["core", "equity"]),
    BenchmarkCase(case_id="macro_outlook", prompt="US macro outlook for the next 12 months.", requested_packs=["core", "macro"]),
    BenchmarkCase(case_id="commodity_outlook", prompt="Crude oil supply-demand and futures curve outlook.", requested_packs=["core", "commodity"]),
]


DEFAULT_BENCHMARK_LANES = [
    BenchmarkLane(lane_id="default"),
    BenchmarkLane(lane_id="wide", request_overrides={"max_agents_per_run": 96, "max_live_children_per_parent": 12}),
]


def _read_jsonl_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _latest_agents(run_root: str) -> dict[str, dict[str, object]]:
    path = registry_paths(run_root)["agents_registry"]
    latest: dict[str, dict[str, object]] = {}
    if not path.exists():
        return latest
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict) and isinstance(payload.get("agent_id"), str):
            latest[payload["agent_id"]] = payload
    return latest


def extract_metrics(response: HarnessResponse) -> BenchmarkMetrics:
    if not response.run_root:
        return BenchmarkMetrics(runtime_status=response.status, stop_reason=response.stop_reason or "unknown")

    latest = _latest_agents(response.run_root)
    counts = {"done": 0, "failed": 0, "stale": 0}
    for payload in latest.values():
        status = str(payload.get("status", ""))
        if status in counts:
            counts[status] += 1
    paths = registry_paths(response.run_root)
    return BenchmarkMetrics(
        runtime_status=response.status,
        stop_reason=response.stop_reason or "unknown",
        agent_count=len(latest),
        done_count=counts["done"],
        failed_count=counts["failed"],
        stale_count=counts["stale"],
        event_count=_read_jsonl_count(paths["events_registry"]),
        object_count=_read_jsonl_count(paths["objects_manifest"]),
        root_final_exists=bool(response.final_report_path and Path(response.final_report_path).exists()),
    )


def run_benchmark_suite(
    *,
    cases: list[BenchmarkCase] | None = None,
    lanes: list[BenchmarkLane] | None = None,
    run_fn: Callable[[HarnessRequest], HarnessResponse] = run_harness,
) -> list[BenchmarkResult]:
    benchmark_cases = cases or DEFAULT_BENCHMARK_CASES
    benchmark_lanes = lanes or DEFAULT_BENCHMARK_LANES
    results: list[BenchmarkResult] = []
    for lane in benchmark_lanes:
        for case in benchmark_cases:
            request = HarnessRequest(
                user_prompt=case.prompt,
                available_skill_packs=case.requested_packs or None,
                **lane.request_overrides,
            )
            response = run_fn(request)
            results.append(
                BenchmarkResult(
                    case_id=case.case_id,
                    lane_id=lane.lane_id,
                    run_root=response.run_root,
                    final_report_path=response.final_report_path,
                    metrics=extract_metrics(response),
                )
            )
    return results
