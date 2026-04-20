#!/usr/bin/env python3
"""Run or summarize the XOM patch-editing validation flow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.harness import HarnessRequest, run_harness
from src.harness.artifacts import agent_workspace_paths, registry_paths
from src.harness.benchmark import extract_metrics
from src.harness.types import HarnessResponse


DEFAULT_PROMPT = (
    "Write a cross-domain investment memo on XOM using current evidence. "
    "Assess valuation, balance-sheet and shareholder-return quality, crude-oil "
    "supply-demand and futures-curve drivers, and the U.S. macro backdrop. "
    "Explain the main bull and bear cases, key quantitative evidence, and the "
    "12-month risk/reward."
)
DEFAULT_BASELINE_RUN_ID = "xom-edit-baseline"
DEFAULT_POSTFIX_RUN_ID = "xom-edit-patch-v1"
EDIT_TOOLS = {"edit_file", "apply_patch", "write_file"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _run_root_from_id(run_id: str) -> Path:
    return REPO_ROOT / "data" / "harness_runs" / run_id


def _latest_agents(run_root: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    agents_path = registry_paths(run_root)["agents_registry"]
    for row in _read_jsonl(agents_path):
        agent_id = row.get("agent_id")
        if isinstance(agent_id, str):
            latest[agent_id] = row
    return latest


def _synthetic_stop_reason(run_root: Path, root_status: str) -> str:
    events_path = registry_paths(run_root)["events_registry"]
    for event in reversed(_read_jsonl(events_path)):
        if str(event.get("event_type") or "") != "root_stop_forced":
            continue
        if str(event.get("agent_id") or "") != "agent_root":
            continue
        details = event.get("details") or event.get("payload") or {}
        stop_reason = str(details.get("stop_reason") or "").strip()
        if stop_reason:
            return stop_reason
    return root_status


def _synthetic_response(run_root: Path) -> HarnessResponse:
    latest = _latest_agents(run_root)
    root_status = str(latest.get("agent_root", {}).get("status") or "unknown")
    stop_reason = _synthetic_stop_reason(run_root, root_status)
    root_paths = agent_workspace_paths(run_root, "agent_root")
    final_path = root_paths["publish_final"]
    final_exists = final_path.exists()
    if stop_reason == "done" and final_exists:
        status = "completed"
    elif stop_reason == "wall_clock_budget_exhausted":
        status = "time_out_with_deliverable" if final_exists else "time_out"
    else:
        status = "failed"
    return HarnessResponse(
        status=status,
        stop_reason=stop_reason,
        run_root=str(run_root),
        root_agent_path=str(root_paths["workspace"]),
        final_report_path=str(final_path) if final_exists else None,
        error=None,
    )


def _tool_name_from_event(event: dict[str, Any]) -> str:
    details = event.get("details") or event.get("payload") or {}
    return str(details.get("tool") or details.get("tool_name") or "")


def _result_from_event(event: dict[str, Any]) -> dict[str, Any]:
    details = event.get("details") or event.get("payload") or {}
    result = details.get("result")
    return result if isinstance(result, dict) else {}


def _arguments_from_event(event: dict[str, Any]) -> dict[str, Any]:
    details = event.get("details") or event.get("payload") or {}
    arguments = details.get("arguments")
    return arguments if isinstance(arguments, dict) else {}


def _error_from_event(event: dict[str, Any]) -> str:
    details = event.get("details") or event.get("payload") or {}
    return str(details.get("error") or "")


def _collect_edit_events(run_root: Path) -> list[dict[str, Any]]:
    events_path = registry_paths(run_root)["events_registry"]
    rows: list[dict[str, Any]] = []
    for event in _read_jsonl(events_path):
        event_type = str(event.get("event_type") or "")
        tool_name = _tool_name_from_event(event)
        if event_type not in {"tool_completed", "tool_failed"} or tool_name not in EDIT_TOOLS:
            continue
        result = _result_from_event(event)
        arguments = _arguments_from_event(event)
        rows.append(
            {
                "created_at": event.get("created_at") or event.get("timestamp"),
                "event_type": event_type,
                "agent_id": event.get("agent_id"),
                "tool_name": tool_name,
                "path": result.get("path") or arguments.get("path") or "",
                "operation": result.get("operation") or arguments.get("operation") or "",
                "hunks_applied": result.get("hunks_applied", 0),
                "error": _error_from_event(event),
            }
        )
    return rows


def _coverage_report(final_text: str) -> dict[str, bool]:
    lower = final_text.lower()

    def has_any(*needles: str) -> bool:
        return any(needle in lower for needle in needles)

    checks = {
        "valuation": has_any("valuation", "p/e", "ev/ebitda"),
        "balance_sheet_shareholder_return": has_any("balance sheet", "debt", "buyback", "dividend", "shareholder return"),
        "crude_supply_demand": has_any("supply-demand", "opec", "inventory", "oil demand"),
        "futures_curve": has_any("futures curve", "contango", "backwardation"),
        "macro_backdrop": has_any("macro", "federal reserve", "gdp", "u.s. dollar"),
        "bull_case": has_any("bull case", "bull thesis"),
        "bear_case": has_any("bear case", "bear thesis"),
        "quantitative_evidence": has_any("%", "$", "mbpd", "b/d", "bps", "x "),
        "twelve_month_risk_reward": has_any("12-month", "12 month", "risk/reward", "upside", "downside"),
    }
    checks["all_requested_topics_present"] = all(checks.values())
    return checks


def summarize_run(run_root: Path, *, response: HarnessResponse | None = None) -> dict[str, Any]:
    response_obj = response or _synthetic_response(run_root)
    metrics = extract_metrics(response_obj).model_dump(mode="json")
    latest = _latest_agents(run_root)
    root_paths = agent_workspace_paths(run_root, "agent_root")
    final_path = root_paths["publish_final"]
    final_text = final_path.read_text(encoding="utf-8") if final_path.exists() else ""
    edit_events = _collect_edit_events(run_root)

    tool_attempt_counts: dict[str, int] = {name: 0 for name in sorted(EDIT_TOOLS)}
    tool_failure_counts: dict[str, int] = {name: 0 for name in sorted(EDIT_TOOLS)}
    for event in edit_events:
        tool_name = str(event["tool_name"])
        tool_attempt_counts[tool_name] = tool_attempt_counts.get(tool_name, 0) + 1
        if event["event_type"] == "tool_failed":
            tool_failure_counts[tool_name] = tool_failure_counts.get(tool_name, 0) + 1

    return {
        "run_root": str(run_root),
        "response": response_obj.model_dump(mode="json"),
        "metrics": metrics,
        "agent_statuses": {agent_id: record.get("status") for agent_id, record in sorted(latest.items())},
        "root_publish": {
            "final_path": str(final_path),
            "final_exists": final_path.exists(),
            "summary_exists": root_paths["publish_summary"].exists(),
            "artifact_index_exists": root_paths["publish_index"].exists(),
        },
        "edit_related": {
            "events": edit_events,
            "tool_attempt_counts": tool_attempt_counts,
            "tool_failure_counts": tool_failure_counts,
            "failure_count": sum(1 for event in edit_events if event["event_type"] == "tool_failed"),
            "edit_file_failure_count": tool_failure_counts.get("edit_file", 0),
            "used_patch_or_full_rewrite": bool(
                tool_attempt_counts.get("apply_patch", 0) or tool_attempt_counts.get("write_file", 0)
            ),
        },
        "coverage": _coverage_report(final_text) if final_text else {},
    }


def run_case(run_id: str, prompt: str) -> dict[str, Any]:
    response = run_harness(HarnessRequest(user_prompt=prompt, run_id=run_id))
    return summarize_run(Path(response.run_root or _run_root_from_id(run_id)), response=response)


def compare_runs(before_run_root: Path, after_run_root: Path) -> dict[str, Any]:
    before = summarize_run(before_run_root)
    after = summarize_run(after_run_root)
    return {
        "before": before,
        "after": after,
        "comparison": {
            "stop_reason": {
                "before": before["response"]["stop_reason"],
                "after": after["response"]["stop_reason"],
            },
            "root_final_exists": {
                "before": before["root_publish"]["final_exists"],
                "after": after["root_publish"]["final_exists"],
            },
            "edit_related_failure_count": {
                "before": before["edit_related"]["failure_count"],
                "after": after["edit_related"]["failure_count"],
            },
            "edit_file_failure_count": {
                "before": before["edit_related"]["edit_file_failure_count"],
                "after": after["edit_related"]["edit_file_failure_count"],
            },
            "tool_attempt_counts": {
                "before": before["edit_related"]["tool_attempt_counts"],
                "after": after["edit_related"]["tool_attempt_counts"],
            },
            "used_patch_or_full_rewrite_after": after["edit_related"]["used_patch_or_full_rewrite"],
            "coverage": {
                "before": before["coverage"],
                "after": after["coverage"],
            },
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _resolve_run_root(*, run_id: str | None, run_root: str | None) -> Path:
    if run_root:
        return Path(run_root).expanduser().resolve()
    if run_id:
        return _run_root_from_id(run_id)
    raise ValueError("Either run_id or run_root is required.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the live XOM validation case and emit a summary.")
    run_parser.add_argument("--run-id", required=True)
    run_parser.add_argument("--prompt", default=DEFAULT_PROMPT)

    summarize_parser = subparsers.add_parser("summarize", help="Summarize an existing harness run.")
    summarize_parser.add_argument("--run-id")
    summarize_parser.add_argument("--run-root")

    compare_parser = subparsers.add_parser("compare", help="Compare two existing harness runs.")
    compare_parser.add_argument("--before-run-id", default=DEFAULT_BASELINE_RUN_ID)
    compare_parser.add_argument("--before-run-root")
    compare_parser.add_argument("--after-run-id", default=DEFAULT_POSTFIX_RUN_ID)
    compare_parser.add_argument("--after-run-root")

    args = parser.parse_args()

    if args.command == "run":
        payload = run_case(args.run_id, args.prompt)
        _write_json(_run_root_from_id(args.run_id) / "validation_summary.json", payload)
    elif args.command == "summarize":
        run_root = _resolve_run_root(run_id=args.run_id, run_root=args.run_root)
        payload = summarize_run(run_root)
        _write_json(run_root / "validation_summary.json", payload)
    else:
        before_run_root = _resolve_run_root(run_id=args.before_run_id, run_root=args.before_run_root)
        after_run_root = _resolve_run_root(run_id=args.after_run_id, run_root=args.after_run_root)
        payload = compare_runs(before_run_root, after_run_root)
        _write_json(after_run_root / "validation_comparison.json", payload)

    print(json.dumps(payload, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
