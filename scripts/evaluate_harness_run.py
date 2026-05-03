#!/usr/bin/env python3
"""Run and evaluate AlphaSeeker harness report versions."""

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

from src.harness.evaluator import evaluate_harness_run, run_eval_case


DEFAULT_XOM_PROMPT = (
    "Write a cross-domain investment memo on XOM using current evidence. "
    "Assess valuation, balance-sheet and shareholder-return quality, crude-oil "
    "supply-demand and futures-curve drivers, and the U.S. macro backdrop. "
    "Explain the main bull and bear cases, key quantitative evidence, and the "
    "12-month risk/reward."
)


def _run_root_from_id(run_id: str) -> Path:
    return REPO_ROOT / "data" / "harness_runs" / run_id


def _print_result(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=True))


def _request_overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for key in (
        "wall_clock_budget_seconds",
        "root_wall_clock_seconds",
        "per_agent_wall_clock_seconds",
        "max_agents_per_run",
        "max_live_agents",
        "max_live_children_per_parent",
        "commenter_interval_seconds",
    ):
        value = getattr(args, key, None)
        if value is not None:
            overrides[key] = value
    return overrides


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate an existing harness run.")
    eval_parser.add_argument("--run-id")
    eval_parser.add_argument("--run-root")
    eval_parser.add_argument("--case-id")
    eval_parser.add_argument("--eval-id")
    eval_parser.add_argument("--evaluator-model")
    eval_parser.add_argument("--acceptable-threshold", type=float, default=7.0)

    run_parser = subparsers.add_parser("run", help="Run a live harness case, then evaluate it.")
    run_parser.add_argument("--run-id", required=True)
    run_parser.add_argument("--case-id", default="xom_live_demo")
    run_parser.add_argument("--eval-id")
    run_parser.add_argument("--prompt", default=DEFAULT_XOM_PROMPT)
    run_parser.add_argument("--evaluator-model")
    run_parser.add_argument("--wall-clock-budget-seconds", type=int)
    run_parser.add_argument("--root-wall-clock-seconds", type=int)
    run_parser.add_argument("--per-agent-wall-clock-seconds", type=int)
    run_parser.add_argument("--max-agents-per-run", type=int)
    run_parser.add_argument("--max-live-agents", type=int)
    run_parser.add_argument("--max-live-children-per-parent", type=int)
    run_parser.add_argument("--commenter-interval-seconds", type=float)

    args = parser.parse_args()

    if args.command == "evaluate":
        if args.run_root:
            run_root = Path(args.run_root).expanduser().resolve()
        elif args.run_id:
            run_root = _run_root_from_id(args.run_id)
        else:
            raise SystemExit("Either --run-id or --run-root is required.")
        result = evaluate_harness_run(
            run_root,
            case_id=args.case_id,
            eval_id=args.eval_id,
            evaluator_model=args.evaluator_model,
            acceptable_threshold=args.acceptable_threshold,
        )
        _print_result(result.trajectory.model_dump(mode="json"))
        return 0

    response, result = run_eval_case(
        prompt=args.prompt,
        case_id=args.case_id,
        run_id=args.run_id,
        eval_id=args.eval_id,
        request_overrides=_request_overrides(args),
        evaluator_model=args.evaluator_model,
    )
    _print_result(
        {
            "response": response.model_dump(mode="json"),
            "evaluation_output_root": result.output_root,
            "trajectory": result.trajectory.model_dump(mode="json"),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
