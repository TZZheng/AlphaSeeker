"""Thin wrapper that feeds vault research state into the existing memo harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.harness import HarnessRequest, HarnessResponse, run_harness
from src.vault.ingest import ingest_file
from src.vault.synthesis import synthesize_company_research_state

DEFAULT_SKILL_PACKS = ["core", "equity", "macro", "commodity", "vault"]


def build_vault_backed_prompt(user_prompt: str, *, ticker: str, research_state_path: str) -> str:
    """Add a short instruction without inlining the generated wiki content."""

    return "\n\n".join(
        [
            user_prompt.strip(),
            (
                f"A generated company research-state wiki for {ticker.upper()} is attached in this run's context as "
                f"`{Path(research_state_path).name}`. Read it first as a source-aware starting point with citations "
                "and open questions. Verify material claims as needed with normal AlphaSeeker tools, then produce the "
                "investment memo in `publish/final.md`."
            ),
        ]
    ).strip()


def run_vault_backed_memo(
    *,
    user_prompt: str,
    ticker: str,
    company_name: str | None = None,
    source_paths: list[str] | None = None,
    root: str | Path | None = None,
    synthesis_model: str | None = None,
    run_id: str | None = None,
    wall_clock_budget_seconds: int = 1200,
    harness_overrides: dict[str, Any] | None = None,
) -> HarnessResponse:
    """Generate vault research state, then run the existing AlphaSeeker memo harness."""

    ticker_norm = ticker.strip().upper()
    if not ticker_norm:
        raise ValueError("ticker is required")
    for path in source_paths or []:
        ingest_file(path, ticker=ticker_norm, source_type="manual_inbox", source_grade="A", root=root)
    synthesis = synthesize_company_research_state(
        ticker_norm,
        company_name=company_name,
        root=root,
        model_name=synthesis_model,
    )
    prompt = build_vault_backed_prompt(user_prompt, ticker=ticker_norm, research_state_path=synthesis["wiki_path"])
    request_payload: dict[str, Any] = {
        "user_prompt": prompt,
        "run_id": run_id,
        "wall_clock_budget_seconds": wall_clock_budget_seconds,
        "available_skill_packs": DEFAULT_SKILL_PACKS,
        "context_files": [synthesis["wiki_path"]],
    }
    if harness_overrides:
        request_payload.update(harness_overrides)
    request = HarnessRequest(**request_payload)
    return run_harness(request)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an AlphaSeeker investment memo backed by vault LLM research state.")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--company-name", default=None)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--source", action="append", default=[])
    parser.add_argument("--root", default=None)
    parser.add_argument("--synthesis-model", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--wall-clock-budget-seconds", type=int, default=1200)
    args = parser.parse_args()
    response = run_vault_backed_memo(
        user_prompt=args.prompt,
        ticker=args.ticker,
        company_name=args.company_name,
        source_paths=args.source,
        root=args.root,
        synthesis_model=args.synthesis_model,
        run_id=args.run_id,
        wall_clock_budget_seconds=args.wall_clock_budget_seconds,
    )
    print(json.dumps(response.model_dump(mode="json"), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
