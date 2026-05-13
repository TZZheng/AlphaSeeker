"""Deterministic company onboarding pipeline for the research vault."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.tools.equity.company_profile import fetch_company_profile
from src.tools.equity.financials import fetch_financial_metrics
from src.vault.extract import extract_company_records
from src.vault.ingest import ingest_file
from src.vault.paths import default_vault_paths
from src.vault.sec_import import import_sec_filings
from src.vault.status import seed_status_patrol_questions
from src.vault.store import VaultStore
from src.vault.wiki import render_company_wiki


def onboard_company(
    *,
    ticker: str,
    company_name: str | None = None,
    forms: list[str] | None = None,
    max_filings: int = 5,
    include_market_support: bool = True,
    inbox_dir: str | Path | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Onboard a company into the local Obsidian-compatible research vault."""

    ticker_norm = ticker.strip().upper()
    if not ticker_norm:
        raise ValueError("ticker is required")
    store = VaultStore(root)
    store.upsert_company(ticker_norm, name=company_name)

    imported_sec = import_sec_filings(
        ticker=ticker_norm,
        company_name=company_name,
        form_types=forms or ["10-K", "10-Q", "8-K"],
        max_filings=max_filings,
        root=root,
        store=store,
    )

    imported_support: list[dict[str, Any]] = []
    if include_market_support:
        support_dir = default_vault_paths(root).ensure().root / "market_support" / ticker_norm
        support_dir.mkdir(parents=True, exist_ok=True)
        try:
            profile_path, profile_meta = fetch_company_profile(ticker_norm, output_dir=support_dir)
            imported_support.append(
                ingest_file(
                    profile_path,
                    ticker=ticker_norm,
                    source_type="derived_profile",
                    title=f"{ticker_norm} derived company profile",
                    source_grade="B",
                    metadata=profile_meta,
                    root=root,
                    store=store,
                )
            )
            if profile_meta.get("company_name") and not company_name:
                store.upsert_company(ticker_norm, name=str(profile_meta["company_name"]))
        except Exception as exc:
            imported_support.append({"status": "failed", "source_type": "derived_profile", "error": str(exc)})
        try:
            financials_path, financials_meta = fetch_financial_metrics(ticker_norm, output_dir=support_dir)
            imported_support.append(
                ingest_file(
                    financials_path,
                    ticker=ticker_norm,
                    source_type="derived_financials",
                    title=f"{ticker_norm} derived financial metrics",
                    source_grade="B",
                    metadata=financials_meta,
                    root=root,
                    store=store,
                )
            )
        except Exception as exc:
            imported_support.append({"status": "failed", "source_type": "derived_financials", "error": str(exc)})

    imported_manual: list[dict[str, Any]] = []
    if inbox_dir:
        inbox_path = Path(inbox_dir)
        if inbox_path.exists():
            for path in sorted(item for item in inbox_path.iterdir() if item.is_file()):
                imported_manual.append(
                    ingest_file(
                        path,
                        ticker=ticker_norm,
                        source_type="manual_inbox",
                        source_grade="A",
                        root=root,
                        store=store,
                    )
                )

    extracted_records = extract_company_records(ticker_norm, root=root, store=store)
    status_checks = seed_status_patrol_questions(ticker_norm, root=root, store=store)
    wiki_path = render_company_wiki(ticker_norm, root=root)
    vault_paths = default_vault_paths(root).ensure()
    return {
        "ticker": ticker_norm,
        "company": store.get_company(ticker_norm),
        "vault_root": str(vault_paths.root),
        "database_path": str(vault_paths.database_path),
        "wiki_path": str(wiki_path),
        "sec_documents": imported_sec,
        "support_documents": imported_support,
        "manual_documents": imported_manual,
        "extracted_records": extracted_records,
        "status_patrol": status_checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Onboard a company into the AlphaSeeker research vault.")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--company-name", default=None)
    parser.add_argument("--forms", nargs="*", default=["10-K", "10-Q", "8-K"])
    parser.add_argument("--max-filings", type=int, default=5)
    parser.add_argument("--no-market-support", action="store_true")
    parser.add_argument("--inbox-dir", default=None)
    parser.add_argument("--root", default=None)
    args = parser.parse_args()
    result = onboard_company(
        ticker=args.ticker,
        company_name=args.company_name,
        forms=args.forms,
        max_filings=args.max_filings,
        include_market_support=not args.no_market_support,
        inbox_dir=args.inbox_dir,
        root=args.root,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
