"""SEC filing import adapter for the persistent research vault."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.tools.equity.sec_filings import search_and_read_filings
from src.vault.ingest import ingest_text
from src.vault.store import VaultStore


def import_sec_filings(
    *,
    ticker: str,
    company_name: str | None = None,
    form_types: list[str] | None = None,
    max_filings: int = 5,
    max_chars_per_filing: int = 15000,
    root: str | Path | None = None,
    store: VaultStore | None = None,
) -> list[dict[str, Any]]:
    """Fetch recent SEC filings and store them as A-grade vault documents."""

    ticker_norm = ticker.strip().upper()
    if not ticker_norm:
        raise ValueError("ticker is required")
    resolved_company = company_name or ticker_norm
    active_store = store or VaultStore(root)
    active_store.upsert_company(ticker_norm, name=company_name)
    filings = search_and_read_filings(
        company_name=resolved_company,
        ticker=ticker_norm,
        form_types=form_types or ["10-K", "10-Q", "8-K"],
        max_filings=max_filings,
        max_chars_per_filing=max_chars_per_filing,
    )
    imported: list[dict[str, Any]] = []
    for filing in filings:
        form_type = str(filing.get("form_type") or "SEC filing")
        filing_date = str(filing.get("filing_date") or "") or None
        url = str(filing.get("url") or "") or None
        text = str(filing.get("text") or "")
        title = f"{ticker_norm} {form_type} {filing_date or ''}".strip()
        imported.append(
            ingest_text(
                text,
                ticker=ticker_norm,
                source_type="sec",
                title=title,
                source_grade="A",
                url=url,
                published_at=filing_date,
                metadata={
                    "company": filing.get("company") or resolved_company,
                    "form_type": form_type,
                    "filing_date": filing_date,
                    "url": url,
                },
                root=root,
                store=active_store,
            )
        )
    return imported
