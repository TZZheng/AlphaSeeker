"""Obsidian-compatible Markdown rendering for company vault pages."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import shutil

from src.vault.paths import default_vault_paths
from src.vault.store import VaultStore, new_id


VALUATION_METRIC_NAMES = {"Current Price", "Market Cap", "Enterprise Value", "Trailing P/E", "Forward P/E", "EV/EBITDA"}
CAPITAL_RETURN_METRIC_NAMES = {"Annual Free Cash Flow", "Annual Operating Cash Flow", "Capital Expenditures", "Share Repurchases", "Cash Dividends Paid"}
SEC_REGISTRY_SECTION = "SEC source registry"
BUSINESS_SECTIONS = {"Business overview", SEC_REGISTRY_SECTION}
COMMENTARY_SECTIONS = {"Management commentary / official commentary"}
RISK_SECTIONS = {"Key risks from official filings"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _filter_facts(facts: list[dict[str, object]], sections: set[str]) -> list[dict[str, object]]:
    return [fact for fact in facts if str(fact.get("section") or "") in sections]


def _md_table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "_None yet._\n"
    header = "| " + " | ".join(label for label, _ in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        cells = []
        for _, key in columns:
            value = row.get(key, "")
            text = "" if value is None else str(value)
            cells.append(text.replace("\n", " ").replace("|", "\\|"))
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *body]) + "\n"


def _metrics_named(metrics: list[dict[str, object]], names: set[str]) -> list[dict[str, object]]:
    return [metric for metric in metrics if metric.get("metric_name") in names]


def _has_unconfirmed_non_a_grade(metrics: list[dict[str, object]]) -> bool:
    confirmed = {
        (str(metric.get("metric_name") or ""), metric.get("period"))
        for metric in metrics
        if str(metric.get("source_grade") or "").upper() == "A"
    }
    return any(
        str(metric.get("source_grade") or "").upper() != "A"
        and (str(metric.get("metric_name") or ""), metric.get("period")) not in confirmed
        for metric in metrics
    )


def _support_note(metrics: list[dict[str, object]]) -> str:
    if not _has_unconfirmed_non_a_grade(metrics):
        return ""
    return "Note: non-A-grade rows are support metrics pending confirmation against A-grade filing tables or company-primary disclosures.\n"


def render_source_index(ticker: str, *, root: str | Path | None = None) -> Path:
    store = VaultStore(root)
    paths = default_vault_paths(root).ensure()
    ticker_norm = ticker.upper()
    company_dir = paths.company_dir(ticker_norm)
    company_dir.mkdir(parents=True, exist_ok=True)
    documents = store.list_documents(ticker=ticker_norm, limit=200)
    text = "\n".join(
        [
            f"# {ticker_norm} Source Index",
            "",
            "Source policy: A-grade sources first. SEC/company-primary evidence should drive MVP records; derived market data is support evidence only.",
            "",
            _md_table(
                documents,
                [
                    ("Doc ID", "doc_id"),
                    ("Title", "title"),
                    ("Type", "source_type"),
                    ("Grade", "source_grade"),
                    ("Published", "published_at"),
                    ("Path", "path"),
                    ("URL", "url"),
                ],
            ),
        ]
    )
    path = company_dir / "source_index.md"
    path.write_text(text, encoding="utf-8")
    return path


def render_support_pages(
    ticker: str,
    *,
    root: str | Path | None = None,
    questions: list[dict[str, object]] | None = None,
    conflicts: list[dict[str, object]] | None = None,
) -> dict[str, Path]:
    paths = default_vault_paths(root).ensure()
    ticker_norm = ticker.upper()
    company_dir = paths.company_dir(ticker_norm)
    company_dir.mkdir(parents=True, exist_ok=True)
    question_path = company_dir / "question_list.md"
    conflict_path = company_dir / "conflicts.md"
    catalyst_path = company_dir / "catalysts.md"
    if questions is None:
        if not question_path.exists():
            question_path.write_text(f"# {ticker_norm} Question List\n\n_Open questions will appear here._\n", encoding="utf-8")
    else:
        question_path.write_text(
            "\n".join(
                [
                    f"# {ticker_norm} Question List",
                    "",
                    _md_table(questions, [("Question", "question"), ("Priority", "priority"), ("Created", "created_at")]),
                ]
            ),
            encoding="utf-8",
        )
    if conflicts is None:
        if not conflict_path.exists():
            conflict_path.write_text(f"# {ticker_norm} Conflicts\n\n_Open conflicts will appear here._\n", encoding="utf-8")
    else:
        conflict_path.write_text(
            "\n".join(
                [
                    f"# {ticker_norm} Conflicts",
                    "",
                    _md_table(conflicts, [("Type", "conflict_type"), ("Summary", "summary"), ("Severity", "severity"), ("Status", "status")]),
                ]
            ),
            encoding="utf-8",
        )
    if not catalyst_path.exists():
        catalyst_path.write_text(f"# {ticker_norm} Catalysts\n\n_Catalysts will appear here._\n", encoding="utf-8")
    return {"question_list": question_path, "conflicts": conflict_path, "catalysts": catalyst_path}


def render_company_wiki(ticker: str, *, root: str | Path | None = None, run_id: str | None = None) -> Path:
    """Render an Obsidian-compatible company database homepage."""

    store = VaultStore(root)
    paths = default_vault_paths(root).ensure()
    ticker_norm = ticker.upper()
    context = store.company_context(ticker_norm, limit=50)
    company = context["company"] or {"ticker": ticker_norm, "name": None}
    company_dir = paths.company_dir(ticker_norm)
    versions_dir = company_dir / "versions"
    company_dir.mkdir(parents=True, exist_ok=True)
    versions_dir.mkdir(parents=True, exist_ok=True)

    wiki_path = company_dir / "wiki.md"
    timestamp = _utc_now_iso()
    if wiki_path.exists():
        version_path = versions_dir / f"wiki_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.md"
        shutil.copy2(wiki_path, version_path)
        with store.connect() as conn:
            conn.execute(
                """
                INSERT INTO wiki_versions(version_id, ticker, path, run_id, created_at, summary)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (new_id("wiki"), ticker_norm, str(version_path), run_id, timestamp, "Archived previous wiki before render."),
            )

    name = company.get("name") or ticker_norm
    business_facts = _filter_facts(context["facts"], BUSINESS_SECTIONS)
    commentary_facts = _filter_facts(context["facts"], COMMENTARY_SECTIONS)
    risk_facts = _filter_facts(context["facts"], RISK_SECTIONS)
    capital_return_metrics = _metrics_named(context["metrics"], CAPITAL_RETURN_METRIC_NAMES)
    valuation_metrics = _metrics_named(context["metrics"], VALUATION_METRIC_NAMES)
    text = "\n".join(
        [
            f"# {ticker_norm} — {name}",
            "",
            f"Last updated: {timestamp}",
            "Source policy: A-grade sources first in MVP v1; derived market data is clearly labeled support evidence.",
            "",
            "Navigation: [[source_index]] · [[question_list]] · [[conflicts]] · [[catalysts]]",
            "",
            "## 1. One-line company summary",
            "_To be filled from active A-grade facts._",
            "",
            "## 2. Business overview",
            _md_table(business_facts, [("Fact", "statement"), ("Section", "section"), ("Source", "source_doc_id"), ("Grade", "source_grade")]),
            "## 3. Revenue / earnings / cash flow snapshot",
            _md_table(context["metrics"], [("Metric", "metric_name"), ("Period", "period"), ("Value", "value"), ("Unit", "unit"), ("Source", "source_doc_id")]),
            "## 4. Balance sheet and capital return",
            _support_note(capital_return_metrics),
            _md_table(
                capital_return_metrics,
                [("Metric", "metric_name"), ("Period", "period"), ("Value", "value"), ("Unit", "unit"), ("Source", "source_doc_id"), ("Grade", "source_grade")],
            ),
            "## 5. Management guidance / official commentary",
            _md_table(commentary_facts, [("Fact", "statement"), ("Source", "source_doc_id"), ("Grade", "source_grade")]),
            "## 6. Key risks from official filings",
            _md_table(risk_facts, [("Fact", "statement"), ("Source", "source_doc_id"), ("Grade", "source_grade")]),
            "## 7. Valuation-relevant metrics",
            _support_note(valuation_metrics),
            _md_table(
                valuation_metrics,
                [("Metric", "metric_name"), ("Period", "period"), ("Value", "value"), ("Unit", "unit"), ("Source", "source_doc_id"), ("Grade", "source_grade")],
            ),
            "## 8. Open questions",
            _md_table(context["questions"], [("Question", "question"), ("Priority", "priority"), ("Created", "created_at")]),
            "## 9. Conflicts / items needing human judgment",
            _md_table(context["conflicts"], [("Type", "conflict_type"), ("Summary", "summary"), ("Severity", "severity"), ("Status", "status")]),
            "## 10. Source index",
            "See [[source_index]]. Latest registered sources:",
            _md_table(context["documents"], [("Doc ID", "doc_id"), ("Title", "title"), ("Type", "source_type"), ("Grade", "source_grade"), ("Published", "published_at")]),
        ]
    )
    wiki_path.write_text(text, encoding="utf-8")
    render_source_index(ticker_norm, root=root)
    render_support_pages(ticker_norm, root=root, questions=context["questions"], conflicts=context["conflicts"])
    return wiki_path
