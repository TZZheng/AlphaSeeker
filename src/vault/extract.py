"""Deterministic first-pass extraction into vault facts, metrics, and questions."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from src.vault.store import VaultStore

_KEY_RATIO_RE = re.compile(r"^-\s+\*\*(?P<name>[^*]+)\*\*:\s*(?P<value>.+?)\s*$")
_MISSING_VALUES = {"", "N/A", "NA", "None", "nan", "NaN"}

_VALUATION_METRICS = {
    "Current Price": (None, "USD"),
    "Market Cap": (None, "USD"),
    "Enterprise Value": (None, "USD"),
    "Trailing P/E": ("TTM", "x"),
    "Forward P/E": ("Forward", "x"),
    "EV/EBITDA": ("TTM", "x"),
}

_FINANCIAL_METRICS = {
    "Total Revenue (TTM)": ("TTM", "USD"),
    "Operating Cash Flow": ("TTM", "USD"),
    "Free Cash Flow": ("TTM", "USD"),
    "Debt/Equity": (None, "ratio"),
    "ROE": ("TTM", "ratio"),
    "Revenue Growth (YoY)": ("YoY", "ratio"),
    "Gross Margin": ("TTM", "ratio"),
    "Operating Margin": ("TTM", "ratio"),
    "Profit Margin": ("TTM", "ratio"),
}

_METRIC_METADATA = {**_VALUATION_METRICS, **_FINANCIAL_METRICS}
_FORM_TYPE_RE = re.compile(r"^(10-K|10-Q|8-K|DEF 14A)\b", re.IGNORECASE)

_CAPITAL_RETURN_ROWS = {
    "Free Cash Flow": ("Annual Free Cash Flow", "USD"),
    "Operating Cash Flow": ("Annual Operating Cash Flow", "USD"),
    "Capital Expenditure": ("Capital Expenditures", "USD"),
    "Repurchase Of Capital Stock": ("Share Repurchases", "USD"),
    "Cash Dividends Paid": ("Cash Dividends Paid", "USD"),
}
_CAPITAL_RETURN_METRIC_NAMES = {metric_name for metric_name, _ in _CAPITAL_RETURN_ROWS.values()}
_VALUATION_METRIC_NAMES = set(_VALUATION_METRICS)

_SEC_SECTION_RULES = [
    (
        "Business overview",
        "principal business involves",
    ),
    (
        "Management commentary / official commentary",
        "Management's Discussion and Analysis of Financial Condition and Results of Operations",
    ),
    (
        "Key risks from official filings",
        "oil, gas, and petrochemical businesses are fundamentally commodity businesses",
    ),
]


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "\n".join("" if part is None else str(part) for part in parts)
    return f"{prefix}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:12]}"


def _read_document_text(document: dict[str, Any]) -> str:
    path = Path(str(document.get("path") or ""))
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _clean_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _clip(text: str, *, max_chars: int = 650) -> str:
    cleaned = _clean_space(text)
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1].rstrip() + "…"


def _paragraphs(text: str) -> list[str]:
    return [_clean_space(line) for line in text.splitlines() if _clean_space(line) and not line.lstrip().startswith("|")]


def _first_paragraph_containing(text: str, needle: str) -> str | None:
    needle_lower = needle.lower()
    for paragraph in _paragraphs(text):
        if needle_lower in paragraph.lower():
            return paragraph
    return None


def _table_cells(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return []
    return [cell.strip() for cell in stripped.strip("|").split("|")]


def _is_separator_row(cells: list[str]) -> bool:
    return bool(cells) and all(set(cell.replace(":", "").strip()) <= {"-"} for cell in cells)


def _period_from_header(header: str) -> str:
    match = re.search(r"(20\d{2})", header)
    return f"FY{match.group(1)}" if match else header.strip()


def _normalize_table_number(value: str) -> str | None:
    cleaned = value.strip().replace(",", "")
    if cleaned in _MISSING_VALUES or cleaned.lower() == "nan":
        return None
    try:
        number = Decimal(cleaned)
    except InvalidOperation:
        return cleaned if cleaned else None
    if number == number.to_integral_value():
        return str(int(number))
    return format(number.normalize(), "f")


def parse_capital_return_metrics(text: str) -> list[dict[str, str | None]]:
    """Parse latest annual cash-flow capital allocation rows from financials Markdown."""

    lines = text.splitlines()
    in_section = False
    header: list[str] | None = None
    metrics: list[dict[str, str | None]] = []
    for line in lines:
        stripped = line.strip()
        if stripped == "### Annual Cash Flow Statement":
            in_section = True
            header = None
            continue
        if in_section and stripped.startswith("### "):
            break
        if not in_section or not stripped.startswith("|"):
            continue
        cells = _table_cells(stripped)
        if not cells or _is_separator_row(cells):
            continue
        if header is None:
            header = cells
            continue
        row_name = cells[0].strip()
        if row_name not in _CAPITAL_RETURN_ROWS or len(cells) < 2 or not header or len(header) < 2:
            continue
        value = _normalize_table_number(cells[1])
        if value is None:
            continue
        metric_name, unit = _CAPITAL_RETURN_ROWS[row_name]
        metrics.append({"metric_name": metric_name, "value": value, "period": _period_from_header(header[1]), "unit": unit})
    return metrics


def parse_key_ratio_metrics(text: str) -> list[dict[str, str | None]]:
    """Parse the deterministic ``## Key Ratios`` bullets emitted by the financials tool."""

    metrics: list[dict[str, str | None]] = []
    in_key_ratios = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "## Key Ratios":
            in_key_ratios = True
            continue
        if in_key_ratios and stripped.startswith("## "):
            break
        if not in_key_ratios:
            continue
        match = _KEY_RATIO_RE.match(stripped)
        if not match:
            continue
        name = match.group("name").strip()
        value = match.group("value").strip()
        if value in _MISSING_VALUES or name not in _METRIC_METADATA:
            continue
        period, unit = _METRIC_METADATA[name]
        metrics.append({"metric_name": name, "value": value, "period": period, "unit": unit})
    return metrics


def extract_financial_metrics(
    ticker: str,
    document: dict[str, Any],
    *,
    store: VaultStore,
) -> list[dict[str, Any]]:
    """Extract B-grade derived market/financial support metrics from one document."""

    text = _read_document_text(document)
    extracted: list[dict[str, Any]] = []
    for metric in [*parse_key_ratio_metrics(text), *parse_capital_return_metrics(text)]:
        metric_id = _stable_id(
            "metric",
            ticker.upper(),
            document.get("doc_id"),
            metric["metric_name"],
            metric.get("period"),
        )
        extracted.append(
            store.add_metric(
                ticker,
                str(metric["metric_name"]),
                str(metric["value"]),
                period=metric.get("period"),
                unit=metric.get("unit"),
                source_doc_id=str(document.get("doc_id") or ""),
                source_grade=str(document.get("source_grade") or "B"),
                observed_at=str(document.get("published_at") or document.get("ingested_at") or "") or None,
                metric_id=metric_id,
            )
        )
    return extracted


def _clean_form_type(value: object) -> str:
    text = str(value or "SEC filing").strip()
    match = _FORM_TYPE_RE.match(text)
    return match.group(1).upper() if match else text


def _metadata(document: dict[str, Any]) -> dict[str, Any]:
    raw = document.get("metadata_json") or "{}"
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def extract_sec_filing_facts(
    ticker: str,
    document: dict[str, Any],
    *,
    store: VaultStore,
) -> list[dict[str, Any]]:
    """Record simple A-grade SEC source facts from filing metadata."""

    meta = _metadata(document)
    form_type = _clean_form_type(meta.get("form_type") or document.get("title") or "SEC filing")
    filing_date = str(meta.get("filing_date") or document.get("published_at") or "").strip()
    date_phrase = f" dated {filing_date}" if filing_date else ""
    statement = f"{ticker.upper()} has an A-grade SEC {form_type} filing{date_phrase} in the research vault."
    source_quote = str(document.get("title") or "")
    if document.get("url"):
        source_quote = f"{source_quote} — {document['url']}" if source_quote else str(document["url"])
    fact_id = _stable_id("fact", ticker.upper(), document.get("doc_id"), "sec_source_fact")
    return [
        store.add_fact(
            ticker,
            statement,
            section="SEC source registry",
            source_doc_id=str(document.get("doc_id") or ""),
            source_quote=source_quote or None,
            source_grade=str(document.get("source_grade") or "A"),
            confidence=0.95,
            observed_at=filing_date or None,
            fact_id=fact_id,
        )
    ]


def extract_latest_sec_section_facts(
    ticker: str,
    document: dict[str, Any],
    *,
    store: VaultStore,
) -> list[dict[str, Any]]:
    """Extract a few high-signal A-grade snippets from the latest SEC filing text."""

    text = _read_document_text(document)
    if not text:
        return []
    meta = _metadata(document)
    filing_date = str(meta.get("filing_date") or document.get("published_at") or "").strip() or None
    extracted: list[dict[str, Any]] = []
    for section, needle in _SEC_SECTION_RULES:
        paragraph = _first_paragraph_containing(text, needle)
        if not paragraph:
            continue
        statement = _clip(paragraph)
        extracted.append(
            store.add_fact(
                ticker,
                statement,
                section=section,
                source_doc_id=str(document.get("doc_id") or ""),
                source_quote=statement,
                source_grade=str(document.get("source_grade") or "A"),
                confidence=0.8,
                observed_at=filing_date,
                fact_id=_stable_id("fact", ticker.upper(), document.get("doc_id"), section, statement),
            )
        )
    return extracted


def _metric_names_with_non_a_grade(metrics: list[dict[str, Any]], target_names: set[str]) -> set[str]:
    return {
        str(metric.get("metric_name") or "")
        for metric in metrics
        if metric.get("metric_name") in target_names and str(metric.get("source_grade") or "").upper() != "A"
    }


def add_default_research_questions(
    ticker: str,
    *,
    store: VaultStore,
    documents: list[dict[str, Any]],
    metrics: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Seed deterministic question list once source coverage and B-grade support metrics exist."""

    ticker_norm = ticker.upper()
    source_types = {str(doc.get("source_type") or "") for doc in documents}
    metrics = metrics or []
    questions: list[tuple[str, str]] = []
    if "sec" in source_types:
        questions.append(("high", f"What are the material risk factors, segment trends, and capital allocation signals in {ticker_norm}'s latest SEC filings?"))
    if "derived_financials" in source_types:
        questions.append(("normal", f"Which derived market-data metrics for {ticker_norm} need confirmation against A-grade filings or company releases?"))

    b_grade_capital_metrics = _metric_names_with_non_a_grade(metrics, _CAPITAL_RETURN_METRIC_NAMES)
    if b_grade_capital_metrics:
        names = ", ".join(sorted(b_grade_capital_metrics))
        questions.append(("high", f"Confirm {ticker_norm}'s latest annual capital-return metrics ({names}) against A-grade filing tables or company-primary disclosures."))

    b_grade_valuation_metrics = _metric_names_with_non_a_grade(metrics, _VALUATION_METRIC_NAMES)
    if b_grade_valuation_metrics:
        names = ", ".join(sorted(b_grade_valuation_metrics))
        questions.append(("normal", f"Confirm {ticker_norm}'s valuation support metrics ({names}) against A-grade filing-derived shares/debt/cash data or company-primary releases before relying on them."))

    inserted: list[dict[str, Any]] = []
    for priority, question in questions:
        inserted.append(
            store.add_question(
                ticker_norm,
                question,
                priority=priority,
                question_id=_stable_id("question", ticker_norm, question),
            )
        )
    return inserted


def extract_company_records(
    ticker: str,
    *,
    root: str | Path | None = None,
    store: VaultStore | None = None,
) -> dict[str, Any]:
    """Populate deterministic first-pass records for a company vault.

    This intentionally avoids LLM judgment. It extracts machine-readable key-ratio
    bullets from derived financial support docs, records SEC filing presence from
    metadata, and seeds a small question list so the company wiki has actionable
    structure after onboarding.
    """

    ticker_norm = ticker.strip().upper()
    if not ticker_norm:
        raise ValueError("ticker is required")
    active_store = store or VaultStore(root)
    documents = active_store.list_documents(ticker=ticker_norm, limit=500)

    metrics: list[dict[str, Any]] = []
    facts: list[dict[str, Any]] = []
    sec_documents: list[dict[str, Any]] = []
    for document in documents:
        source_type = str(document.get("source_type") or "")
        if source_type == "derived_financials":
            metrics.extend(extract_financial_metrics(ticker_norm, document, store=active_store))
        elif source_type == "sec":
            sec_documents.append(document)
            facts.extend(extract_sec_filing_facts(ticker_norm, document, store=active_store))

    if sec_documents:
        latest_sec = max(sec_documents, key=lambda doc: str(doc.get("published_at") or doc.get("ingested_at") or ""))
        facts.extend(extract_latest_sec_section_facts(ticker_norm, latest_sec, store=active_store))

    questions = add_default_research_questions(ticker_norm, store=active_store, documents=documents, metrics=metrics)
    return {
        "ticker": ticker_norm,
        "metrics": metrics,
        "facts": facts,
        "questions": questions,
        "counts": {"metrics": len(metrics), "facts": len(facts), "questions": len(questions)},
    }
