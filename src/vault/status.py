"""Deterministic status patrol checks for company research vault pages."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any

from src.vault.paths import default_vault_paths
from src.vault.store import VaultStore

VALUATION_METRIC_NAMES = {"Current Price", "Market Cap", "Enterprise Value", "Trailing P/E", "Forward P/E", "EV/EBITDA"}
STALE_SOURCE_DAYS = 180


def _stable_id(prefix: str, *parts: object) -> str:
    raw = "\n".join("" if part is None else str(part) for part in parts)
    return f"{prefix}_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:12]}"


def _parse_datetime(value: object) -> datetime | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
    except ValueError:
        try:
            parsed = datetime.fromisoformat(text[:10])
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _latest_source_datetime(documents: list[dict[str, Any]]) -> datetime | None:
    dates = [_parse_datetime(doc.get("published_at") or doc.get("ingested_at")) for doc in documents]
    present = [date for date in dates if date is not None]
    return max(present) if present else None


def _metrics_named(metrics: list[dict[str, Any]], names: set[str]) -> list[dict[str, Any]]:
    return [metric for metric in metrics if metric.get("metric_name") in names]


def _has_unconfirmed_non_a_grade(metrics: list[dict[str, Any]]) -> bool:
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


def _check(
    ticker: str,
    check_type: str,
    *,
    severity: str,
    status: str,
    summary: str,
    action: str,
    ref: str | None = None,
) -> dict[str, Any]:
    return {
        "check_id": _stable_id("patrol", ticker.upper(), check_type),
        "ticker": ticker.upper(),
        "check_type": check_type,
        "severity": severity,
        "status": status,
        "summary": summary,
        "action": action,
        "ref": ref,
    }


def evaluate_status_patrol(
    ticker: str,
    *,
    root: str | Path | None = None,
    store: VaultStore | None = None,
    context: dict[str, Any] | None = None,
    stale_days: int = STALE_SOURCE_DAYS,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Return deterministic platform-QC checks for a company research page."""

    ticker_norm = ticker.strip().upper()
    active_store = store or VaultStore(root)
    active_context = context or active_store.company_context(ticker_norm, limit=100)
    documents = list(active_context.get("documents") or [])
    metrics = list(active_context.get("metrics") or [])
    questions = list(active_context.get("questions") or [])
    conflicts = list(active_context.get("conflicts") or [])

    checks: list[dict[str, Any]] = []

    valuation_metrics = _metrics_named(metrics, VALUATION_METRIC_NAMES)
    if valuation_metrics and _has_unconfirmed_non_a_grade(valuation_metrics):
        names = sorted({str(metric.get("metric_name")) for metric in valuation_metrics if metric.get("metric_name")})
        checks.append(
            _check(
                ticker_norm,
                "missing_a_grade_valuation_support",
                severity="medium",
                status="open",
                summary=(
                    f"Valuation support metrics lack same-period A-grade confirmation: {', '.join(names)}."
                ),
                action="Confirm valuation support metrics against filing-derived shares/debt/cash data or company-primary releases before relying on them.",
                ref="metrics",
            )
        )
    else:
        checks.append(
            _check(
                ticker_norm,
                "missing_a_grade_valuation_support",
                severity="low",
                status="ok",
                summary="No unconfirmed B-grade valuation support metrics detected.",
                action="No action.",
                ref="metrics",
            )
        )

    if conflicts:
        checks.append(
            _check(
                ticker_norm,
                "unresolved_conflicts",
                severity="high",
                status="open",
                summary=f"{len(conflicts)} open conflict(s) need human judgment.",
                action="Open conflicts.md, compare both refs side by side, and resolve or convert the gap into follow-up research questions.",
                ref="conflicts",
            )
        )
    else:
        checks.append(
            _check(
                ticker_norm,
                "unresolved_conflicts",
                severity="low",
                status="ok",
                summary="No open conflicts detected.",
                action="No action.",
                ref="conflicts",
            )
        )

    sec_documents = [doc for doc in documents if str(doc.get("source_type") or "").lower().startswith("sec")]
    if not sec_documents:
        checks.append(
            _check(
                ticker_norm,
                "no_recent_sec_source",
                severity="high",
                status="open",
                summary="No linked SEC filing source is registered for this company.",
                action="Import recent 10-K/10-Q/8-K filings before treating the wiki as A-grade grounded.",
                ref="source_index",
            )
        )
    else:
        checks.append(
            _check(
                ticker_norm,
                "no_recent_sec_source",
                severity="low",
                status="ok",
                summary=f"SEC source coverage present ({len(sec_documents)} linked filing source(s)).",
                action="No action.",
                ref="source_index",
            )
        )

    latest_source = _latest_source_datetime(documents)
    if latest_source is None:
        checks.append(
            _check(
                ticker_norm,
                "stale_source_inventory",
                severity="medium",
                status="open",
                summary="No dated source is available to evaluate source freshness.",
                action="Add dated filings, reports, notes, or market-data sources so freshness can be patrolled.",
                ref="source_index",
            )
        )
    else:
        current_time = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        age_days = (current_time - latest_source).days
        if age_days > stale_days:
            checks.append(
                _check(
                    ticker_norm,
                    "stale_source_inventory",
                    severity="medium",
                    status="open",
                    summary=f"Latest linked source is {age_days} day(s) old, above the {stale_days}-day patrol threshold.",
                    action="Refresh source inventory with recent filings/news/reports or explicitly mark coverage as stale.",
                    ref="source_index",
                )
            )
        else:
            checks.append(
                _check(
                    ticker_norm,
                    "stale_source_inventory",
                    severity="low",
                    status="ok",
                    summary=f"Latest linked source is {age_days} day(s) old, within the {stale_days}-day threshold.",
                    action="No action.",
                    ref="source_index",
                )
            )

    if not questions:
        checks.append(
            _check(
                ticker_norm,
                "no_open_questions",
                severity="medium",
                status="open",
                summary="No open research questions are queued after onboarding.",
                action="Review whether extraction failed to surface gaps; create section-specific questions before using the wiki for meeting prep.",
                ref="question_list",
            )
        )
    else:
        checks.append(
            _check(
                ticker_norm,
                "no_open_questions",
                severity="low",
                status="ok",
                summary=f"Open question queue present ({len(questions)} question(s)).",
                action="No action.",
                ref="question_list",
            )
        )

    return checks


def _existing_questions_cover_check(check: dict[str, Any], questions: list[dict[str, Any]]) -> bool:
    """Return true when a non-patrol question already covers the patrol gap."""

    check_type = str(check.get("check_type") or "")
    coverage_keywords = {
        "missing_a_grade_valuation_support": ("valuation support metrics",),
        "unresolved_conflicts": ("conflict", "conflicts"),
        "no_recent_sec_source": ("sec", "10-k", "10-q", "8-k"),
        "stale_source_inventory": ("stale", "source inventory", "refresh source"),
        # no_open_questions intentionally has no keyword coverage: seeding any patrol
        # question changes the queue from empty to non-empty and self-resolves this check.
    }
    keywords = coverage_keywords.get(check_type)
    if not keywords:
        return False
    for question in questions:
        text = str(question.get("question") or "").lower()
        if text.startswith("[status patrol]"):
            continue
        if any(keyword in text for keyword in keywords):
            return True
    return False


def seed_status_patrol_questions(
    ticker: str,
    *,
    root: str | Path | None = None,
    store: VaultStore | None = None,
    checks: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Convert open patrol checks into deterministic review-queue questions when no question already covers them."""

    ticker_norm = ticker.strip().upper()
    active_store = store or VaultStore(root)
    active_context = active_store.company_context(ticker_norm, limit=100)
    active_checks = checks or evaluate_status_patrol(ticker_norm, root=root, store=active_store, context=active_context)
    existing_questions = list(active_context.get("questions") or [])
    inserted_any = False
    for check in active_checks:
        if check.get("status") == "ok" or _existing_questions_cover_check(check, existing_questions):
            continue
        question = f"[Status patrol] {check['action']} ({check['summary']})"
        priority = "high" if check.get("severity") == "high" else "normal"
        inserted = active_store.add_question(
            ticker_norm,
            question,
            priority=priority,
            question_id=_stable_id("question", ticker_norm, "status_patrol", check.get("check_type")),
        )
        existing_questions.append(inserted)
        inserted_any = True
    if inserted_any:
        return evaluate_status_patrol(ticker_norm, root=root, store=active_store)
    return active_checks


def _md_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
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


def render_status_patrol(
    ticker: str,
    *,
    root: str | Path | None = None,
    checks: list[dict[str, Any]] | None = None,
) -> Path:
    """Render the deterministic status patrol support page."""

    ticker_norm = ticker.strip().upper()
    paths = default_vault_paths(root).ensure()
    company_dir = paths.company_dir(ticker_norm)
    company_dir.mkdir(parents=True, exist_ok=True)
    active_checks = checks or evaluate_status_patrol(ticker_norm, root=root)
    text = "\n".join(
        [
            f"# {ticker_norm} Status Patrol",
            "",
            "Deterministic workflow-QC checks. Open rows are review-queue items; OK rows document what was checked.",
            "",
            _md_table(
                active_checks,
                [
                    ("Check", "check_type"),
                    ("Severity", "severity"),
                    ("Status", "status"),
                    ("Summary", "summary"),
                    ("Action", "action"),
                    ("Ref", "ref"),
                ],
            ),
        ]
    )
    path = company_dir / "status_patrol.md"
    path.write_text(text, encoding="utf-8")
    return path
