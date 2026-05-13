from __future__ import annotations

from datetime import datetime, timezone

from src.vault.ingest import ingest_text
from src.vault.status import evaluate_status_patrol, seed_status_patrol_questions
from src.vault.store import VaultStore


def test_status_patrol_flags_missing_sec_and_unconfirmed_valuation(tmp_path):
    root = tmp_path / "research_vault"
    store = VaultStore(root)
    doc = ingest_text(
        "Derived market support",
        ticker="XOM",
        source_type="derived_financials",
        source_grade="B",
        title="XOM derived financials",
        published_at="2026-01-01",
        root=root,
        store=store,
    )
    store.add_metric("XOM", "EV/EBITDA", "11.9", period="TTM", unit="x", source_doc_id=doc["doc_id"], source_grade="B")
    store.add_conflict("XOM", "metric_mismatch", "EV/EBITDA differs across sources.")

    checks = evaluate_status_patrol("xom", root=root, store=store)
    by_type = {check["check_type"]: check for check in checks}

    assert by_type["missing_a_grade_valuation_support"]["status"] == "open"
    assert "EV/EBITDA" in by_type["missing_a_grade_valuation_support"]["summary"]
    assert by_type["unresolved_conflicts"]["status"] == "open"
    assert by_type["no_recent_sec_source"]["status"] == "open"
    assert by_type["no_open_questions"]["status"] == "open"


def test_status_patrol_source_freshness_uses_injected_now(tmp_path):
    root = tmp_path / "research_vault"
    store = VaultStore(root)
    ingest_text(
        "Official filing",
        ticker="XOM",
        source_type="sec",
        source_grade="A",
        title="XOM 10-K",
        published_at="2026-01-01",
        root=root,
        store=store,
    )
    now = datetime(2026, 7, 1, tzinfo=timezone.utc)

    fresh = evaluate_status_patrol("XOM", root=root, store=store, stale_days=181, now=now)
    stale = evaluate_status_patrol("XOM", root=root, store=store, stale_days=180, now=now)
    fresh_by_type = {check["check_type"]: check for check in fresh}
    stale_by_type = {check["check_type"]: check for check in stale}

    assert fresh_by_type["stale_source_inventory"]["status"] == "ok"
    assert stale_by_type["stale_source_inventory"]["status"] == "open"
    assert "181 day(s) old" in stale_by_type["stale_source_inventory"]["summary"]


def test_status_patrol_seeds_deterministic_review_questions(tmp_path):
    root = tmp_path / "research_vault"
    store = VaultStore(root)
    ingest_text(
        "Derived market support",
        ticker="XOM",
        source_type="derived_financials",
        source_grade="B",
        title="XOM derived financials",
        published_at="2026-01-01",
        root=root,
        store=store,
    )

    initial = evaluate_status_patrol("XOM", root=root, store=store)
    first = seed_status_patrol_questions("XOM", root=root, store=store)
    second = seed_status_patrol_questions("XOM", root=root, store=store)
    context = store.company_context("XOM", limit=20)
    patrol_questions = [question for question in context["questions"] if question["question"].startswith("[Status patrol]")]

    assert len(patrol_questions) == sum(1 for check in initial if check["status"] != "ok")
    assert len({question["question_id"] for question in patrol_questions}) == len(patrol_questions)
    assert first == second
    assert sum(1 for check in first if check["status"] != "ok") < sum(1 for check in initial if check["status"] != "ok")
    assert any("Import recent 10-K/10-Q/8-K" in question["question"] for question in patrol_questions)


def test_status_patrol_does_not_duplicate_existing_specific_question(tmp_path):
    root = tmp_path / "research_vault"
    store = VaultStore(root)
    doc = ingest_text(
        "Derived market support",
        ticker="XOM",
        source_type="derived_financials",
        source_grade="B",
        title="XOM derived financials",
        published_at="2026-01-01",
        root=root,
        store=store,
    )
    store.add_metric("XOM", "EV/EBITDA", "11.9", period="TTM", unit="x", source_doc_id=doc["doc_id"], source_grade="B")
    store.add_question(
        "XOM",
        "Confirm XOM's valuation support metrics (EV/EBITDA) against A-grade filing-derived shares/debt/cash data.",
    )

    seed_status_patrol_questions("XOM", root=root, store=store)
    context = store.company_context("XOM", limit=20)
    patrol_questions = [question for question in context["questions"] if question["question"].startswith("[Status patrol]")]

    assert not any("valuation support metrics" in question["question"] for question in patrol_questions)
    assert any("Import recent 10-K/10-Q/8-K" in question["question"] for question in patrol_questions)
