from __future__ import annotations

import sqlite3

from src.vault.schema import init_vault
from src.vault.store import VaultStore


def test_init_vault_creates_schema(tmp_path):
    paths = init_vault(tmp_path / "research_vault")

    assert paths.database_path.exists()
    assert paths.documents_dir.exists()
    assert paths.companies_dir.exists()

    with sqlite3.connect(paths.database_path) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}

    assert {
        "documents",
        "companies",
        "document_companies",
        "facts",
        "metrics",
        "questions",
        "answers",
        "conflicts",
        "wiki_versions",
    }.issubset(tables)


def test_store_round_trip_company_document_and_context(tmp_path):
    store = VaultStore(tmp_path / "research_vault")

    company = store.upsert_company("xom", name="Exxon Mobil", exchange="NYSE", sector="Energy")
    document = store.insert_document(
        doc_id="doc_test",
        source_type="sec",
        title="XOM 10-K",
        path="documents/doc_test/extracted.md",
        url="https://www.sec.gov/example",
        published_at="2026-02-01",
        source_grade="A",
        checksum="abc",
        metadata={"form_type": "10-K"},
    )
    store.link_document_company("doc_test", "xom", relevance="primary")
    fact = store.add_fact("xom", "XOM is an integrated oil and gas company.", source_doc_id="doc_test", source_grade="A")
    metric = store.add_metric("xom", "Revenue", "100", period="FY2025", unit="USD", source_doc_id="doc_test", source_grade="A")
    question = store.add_question("xom", "What is mid-cycle FCF?", priority="high")
    conflict = store.add_conflict("xom", "metric_mismatch", "Revenue differs across sources.", conflict_id="conflict_revenue")
    updated_conflict = store.add_conflict(
        "xom",
        "metric_mismatch",
        "Revenue matches after source refresh.",
        severity="low",
        status="resolved",
        conflict_id="conflict_revenue",
    )

    context = store.company_context("XOM")

    assert company["ticker"] == "XOM"
    assert document["doc_id"] == "doc_test"
    assert fact["status"] == "active"
    assert metric["metric_name"] == "Revenue"
    assert question["priority"] == "high"
    assert conflict["status"] == "open"
    assert updated_conflict["summary"] == "Revenue matches after source refresh."
    assert updated_conflict["severity"] == "low"
    assert context["company"]["name"] == "Exxon Mobil"
    assert [doc["doc_id"] for doc in context["documents"]] == ["doc_test"]
    assert context["facts"][0]["statement"].startswith("XOM is")
    assert context["metrics"][0]["value"] == "100"
    assert context["questions"][0]["question"] == "What is mid-cycle FCF?"
    assert context["conflicts"] == []
