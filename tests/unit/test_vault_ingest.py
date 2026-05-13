from __future__ import annotations

from pathlib import Path

from src.vault.ingest import ingest_file, ingest_text
from src.vault.store import VaultStore


def test_ingest_file_copies_extracts_and_links_document(tmp_path):
    source = tmp_path / "sample_10k_excerpt.md"
    source.write_text("# XOM 10-K\n\nOfficial filing excerpt.", encoding="utf-8")
    root = tmp_path / "research_vault"

    result = ingest_file(source, ticker="xom", source_type="sec", source_grade="A", root=root)

    extracted_path = Path(result["extracted_path"])
    assert result["doc_id"].startswith("doc_")
    assert extracted_path.exists()
    assert extracted_path.read_text(encoding="utf-8").startswith("# XOM 10-K")
    assert Path(result["original_path"]).exists()
    assert Path(result["metadata_path"]).exists()

    context = VaultStore(root).company_context("XOM")
    assert context["documents"][0]["doc_id"] == result["doc_id"]
    assert context["documents"][0]["source_grade"] == "A"


def test_ingest_text_registers_a_grade_sec_document(tmp_path):
    root = tmp_path / "research_vault"

    result = ingest_text(
        "Item 1. Business. Exxon Mobil explores for and produces crude oil.",
        ticker="xom",
        source_type="sec",
        title="XOM 10-K 2026",
        source_grade="A",
        url="https://www.sec.gov/xom",
        published_at="2026-02-01",
        root=root,
    )

    context = VaultStore(root).company_context("XOM")
    doc = context["documents"][0]
    assert doc["doc_id"] == result["doc_id"]
    assert doc["source_type"] == "sec"
    assert doc["source_grade"] == "A"
    assert doc["url"] == "https://www.sec.gov/xom"
    assert Path(result["extracted_path"]).read_text(encoding="utf-8").startswith("Item 1")
