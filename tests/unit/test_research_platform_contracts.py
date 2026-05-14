from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.harness.context_types import Citation, CleanupPolicy, MemoContextPackage
from src.retrieval.types import ResearchTask, RetrievalBatch, RetrievalRequest, SourceRecord
from src.vault.contracts import (
    ClaimRecord,
    ConflictRecord,
    DocumentRef,
    EvidenceRef,
    QuestionRecord,
)


def _evidence() -> EvidenceRef:
    return EvidenceRef(
        evidence_id="ev_xom_risk",
        document_id="doc_xom_10k_2024",
        vault_relative_path="companies/XOM/sources/sec_filings/10-K-2024.md",
        display_title="XOM 10-K FY2024",
        heading_path=["Item 1A. Risk Factors"],
        quoted_snippet="Commodity prices affect operating results.",
        source_grade="A",
    )


def test_retrieval_contracts_round_trip_with_llm_grade_proposal() -> None:
    task = ResearchTask(
        task_id="task_xom_memo",
        ticker="xom",
        user_prompt="Write an investment memo for XOM.",
        vault_root="vault",
        manual_source_paths=["inputs/xom-note.md"],
        required_source_types=["manual_file", "sec_filing", "company_profile"],
    )
    request = RetrievalRequest(
        request_id="req_xom_memo",
        task=task,
        source_types=["manual_file", "sec_filing", "company_profile"],
        allow_llm_discovery=True,
        allow_llm_grading=True,
    )
    source = SourceRecord(
        source_id="src_news_1",
        source_type="news",
        title="XOM announces project update",
        ticker="xom",
        url="https://example.com/xom-project",
        source_grade="unknown",
        proposed_source_grade="B",
        proposed_source_grade_rationale="Recognized financial news source; LLM-proposed pending policy acceptance.",
        retrieval_method="llm_assisted",
        vault_relative_path="companies/XOM/sources/news/xom-project.md",
        display_title="XOM project update",
    )
    batch = RetrievalBatch(
        batch_id="batch_xom_memo",
        request_id=request.request_id,
        task_id=task.task_id,
        ticker="xom",
        sources=[source],
        missing_source_types=["financial_snapshot"],
    )

    payload = batch.model_dump()
    restored = RetrievalBatch.model_validate(payload)

    assert restored.ticker == "XOM"
    assert restored.sources[0].ticker == "XOM"
    assert restored.sources[0].proposed_source_grade == "B"
    assert restored.sources[0].source_grade == "unknown"
    assert restored.missing_source_types == ["financial_snapshot"]


def test_vault_relative_paths_reject_absolute_or_parent_paths() -> None:
    with pytest.raises(ValidationError):
        SourceRecord(
            source_id="src_abs",
            source_type="manual_file",
            title="Bad absolute path",
            vault_relative_path="/companies/XOM/source.md",
        )

    with pytest.raises(ValidationError):
        DocumentRef(
            document_id="doc_bad",
            title="Bad parent path",
            display_title="Bad parent path",
            vault_relative_path="companies/XOM/../secret.md",
        )


def test_llm_active_claim_allowed_for_narrative_with_evidence() -> None:
    claim = ClaimRecord(
        claim_id="claim_news_catalyst",
        ticker="XOM",
        statement="Management described the project as on schedule.",
        field_kind="narrative",
        claim_status="active",
        extraction_method="llm",
        extractor_model="claude-opus-4-7",
        extractor_version="2026-05-13",
        extraction_run_id="run_news_extract",
        source_grade="B",
        evidence_refs=[_evidence()],
    )

    assert claim.claim_status == "active"
    assert claim.extraction_method == "llm"
    assert claim.field_kind == "narrative"


def test_llm_quantitative_claim_cannot_default_active() -> None:
    with pytest.raises(ValidationError):
        ClaimRecord(
            claim_id="claim_revenue",
            ticker="XOM",
            statement="Revenue was $100 billion.",
            field_kind="quantitative",
            claim_status="active",
            extraction_method="llm",
            extractor_model="claude-opus-4-7",
            source_grade="B",
            evidence_refs=[_evidence()],
        )


def test_claim_requires_evidence_and_llm_model_metadata() -> None:
    with pytest.raises(ValidationError):
        ClaimRecord(
            claim_id="claim_no_evidence",
            ticker="XOM",
            statement="Uncited claim.",
            field_kind="narrative",
            claim_status="candidate",
            extraction_method="manual",
        )

    with pytest.raises(ValidationError):
        ClaimRecord(
            claim_id="claim_no_model",
            ticker="XOM",
            statement="LLM claim with missing model.",
            field_kind="narrative",
            claim_status="candidate",
            extraction_method="llm",
            evidence_refs=[_evidence()],
        )


def test_question_proposed_close_and_conflict_severity_contracts() -> None:
    question = QuestionRecord(
        question_id="q_capex",
        ticker="XOM",
        question="What is FY2025 capex guidance?",
        question_status="proposed_close",
        proposed_answer="The latest filing provides the range.",
        proposed_answer_ref=_evidence(),
    )
    conflict = ConflictRecord(
        conflict_id="conf_revenue",
        ticker="XOM",
        summary="A-grade filing and vendor snapshot disagree on revenue.",
        left_ref="doc_xom_10k_2024",
        right_ref="doc_vendor_snapshot",
        severity="blocker_review",
        severity_rule_id="current_a_grade_material_metric_mismatch",
    )

    assert question.question_status == "proposed_close"
    assert conflict.severity == "blocker_review"
    assert conflict.status == "open"


def test_memo_context_package_uses_citation_table_and_nonblocking_defaults() -> None:
    evidence = _evidence()
    citation = Citation.from_evidence("xom_10k_risk", evidence)
    package = MemoContextPackage(
        package_id="pkg_xom_memo",
        task_id="task_xom_memo",
        ticker="xom",
        user_prompt="Write an investment memo for XOM.",
        citations=[citation],
        blockers=[],
    )
    cleanup = CleanupPolicy()

    assert package.ticker == "XOM"
    assert package.citations[0].vault_relative_path == "companies/XOM/sources/sec_filings/10-K-2024.md"
    assert package.blockers == []
    assert cleanup.mode == "product_final_only"
    assert cleanup.keep_final is True
    assert cleanup.keep_manifest is True
    assert cleanup.keep_status is True
    assert cleanup.keep_scratch is False
