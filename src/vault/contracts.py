"""Typed contracts for vault ingestion and research lifecycle boundaries.

These v0 models are serialization contracts only. Runtime ingestion and
lifecycle behavior remain in their existing modules until later vertical slices.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


AnchorStrategy = Literal["heading", "block_id", "none"]
ClaimStatus = Literal["active", "candidate", "rejected", "superseded"]
ConflictSeverity = Literal["low", "medium", "high", "blocker_review"]
ConflictStatus = Literal["open", "resolved", "ignored"]
ExtractionMethod = Literal["deterministic", "llm", "manual"]
FieldKind = Literal["narrative", "quantitative", "mixed", "unknown"]
IngestionStatus = Literal["stored", "skipped_duplicate", "failed"]
QuestionStatus = Literal["open", "proposed_close", "answered", "rejected"]
SourceGrade = Literal["A", "B", "C", "unknown"]
SourceType = Literal[
    "manual_file",
    "sec_filing",
    "company_profile",
    "financial_snapshot",
    "market_snapshot",
    "news",
    "web_search_result",
    "company_ir",
    "other",
]


class DocumentRef(BaseModel):
    """Stable reference to a source document registered in the vault."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    source_id: str | None = None
    source_type: SourceType | str = "other"
    title: str
    display_title: str
    vault_relative_path: str
    extracted_text_path: str | None = None
    source_grade: SourceGrade = "unknown"
    source_grade_rationale: str = ""
    checksum: str | None = None
    url: str | None = None
    published_at: str | None = None
    ingested_at: str = Field(default_factory=_utc_now_iso)
    anchor_strategy: AnchorStrategy = "heading"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("document_id", "title", "display_title", "vault_relative_path")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("field must be non-empty")
        return normalized

    @field_validator("vault_relative_path")
    @classmethod
    def _relative_path(cls, value: str) -> str:
        normalized = value.replace("\\", "/")
        if normalized.startswith("/"):
            raise ValueError("vault_relative_path must be relative")
        if ".." in normalized.split("/"):
            raise ValueError("vault_relative_path must not contain '..'")
        return normalized


class EvidenceRef(BaseModel):
    """Citation-ready pointer to evidence inside a vault document."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    document_id: str
    vault_relative_path: str
    display_title: str
    heading_path: list[str] = Field(default_factory=list)
    quoted_snippet: str
    start_offset: int | None = None
    end_offset: int | None = None
    source_grade: SourceGrade = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("evidence_id", "document_id", "vault_relative_path", "display_title", "quoted_snippet")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("field must be non-empty")
        return normalized

    @model_validator(mode="after")
    def _validate_offsets(self) -> "EvidenceRef":
        if self.start_offset is not None and self.start_offset < 0:
            raise ValueError("start_offset must be non-negative")
        if self.end_offset is not None and self.end_offset < 0:
            raise ValueError("end_offset must be non-negative")
        if self.start_offset is not None and self.end_offset is not None and self.end_offset < self.start_offset:
            raise ValueError("end_offset must be greater than or equal to start_offset")
        return self


class IngestionResult(BaseModel):
    """Vault ingestion output for one retrieval batch."""

    model_config = ConfigDict(extra="forbid")

    ingestion_id: str
    batch_id: str
    ticker: str
    documents: list[DocumentRef] = Field(default_factory=list)
    status: IngestionStatus = "stored"
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=_utc_now_iso)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClaimRecord(BaseModel):
    """Source-grounded research claim maintained by the lifecycle layer."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    ticker: str
    statement: str
    field_kind: FieldKind = "unknown"
    claim_status: ClaimStatus = "candidate"
    extraction_method: ExtractionMethod
    source_grade: SourceGrade = "unknown"
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    extractor_model: str | None = None
    extractor_version: str | None = None
    extraction_run_id: str | None = None
    confidence: float | None = None
    created_at: str = Field(default_factory=_utc_now_iso)
    updated_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _evidence_required(self) -> "ClaimRecord":
        if not self.evidence_refs:
            raise ValueError("ClaimRecord requires at least one evidence_ref")
        if self.extraction_method == "llm" and not self.extractor_model:
            raise ValueError("LLM-extracted claims require extractor_model")
        if self.extraction_method == "llm" and self.claim_status == "active" and self.field_kind == "quantitative":
            raise ValueError("LLM quantitative claims must not default to active")
        return self


class QuestionRecord(BaseModel):
    """Open or proposed-close research question for a company."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    ticker: str
    question: str
    question_status: QuestionStatus = "open"
    priority: Literal["low", "normal", "high"] = "normal"
    proposed_answer: str | None = None
    proposed_answer_ref: EvidenceRef | None = None
    created_at: str = Field(default_factory=_utc_now_iso)
    updated_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConflictRecord(BaseModel):
    """Recorded contradiction between source-grounded claims or documents."""

    model_config = ConfigDict(extra="forbid")

    conflict_id: str
    ticker: str
    summary: str
    left_ref: str
    right_ref: str
    severity: ConflictSeverity = "medium"
    severity_rule_id: str
    status: ConflictStatus = "open"
    created_at: str = Field(default_factory=_utc_now_iso)
    resolved_at: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchStateDelta(BaseModel):
    """Lifecycle-layer changes produced from newly ingested evidence."""

    model_config = ConfigDict(extra="forbid")

    delta_id: str
    ticker: str
    ingestion_id: str | None = None
    claims: list[ClaimRecord] = Field(default_factory=list)
    questions: list[QuestionRecord] = Field(default_factory=list)
    conflicts: list[ConflictRecord] = Field(default_factory=list)
    created_at: str = Field(default_factory=_utc_now_iso)
    metadata: dict[str, Any] = Field(default_factory=dict)
