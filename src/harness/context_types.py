"""Context-package and output contracts for research-platform memo runs."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.vault.contracts import ConflictSeverity, DocumentRef, EvidenceRef, QuestionRecord


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


CleanupMode = Literal["debug_keep_all", "product_final_only", "archive_on_failure"]
OutputStatus = Literal["succeeded", "failed", "partial"]


class Citation(BaseModel):
    """A render-neutral citation envelope for memo/deck outputs."""

    model_config = ConfigDict(extra="forbid")

    citation_key: str
    document_id: str
    vault_relative_path: str
    display_title: str
    heading_path: list[str] = Field(default_factory=list)
    snippet: str
    source_grade: str = "unknown"
    evidence_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_evidence(cls, citation_key: str, evidence: EvidenceRef) -> "Citation":
        return cls(
            citation_key=citation_key,
            document_id=evidence.document_id,
            vault_relative_path=evidence.vault_relative_path,
            display_title=evidence.display_title,
            heading_path=list(evidence.heading_path),
            snippet=evidence.quoted_snippet,
            source_grade=evidence.source_grade,
            evidence_id=evidence.evidence_id,
        )


class Caveat(BaseModel):
    """Non-blocking warning that should be visible to the memo writer/reader."""

    model_config = ConfigDict(extra="forbid")

    caveat_id: str
    message: str
    severity: ConflictSeverity | Literal["info"] = "info"
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Blocker(BaseModel):
    """Strict-mode blocker; empty by default because v1 memo generation is non-blocking."""

    model_config = ConfigDict(extra="forbid")

    blocker_id: str
    message: str
    required_action: str = "human_review"
    evidence_refs: list[EvidenceRef] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoContextPackage(BaseModel):
    """Prepared research state consumed by the memo/output harness."""

    model_config = ConfigDict(extra="forbid")

    package_id: str
    task_id: str
    ticker: str
    user_prompt: str
    documents: list[DocumentRef] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    open_questions: list[QuestionRecord] = Field(default_factory=list)
    caveats: list[Caveat] = Field(default_factory=list)
    blockers: list[Blocker] = Field(default_factory=list)
    source_index_path: str | None = None
    question_list_path: str | None = None
    review_queue_path: str | None = None
    conflict_list_path: str | None = None
    created_at: str = Field(default_factory=_utc_now_iso)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("ticker")
    @classmethod
    def _normalize_ticker(cls, value: str) -> str:
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("ticker must be non-empty")
        return normalized


class CleanupPolicy(BaseModel):
    """Retention policy for output-harness scratch and product artifacts."""

    model_config = ConfigDict(extra="forbid")

    mode: CleanupMode = "product_final_only"
    keep_final: bool = True
    keep_manifest: bool = True
    keep_status: bool = True
    keep_scratch: bool = False
    archive_on_failure: bool = True


class OutputRunResult(BaseModel):
    """Final output-harness result and reproducibility pointers."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    package_id: str
    ticker: str
    status: OutputStatus
    final_path: str | None = None
    manifest_path: str | None = None
    status_path: str | None = None
    cleanup_policy: CleanupPolicy = Field(default_factory=CleanupPolicy)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=_utc_now_iso)
    metadata: dict[str, Any] = Field(default_factory=dict)
