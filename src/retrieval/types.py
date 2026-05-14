"""Typed contracts for AlphaSeeker research input retrieval.

These v0 models define the boundary between task planning, source retrieval,
and deterministic vault ingestion. They intentionally do not fetch live data or
write vault state.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
RetrievalWorkflow = Literal["onboard", "memo", "monitor", "meeting_prep", "ad_hoc"]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ResearchTask(BaseModel):
    """User-facing research request normalized before retrieval begins."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    ticker: str
    user_prompt: str
    workflow: RetrievalWorkflow = "memo"
    company_name: str | None = None
    vault_root: str | None = None
    manual_source_paths: list[str] = Field(default_factory=list)
    required_source_types: list[SourceType] = Field(default_factory=list)
    optional_source_types: list[SourceType] = Field(default_factory=list)
    created_at: str = Field(default_factory=_utc_now_iso)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("ticker")
    @classmethod
    def _normalize_ticker(cls, value: str) -> str:
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("ticker must be non-empty")
        return normalized


class RetrievalRequest(BaseModel):
    """Instructions for the retrieval layer for one research task."""

    model_config = ConfigDict(extra="forbid")

    request_id: str
    task: ResearchTask
    source_types: list[SourceType] = Field(default_factory=list)
    allow_llm_discovery: bool = False
    allow_llm_grading: bool = False
    max_sources_per_type: int | None = None
    freshness_days: int | None = None
    created_at: str = Field(default_factory=_utc_now_iso)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SourceRecord(BaseModel):
    """One retrieved or user-supplied source before vault ingestion."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    source_type: SourceType
    title: str
    ticker: str | None = None
    local_path: str | None = None
    url: str | None = None
    published_at: str | None = None
    retrieved_at: str = Field(default_factory=_utc_now_iso)
    source_grade: SourceGrade = "unknown"
    source_grade_rationale: str = ""
    proposed_source_grade: SourceGrade | None = None
    proposed_source_grade_rationale: str = ""
    retrieval_method: Literal["manual", "deterministic", "llm_assisted"] = "deterministic"
    checksum: str | None = None
    vault_relative_path: str | None = None
    display_title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("ticker")
    @classmethod
    def _normalize_optional_ticker(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().upper()
        return normalized or None

    @field_validator("source_id", "title")
    @classmethod
    def _non_empty_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("field must be non-empty")
        return normalized

    @field_validator("vault_relative_path")
    @classmethod
    def _relative_vault_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().replace("\\", "/")
        if not normalized:
            return None
        if normalized.startswith("/"):
            raise ValueError("vault_relative_path must be relative")
        if ".." in normalized.split("/"):
            raise ValueError("vault_relative_path must not contain '..'")
        return normalized


class RetrievalBatch(BaseModel):
    """Retrieval-layer output handed to deterministic vault ingestion."""

    model_config = ConfigDict(extra="forbid")

    batch_id: str
    request_id: str
    task_id: str
    ticker: str
    sources: list[SourceRecord] = Field(default_factory=list)
    missing_source_types: list[SourceType] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=_utc_now_iso)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("ticker")
    @classmethod
    def _normalize_batch_ticker(cls, value: str) -> str:
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("ticker must be non-empty")
        return normalized
