"""Contracts for markdown-first durable company research state.

The canonical analyst artifact is ``research_state.md``.  These models cover
only the lifecycle sidecars and proposal envelopes that need machine-readable
identity, status, or audit semantics.
"""

from __future__ import annotations

from datetime import datetime, timezone
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.vault.contracts import EvidenceRef


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


SECTION_KEY_RE = re.compile(r"^[a-z][a-z0-9_]*$")
CITE_KEY_RE = re.compile(r"^S[1-9][0-9]*$")


ProposalType = Literal[
    "propose_section_update",
    "propose_question",
    "propose_close_question",
    "propose_conflict",
    "propose_valuation_snapshot",
    "propose_no_op",
]
SectionAction = Literal["replace", "append", "create", "remove"]
QuestionPriority = Literal["low", "normal", "high"]
QuestionStatus = Literal["open", "proposed_close", "answered", "rejected", "stale", "deferred"]
ConflictSeverity = Literal["low", "medium", "high"]
ConflictStatus = Literal["open", "resolved", "stale", "rejected"]
DecisionStatus = Literal["accepted", "rejected", "revised"]


class SectionMetadata(BaseModel):
    """Machine-readable index record for one markdown section."""

    model_config = ConfigDict(extra="forbid")

    section_key: str
    heading: str
    byte_range: tuple[int, int] | None = None
    last_updated_at: str | None = None
    last_updated_run_id: str | None = None
    synthesizer_model: str | None = None
    synthesizer_version: str | None = None
    cite_keys: list[str] = Field(default_factory=list)
    freshness: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("section_key")
    @classmethod
    def _valid_section_key(cls, value: str) -> str:
        normalized = value.strip()
        if not SECTION_KEY_RE.match(normalized):
            raise ValueError("section_key must match [a-z][a-z0-9_]*")
        return normalized

    @field_validator("heading")
    @classmethod
    def _non_empty_heading(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("heading must be non-empty")
        return normalized

    @field_validator("cite_keys")
    @classmethod
    def _valid_cite_keys(cls, value: list[str]) -> list[str]:
        for key in value:
            if not CITE_KEY_RE.match(key):
                raise ValueError(f"invalid cite key: {key}")
        return value

    @model_validator(mode="after")
    def _valid_byte_range(self) -> "SectionMetadata":
        if self.byte_range is not None:
            start, end = self.byte_range
            if start < 0 or end < start:
                raise ValueError("byte_range must be non-negative and ordered")
        return self


class StateIndex(BaseModel):
    """Section index over canonical ``research_state.md``."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    ticker: str
    last_run_id: str | None = None
    last_updated_at: str = Field(default_factory=utc_now_iso)
    sections: list[SectionMetadata] = Field(default_factory=list)

    @field_validator("ticker")
    @classmethod
    def _uppercase_ticker(cls, value: str) -> str:
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("ticker must be non-empty")
        return normalized

    @model_validator(mode="after")
    def _unique_sections(self) -> "StateIndex":
        keys = [section.section_key for section in self.sections]
        if len(keys) != len(set(keys)):
            raise ValueError("section_key values must be unique")
        return self


class EvidenceIndex(BaseModel):
    """Append-only mapping from stable cite keys (S1, S2, ...) to evidence."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    ticker: str
    entries: dict[str, EvidenceRef] = Field(default_factory=dict)

    @field_validator("ticker")
    @classmethod
    def _uppercase_ticker(cls, value: str) -> str:
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("ticker must be non-empty")
        return normalized

    @field_validator("entries")
    @classmethod
    def _valid_entries(cls, value: dict[str, EvidenceRef]) -> dict[str, EvidenceRef]:
        evidence_ids: set[str] = set()
        for key, evidence in value.items():
            if not CITE_KEY_RE.match(key):
                raise ValueError(f"invalid cite key: {key}")
            if evidence.evidence_id in evidence_ids:
                raise ValueError(f"duplicate evidence_id: {evidence.evidence_id}")
            evidence_ids.add(evidence.evidence_id)
        return value


class OpenQuestion(BaseModel):
    """Durable question with lifecycle state."""

    model_config = ConfigDict(extra="forbid")

    question_id: str
    text: str
    status: QuestionStatus = "open"
    priority: QuestionPriority = "normal"
    related_section_key: str | None = None
    evidence_keys: list[str] = Field(default_factory=list)
    proposed_answer: str | None = None
    rationale: str | None = None
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str | None = None

    @field_validator("question_id", "text")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("field must be non-empty")
        return normalized

    @field_validator("related_section_key")
    @classmethod
    def _valid_related_section(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not SECTION_KEY_RE.match(normalized):
            raise ValueError("related_section_key must match [a-z][a-z0-9_]*")
        return normalized

    @field_validator("evidence_keys")
    @classmethod
    def _valid_evidence_keys(cls, value: list[str]) -> list[str]:
        for key in value:
            if not CITE_KEY_RE.match(key):
                raise ValueError(f"invalid cite key: {key}")
        return value


class OpenQuestionsFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    ticker: str
    questions: list[OpenQuestion] = Field(default_factory=list)

    @field_validator("ticker")
    @classmethod
    def _uppercase_ticker(cls, value: str) -> str:
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("ticker must be non-empty")
        return normalized

    @model_validator(mode="after")
    def _unique_questions(self) -> "OpenQuestionsFile":
        ids = [question.question_id for question in self.questions]
        if len(ids) != len(set(ids)):
            raise ValueError("question_id values must be unique")
        return self


class ConflictEntry(BaseModel):
    """Durable conflict between evidence-backed statements."""

    model_config = ConfigDict(extra="forbid")

    conflict_id: str
    summary: str
    left_evidence_key: str
    right_evidence_key: str
    severity: ConflictSeverity = "medium"
    status: ConflictStatus = "open"
    rationale: str | None = None
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str | None = None

    @field_validator("conflict_id", "summary", "left_evidence_key", "right_evidence_key")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("field must be non-empty")
        return normalized

    @field_validator("left_evidence_key", "right_evidence_key")
    @classmethod
    def _valid_evidence_key(cls, value: str) -> str:
        if not CITE_KEY_RE.match(value):
            raise ValueError("evidence key must match S<number>")
        return value


class ConflictsFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    ticker: str
    conflicts: list[ConflictEntry] = Field(default_factory=list)

    @field_validator("ticker")
    @classmethod
    def _uppercase_ticker(cls, value: str) -> str:
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("ticker must be non-empty")
        return normalized

    @model_validator(mode="after")
    def _unique_conflicts(self) -> "ConflictsFile":
        ids = [conflict.conflict_id for conflict in self.conflicts]
        if len(ids) != len(set(ids)):
            raise ValueError("conflict_id values must be unique")
        return self


class ValuationSnapshot(BaseModel):
    """Point-in-time valuation sidecar."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    ticker: str
    as_of: str | None = None
    fields: dict[str, float | int | str | None] = Field(default_factory=dict)
    source_keys: list[str] = Field(default_factory=list)
    assumptions_markdown: str | None = None
    warnings: list[str] = Field(default_factory=list)
    updated_at: str = Field(default_factory=utc_now_iso)

    @field_validator("ticker")
    @classmethod
    def _uppercase_ticker(cls, value: str) -> str:
        normalized = value.strip().upper()
        if not normalized:
            raise ValueError("ticker must be non-empty")
        return normalized

    @field_validator("source_keys")
    @classmethod
    def _valid_source_keys(cls, value: list[str]) -> list[str]:
        for key in value:
            if not CITE_KEY_RE.match(key):
                raise ValueError(f"invalid source key: {key}")
        return value


class ProposalRecord(BaseModel):
    """One StateDelta proposal envelope with markdown content where needed."""

    model_config = ConfigDict(extra="forbid")

    proposal_id: str
    type: ProposalType
    section_key: str | None = None
    action: SectionAction | None = None
    body_markdown: str | None = None
    evidence_keys: list[str] = Field(default_factory=list)
    rationale: str = ""
    question_id: str | None = None
    text: str | None = None
    priority: QuestionPriority = "normal"
    related_section_key: str | None = None
    proposed_answer: str | None = None
    summary: str | None = None
    left_evidence_key: str | None = None
    right_evidence_key: str | None = None
    severity: ConflictSeverity = "medium"
    as_of: str | None = None
    fields: dict[str, float | int | str | None] = Field(default_factory=dict)
    source_keys: list[str] = Field(default_factory=list)
    assumptions_markdown: str | None = None
    created_at: str = Field(default_factory=utc_now_iso)

    @field_validator("proposal_id")
    @classmethod
    def _proposal_id_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("proposal_id must be non-empty")
        return normalized

    @field_validator("section_key", "related_section_key")
    @classmethod
    def _valid_optional_section_key(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not SECTION_KEY_RE.match(normalized):
            raise ValueError("section_key must match [a-z][a-z0-9_]*")
        return normalized

    @field_validator("evidence_keys", "source_keys")
    @classmethod
    def _valid_keys(cls, value: list[str]) -> list[str]:
        for key in value:
            if not CITE_KEY_RE.match(key):
                raise ValueError(f"invalid cite key: {key}")
        return value

    @model_validator(mode="after")
    def _validate_by_type(self) -> "ProposalRecord":
        if self.type == "propose_section_update":
            if not self.section_key or not self.action:
                raise ValueError("section update proposals require section_key and action")
            if self.action != "remove" and not (self.body_markdown or "").strip():
                raise ValueError("section update proposals require body_markdown unless action=remove")
            if self.action != "remove" and not self.evidence_keys:
                raise ValueError("section update proposals require evidence_keys unless action=remove")
        elif self.type == "propose_question":
            if not (self.text or "").strip():
                raise ValueError("question proposals require text")
        elif self.type == "propose_close_question":
            if not self.question_id or not (self.proposed_answer or "").strip() or not self.evidence_keys:
                raise ValueError("close-question proposals require question_id, proposed_answer, and evidence_keys")
        elif self.type == "propose_conflict":
            if not (self.summary or "").strip() or not self.left_evidence_key or not self.right_evidence_key:
                raise ValueError("conflict proposals require summary and both evidence keys")
            for key in (self.left_evidence_key, self.right_evidence_key):
                if not CITE_KEY_RE.match(key):
                    raise ValueError(f"invalid cite key: {key}")
        elif self.type == "propose_valuation_snapshot":
            if not self.as_of or not self.source_keys:
                raise ValueError("valuation proposals require as_of and source_keys")
        elif self.type == "propose_no_op":
            if not self.rationale.strip():
                raise ValueError("no-op proposals require rationale")
        return self


class StateUpdateDecision(BaseModel):
    """StateOwner decision for one proposal."""

    model_config = ConfigDict(extra="forbid")

    proposal_id: str
    decision: DecisionStatus
    reason: str
    applied_at: str = Field(default_factory=utc_now_iso)
    warnings: list[str] = Field(default_factory=list)
    revised_proposal: ProposalRecord | None = None

    @field_validator("proposal_id", "reason")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("field must be non-empty")
        return normalized
