"""Typed contracts for the dynamic harness runtime."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class EvidenceItem(BaseModel):
    """A normalized evidence record used to ground final answers."""

    model_config = ConfigDict(extra="forbid")

    id: str = ""
    skill_name: str
    source_type: Literal["url", "artifact", "dataset", "note"]
    summary: str
    content: str | None = None
    sources: list[str] = Field(default_factory=list)
    artifact_paths: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_utc_now_iso)


class SkillCall(BaseModel):
    """One skill invocation requested by the controller."""

    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ControllerDecision(BaseModel):
    """Structured controller output for a single harness step."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["call_skill", "draft", "finalize"]
    rationale: str = ""
    skill_call: SkillCall | None = None

    @model_validator(mode="after")
    def _validate_skill_call(self) -> "ControllerDecision":
        if self.action == "call_skill" and self.skill_call is None:
            raise ValueError("skill_call is required when action='call_skill'")
        if self.action != "call_skill" and self.skill_call is not None:
            raise ValueError("skill_call is only allowed when action='call_skill'")
        return self


class SkillResult(BaseModel):
    """Normalized skill execution result captured in the harness trace."""

    model_config = ConfigDict(extra="forbid")

    skill_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    status: Literal["ok", "partial", "failed"]
    summary: str
    structured_data: dict[str, Any] = Field(default_factory=dict)
    output_text: str | None = None
    artifacts: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    error: str | None = None


class SkillSpec(BaseModel):
    """Registry entry for one public harness skill."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    name: str
    description: str
    pack: str
    input_schema: dict[str, Any] = Field(default_factory=dict)
    produces_artifacts: bool = False
    timeout_budget_seconds: int = 30
    executor: Callable[[dict[str, Any], "HarnessState"], SkillResult] | None = Field(
        default=None,
        exclude=True,
        repr=False,
    )


class VerificationReport(BaseModel):
    """Structured judge output for a generated draft."""

    model_config = ConfigDict(extra="forbid")

    decision: Literal["pass", "revise", "fail"]
    summary: str
    grounding: Literal["pass", "revise", "fail"]
    completeness: Literal["pass", "revise", "fail"]
    numeric_consistency: Literal["pass", "revise", "fail"]
    citation_coverage: Literal["pass", "revise", "fail"]
    formatting: Literal["pass", "revise", "fail"]
    improvement_instructions: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    raw_feedback: str = ""


class HarnessRequest(BaseModel):
    """Input contract for one harness run."""

    model_config = ConfigDict(extra="forbid")

    user_prompt: str
    runtime: str = "harness"
    max_steps: int = 12
    max_revision_rounds: int = 2
    max_chars_before_condense: int = 6000
    selected_packs: list[str] | None = None


class HarnessState(BaseModel):
    """Mutable working state carried by the harness runtime."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    request: HarnessRequest
    enabled_packs: list[str] = Field(default_factory=list)
    available_skills: list[SkillSpec] = Field(default_factory=list)
    evidence_ledger: list[EvidenceItem] = Field(default_factory=list)
    skill_history: list[SkillResult] = Field(default_factory=list)
    controller_log: list[ControllerDecision] = Field(default_factory=list)
    verification_reports: list[VerificationReport] = Field(default_factory=list)
    working_memory: list[str] = Field(default_factory=list)
    revision_notes: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    latest_draft: str | None = None
    final_response: str | None = None
    step_count: int = 0
    revision_count: int = 0
    status: Literal["running", "completed", "failed"] = "running"
    report_path: str | None = None
    trace_path: str | None = None
    error: str | None = None


class HarnessResponse(BaseModel):
    """Final public response produced by the harness."""

    model_config = ConfigDict(extra="forbid")

    final_response: str
    status: Literal["completed", "failed"]
    report_path: str | None = None
    trace_path: str | None = None
    verification: VerificationReport | None = None
    enabled_packs: list[str] = Field(default_factory=list)
    skills_used: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
