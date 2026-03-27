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
    """One skill invocation requested by the planner, worker, or evaluator."""

    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ResearchBrief(BaseModel):
    """High-level decomposition of the user request before step planning."""

    model_config = ConfigDict(extra="forbid")

    primary_question: str
    sub_questions: list[str] = Field(default_factory=list)
    domain_packs: list[str] = Field(default_factory=list)
    user_constraints: list[str] = Field(default_factory=list)
    likely_report_shape: list[str] = Field(default_factory=list)
    key_unknowns: list[str] = Field(default_factory=list)
    likely_risks_of_failure: list[str] = Field(default_factory=list)
    rationale: str = ""


class ResearchStep(BaseModel):
    """One planned research step with explicit dependencies and outputs."""

    model_config = ConfigDict(extra="forbid")

    id: str
    objective: str
    depends_on: list[str] = Field(default_factory=list)
    recommended_skill_calls: list[SkillCall] = Field(default_factory=list)
    required_outputs: list[str] = Field(default_factory=list)
    completion_criteria: list[str] = Field(default_factory=list)
    counterevidence: bool = False
    can_run_parallel: bool = False
    status: Literal["pending", "running", "completed", "blocked", "failed"] = "pending"


class ContractClause(BaseModel):
    """One acceptance clause inside the research contract."""

    model_config = ConfigDict(extra="forbid")

    id: str
    category: str
    text: str
    severity: Literal["required", "important", "optional"] = "required"
    applies_to_steps: list[str] = Field(default_factory=list)
    applies_to_sections: list[str] = Field(default_factory=list)


class ResearchContract(BaseModel):
    """Definition of done for the current run."""

    model_config = ConfigDict(extra="forbid")

    global_clauses: list[ContractClause] = Field(default_factory=list)
    section_clauses: list[ContractClause] = Field(default_factory=list)
    step_clauses: list[ContractClause] = Field(default_factory=list)
    freshness_clauses: list[ContractClause] = Field(default_factory=list)
    numeric_clauses: list[ContractClause] = Field(default_factory=list)
    counterevidence_clauses: list[ContractClause] = Field(default_factory=list)


class ResearchPlan(BaseModel):
    """Executable research plan produced by the planner."""

    model_config = ConfigDict(extra="forbid")

    primary_question: str
    sub_questions: list[str] = Field(default_factory=list)
    domain_packs: list[str] = Field(default_factory=list)
    required_sections: list[str] = Field(default_factory=list)
    required_evidence: list[str] = Field(default_factory=list)
    freshness_requirements: list[str] = Field(default_factory=list)
    required_numeric_checks: list[str] = Field(default_factory=list)
    counterevidence_topics: list[str] = Field(default_factory=list)
    steps: list[ResearchStep] = Field(default_factory=list)
    rationale: str = ""


class StepExecutionResult(BaseModel):
    """Structured result from a step worker subagent."""

    model_config = ConfigDict(extra="forbid")

    step_id: str
    status: Literal["completed", "partial", "blocked", "failed"]
    summary: str
    evidence_ids: list[str] = Field(default_factory=list)
    artifact_paths: list[str] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    suggested_next_steps: list[str] = Field(default_factory=list)


class ClaimRecord(BaseModel):
    """Normalized claim record for draft and evaluator analysis."""

    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    section_label: str = ""
    claim_type: Literal["fact", "inference"] = "fact"
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    complicating_evidence_ids: list[str] = Field(default_factory=list)
    freshness_date: str | None = None
    appears_in_report: bool = True


class ReportSectionFeedback(BaseModel):
    """Targeted evaluator feedback for one report section."""

    model_config = ConfigDict(extra="forbid")

    section_label: str
    quoted_text: str = ""
    issue: str
    why_it_fails: str
    suggested_fix: str
    missing_evidence_ids: list[str] = Field(default_factory=list)


class ControllerDecision(BaseModel):
    """Structured controller output for a single harness step."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["call_skill", "run_step", "run_steps", "draft", "finalize"]
    rationale: str = ""
    skill_call: SkillCall | None = None
    step_id: str | None = None
    step_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_payload(self) -> "ControllerDecision":
        if self.action == "call_skill" and self.skill_call is None:
            raise ValueError("skill_call is required when action='call_skill'")
        if self.action != "call_skill" and self.skill_call is not None:
            raise ValueError("skill_call is only allowed when action='call_skill'")
        if self.action == "run_step" and not self.step_id:
            raise ValueError("step_id is required when action='run_step'")
        if self.action != "run_step" and self.step_id is not None:
            raise ValueError("step_id is only allowed when action='run_step'")
        if self.action == "run_steps" and not self.step_ids:
            raise ValueError("step_ids is required when action='run_steps'")
        if self.action != "run_steps" and self.step_ids:
            raise ValueError("step_ids is only allowed when action='run_steps'")
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
    """Structured evaluator output for a generated draft."""

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
    blocking_issues: list[str] = Field(default_factory=list)
    missing_sections: list[str] = Field(default_factory=list)
    missing_evidence_types: list[str] = Field(default_factory=list)
    missing_citations: list[str] = Field(default_factory=list)
    freshness_warnings: list[str] = Field(default_factory=list)
    numeric_inconsistencies: list[str] = Field(default_factory=list)
    required_follow_up_calls: list[SkillCall] = Field(default_factory=list)
    counterevidence_gaps: list[str] = Field(default_factory=list)
    report_section_feedback: list[ReportSectionFeedback] = Field(default_factory=list)


class HarnessRequest(BaseModel):
    """Input contract for one harness run."""

    model_config = ConfigDict(extra="forbid")

    user_prompt: str
    runtime: str = "harness"
    max_steps: int = 12
    max_revision_rounds: int = 2
    max_chars_before_condense: int = 6000
    selected_packs: list[str] | None = None
    max_step_worker_iterations: int = 3
    enable_parallel_steps: bool = True
    resume_from_checkpoint: str | None = None
    benchmark_label: str = ""


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
    research_brief: ResearchBrief | None = None
    research_plan: ResearchPlan | None = None
    research_contract: ResearchContract | None = None
    step_results: list[StepExecutionResult] = Field(default_factory=list)
    claim_map: list[ClaimRecord] = Field(default_factory=list)
    progress_updates: list[str] = Field(default_factory=list)
    completed_step_ids: list[str] = Field(default_factory=list)
    failed_step_ids: list[str] = Field(default_factory=list)
    pending_follow_up_calls: list[SkillCall] = Field(default_factory=list)
    latest_draft: str | None = None
    final_response: str | None = None
    step_count: int = 0
    revision_count: int = 0
    status: Literal["running", "completed", "failed"] = "running"
    run_dir: str | None = None
    dossier_dir: str | None = None
    mission_path: str | None = None
    progress_path: str | None = None
    plan_path: str | None = None
    contract_path: str | None = None
    qa_report_path: str | None = None
    report_path: str | None = None
    trace_path: str | None = None
    checkpoint_paths: list[str] = Field(default_factory=list)
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
