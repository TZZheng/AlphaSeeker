"""Typed contracts for the harness runtime and its persistent artifacts."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


DOMAIN_PACKS = ("equity", "macro", "commodity")
ALL_PACKS = ("core", *DOMAIN_PACKS)
STEP_TERMINAL_STATUSES = ("completed", "partial", "failed", "skipped")


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
    """One skill invocation requested by the planner, controller, or evaluator."""

    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ResearchBrief(BaseModel):
    """Planner pass A: summarize the research mission and likely report shape."""

    model_config = ConfigDict(extra="forbid")

    primary_question: str
    sub_questions: list[str] = Field(default_factory=list)
    domain_packs: list[str] = Field(default_factory=list)
    user_constraints: list[str] = Field(default_factory=list)
    likely_report_shape: list[str] = Field(default_factory=list)
    key_unknowns: list[str] = Field(default_factory=list)
    likely_risks_of_failure: list[str] = Field(default_factory=list)
    rationale: str = ""

    @field_validator("domain_packs")
    @classmethod
    def _validate_domain_packs(cls, values: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for pack in values:
            pack_name = pack.strip().lower()
            if pack_name == "core":
                continue
            if pack_name not in DOMAIN_PACKS:
                raise ValueError(f"Illegal domain pack: {pack}")
            if pack_name not in seen:
                normalized.append(pack_name)
                seen.add(pack_name)
        return normalized


class ResearchStep(BaseModel):
    """A validated research work unit with explicit dependencies and outputs."""

    model_config = ConfigDict(extra="forbid")

    id: str
    objective: str
    depends_on: list[str] = Field(default_factory=list)
    recommended_skill_calls: list[SkillCall] = Field(default_factory=list)
    required_outputs: list[str] = Field(default_factory=list)
    completion_criteria: list[str] = Field(default_factory=list)
    report_sections: list[str] = Field(default_factory=list)
    domain_pack: str | None = None
    parallel_safe: bool = False

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("ResearchStep.id must be non-empty.")
        return text

    @field_validator("domain_pack")
    @classmethod
    def _validate_domain_pack(cls, value: str | None) -> str | None:
        if value is None:
            return None
        pack_name = value.strip().lower()
        if pack_name not in DOMAIN_PACKS:
            raise ValueError(f"Illegal step domain pack: {value}")
        return pack_name


class ResearchPlan(BaseModel):
    """Planner pass B: dependency-aware research plan."""

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

    @field_validator("domain_packs")
    @classmethod
    def _validate_domain_packs(cls, values: list[str]) -> list[str]:
        return ResearchBrief._validate_domain_packs(values)

    @model_validator(mode="after")
    def _validate_graph(self) -> "ResearchPlan":
        step_ids = [step.id for step in self.steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("ResearchPlan contains duplicate step ids.")

        known = set(step_ids)
        for step in self.steps:
            for dependency in step.depends_on:
                if dependency not in known:
                    raise ValueError(
                        f"ResearchStep '{step.id}' depends on unknown step '{dependency}'."
                    )

        visiting: set[str] = set()
        visited: set[str] = set()
        graph = {step.id: step.depends_on for step in self.steps}

        def _visit(node: str) -> None:
            if node in visited:
                return
            if node in visiting:
                raise ValueError("ResearchPlan contains cyclic step dependencies.")
            visiting.add(node)
            for dep in graph[node]:
                _visit(dep)
            visiting.remove(node)
            visited.add(node)

        for step_id in step_ids:
            _visit(step_id)
        return self


class ContractClause(BaseModel):
    """One definition-of-done clause for the final report."""

    model_config = ConfigDict(extra="forbid")

    id: str
    category: str
    text: str
    severity: Literal["required", "important", "optional"]
    applies_to_steps: list[str] = Field(default_factory=list)
    applies_to_sections: list[str] = Field(default_factory=list)


class ResearchContract(BaseModel):
    """Planner pass C: explicit acceptance criteria for the report."""

    model_config = ConfigDict(extra="forbid")

    global_clauses: list[ContractClause] = Field(default_factory=list)
    section_clauses: list[ContractClause] = Field(default_factory=list)
    step_clauses: list[ContractClause] = Field(default_factory=list)
    freshness_clauses: list[ContractClause] = Field(default_factory=list)
    numeric_clauses: list[ContractClause] = Field(default_factory=list)
    counterevidence_clauses: list[ContractClause] = Field(default_factory=list)


class PhaseUpdate(BaseModel):
    """A durable progress event recorded at each major harness phase."""

    model_config = ConfigDict(extra="forbid")

    phase: Literal["planner", "controller", "worker", "writer", "evaluator", "finalize"]
    summary: str
    created_at: str = Field(default_factory=_utc_now_iso)


class StepExecutionResult(BaseModel):
    """Worker result for one structured research step."""

    model_config = ConfigDict(extra="forbid")

    step_id: str
    status: Literal["completed", "partial", "blocked", "failed", "skipped"]
    summary: str
    evidence_ids: list[str] = Field(default_factory=list)
    artifact_paths: list[str] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    suggested_next_steps: list[str] = Field(default_factory=list)
    executed_skill_calls: list[SkillCall] = Field(default_factory=list)
    iteration_count: int = 0


class ReportSectionFeedback(BaseModel):
    """Section-level evaluator feedback used for targeted rewrites."""

    model_config = ConfigDict(extra="forbid")

    section_label: str
    quoted_text: str = ""
    issue: str
    why_it_fails: str
    suggested_fix: str
    missing_evidence_ids: list[str] = Field(default_factory=list)


class ClaimRecord(BaseModel):
    """A normalized claim in the report, linked to evidence."""

    model_config = ConfigDict(extra="forbid")

    id: str
    claim_text: str
    claim_type: Literal["fact", "inference"]
    section_label: str
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    complicating_evidence_ids: list[str] = Field(default_factory=list)
    freshness_date: str | None = None
    appears_in_final_report: bool = True


class ControllerDecision(BaseModel):
    """Structured controller output for a single harness step."""

    model_config = ConfigDict(extra="forbid")

    action: Literal[
        "call_skill",
        "execute_step",
        "execute_parallel_steps",
        "draft",
        "finalize",
    ]
    rationale: str = ""
    skill_call: SkillCall | None = None
    step_id: str | None = None
    step_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_action_payload(self) -> "ControllerDecision":
        if self.action == "call_skill" and self.skill_call is None:
            raise ValueError("skill_call is required when action='call_skill'")
        if self.action != "call_skill" and self.skill_call is not None:
            raise ValueError("skill_call is only allowed when action='call_skill'")
        if self.action == "execute_step" and not self.step_id:
            raise ValueError("step_id is required when action='execute_step'")
        if self.action != "execute_step" and self.step_id is not None:
            raise ValueError("step_id is only allowed when action='execute_step'")
        if self.action == "execute_parallel_steps" and not self.step_ids:
            raise ValueError("step_ids are required when action='execute_parallel_steps'")
        if self.action != "execute_parallel_steps" and self.step_ids:
            raise ValueError(
                "step_ids are only allowed when action='execute_parallel_steps'"
            )
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
    blocking_issues: list[str] = Field(default_factory=list)
    missing_sections: list[str] = Field(default_factory=list)
    missing_evidence_types: list[str] = Field(default_factory=list)
    missing_citations: list[str] = Field(default_factory=list)
    freshness_warnings: list[str] = Field(default_factory=list)
    numeric_inconsistencies: list[str] = Field(default_factory=list)
    required_follow_up_calls: list[SkillCall] = Field(default_factory=list)
    counterevidence_gaps: list[str] = Field(default_factory=list)
    report_section_feedback: list[ReportSectionFeedback] = Field(default_factory=list)
    raw_feedback: str = ""


class HarnessRequest(BaseModel):
    """Input contract for one harness run."""

    model_config = ConfigDict(extra="forbid")

    user_prompt: str
    runtime: str = "harness"
    max_steps: int = 12
    max_revision_rounds: int = 2
    max_chars_before_condense: int = 6000
    max_worker_iterations: int = 3
    max_parallel_steps: int = 3
    allow_parallel_steps: bool = True
    reset_context_each_phase: bool = False
    selected_packs: list[str] | None = None
    run_id: str | None = None
    resume_from: str | None = None


class HarnessState(BaseModel):
    """Mutable working state carried by the harness runtime."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    request: HarnessRequest
    run_id: str = ""
    run_root: str | None = None
    dossier_paths: dict[str, str] = Field(default_factory=dict)
    enabled_packs: list[str] = Field(default_factory=list)
    available_skills: list[SkillSpec] = Field(default_factory=list)
    mission_text: str = ""
    progress_text: str = ""
    research_brief: ResearchBrief | None = None
    research_plan: ResearchPlan | None = None
    research_contract: ResearchContract | None = None
    evidence_ledger: list[EvidenceItem] = Field(default_factory=list)
    skill_history: list[SkillResult] = Field(default_factory=list)
    step_results: list[StepExecutionResult] = Field(default_factory=list)
    step_statuses: dict[str, str] = Field(default_factory=dict)
    phase_history: list[PhaseUpdate] = Field(default_factory=list)
    controller_log: list[ControllerDecision] = Field(default_factory=list)
    verification_reports: list[VerificationReport] = Field(default_factory=list)
    working_memory: list[str] = Field(default_factory=list)
    revision_notes: list[str] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)
    claim_map: list[ClaimRecord] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    checkpoint_paths: list[str] = Field(default_factory=list)
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
    run_root: str | None = None
    dossier_paths: dict[str, str] = Field(default_factory=dict)
