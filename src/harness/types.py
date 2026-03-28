"""Typed contracts for the harness runtime and its persistent artifacts."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


DOMAIN_PACKS = ("equity", "macro", "commodity")
ALL_PACKS = ("core", *DOMAIN_PACKS)
STEP_TERMINAL_STATUSES = ("completed", "partial", "failed", "skipped")
RESEARCH_PROFILES = ("standard", "deep")
EVALUATOR_PARSE_MODES = (
    "direct_structured",
    "normalized_structured",
    "rule_based_fallback",
)


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


class RetrievalQueryBucket(BaseModel):
    """A themed bucket of retrieval queries for deep research."""

    model_config = ConfigDict(extra="forbid")

    label: str
    intent: str
    queries: list[str] = Field(default_factory=list)


class DiscoveredSource(BaseModel):
    """One candidate source discovered during broad retrieval."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    query: str
    query_bucket: str
    search_type: Literal["web", "news", "artifact", "dataset"]
    title: str
    url: str
    canonical_url: str
    domain: str = ""
    snippet: str = ""
    publication_date: str | None = None
    discovered_rank: int = 0
    freshness_score: float = 0.0
    relevance_score: float = 0.0
    uniqueness_score: float = 0.0
    source_quality_score: float = 0.0
    composite_score: float = 0.0
    coverage_tags: list[str] = Field(default_factory=list)


class ReadQueueEntry(BaseModel):
    """A ranked source selected for full-text reading."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    canonical_url: str
    title: str
    domain: str = ""
    query_bucket: str = ""
    priority_rank: int
    priority_score: float
    coverage_tags: list[str] = Field(default_factory=list)
    reason: str = ""


class ReadResultRecord(BaseModel):
    """One attempted full-text ingestion result."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    canonical_url: str
    title: str
    status: Literal["read", "failed"]
    text: str = ""
    text_chars: int = 0
    publication_date: str | None = None
    error: str | None = None
    query_bucket: str = ""
    coverage_tags: list[str] = Field(default_factory=list)


class SourceCard(BaseModel):
    """A normalized representation of one ingested source."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    title: str
    canonical_url: str = ""
    domain: str = ""
    source_kind: Literal["web", "news", "artifact", "dataset", "note"] = "web"
    publication_date: str | None = None
    summary: str
    extracted_facts: list[str] = Field(default_factory=list)
    extracted_numbers: list[str] = Field(default_factory=list)
    extracted_dates: list[str] = Field(default_factory=list)
    supporting_evidence: list[str] = Field(default_factory=list)
    counterevidence: list[str] = Field(default_factory=list)
    section_relevance: list[str] = Field(default_factory=list)
    freshness_label: str = ""
    evidence_ids: list[str] = Field(default_factory=list)


class FactIndexRecord(BaseModel):
    """A normalized fact extracted from one or more source cards."""

    model_config = ConfigDict(extra="forbid")

    fact_id: str
    fact: str
    source_ids: list[str] = Field(default_factory=list)
    section_labels: list[str] = Field(default_factory=list)
    numbers: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    stance: Literal["supporting", "counterevidence", "neutral"] = "neutral"


class SectionBrief(BaseModel):
    """A compressed section-level brief used by the deep writer."""

    model_config = ConfigDict(extra="forbid")

    section_label: str
    summary: str
    evidence_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    key_facts: list[str] = Field(default_factory=list)
    counterpoints: list[str] = Field(default_factory=list)
    coverage_status: Literal["strong", "partial", "missing"] = "missing"


class CoverageMatrixEntry(BaseModel):
    """One machine-readable coverage row for retrieval or QA scheduling."""

    model_config = ConfigDict(extra="forbid")

    coverage_type: Literal["section", "contract", "freshness", "counterevidence", "evidence_type"]
    label: str
    status: Literal["strong", "partial", "missing"]
    evidence_count: int = 0
    evidence_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class CoverageMatrix(BaseModel):
    """Coverage summary used by the controller and evaluator in deep mode."""

    model_config = ConfigDict(extra="forbid")

    sections: list[CoverageMatrixEntry] = Field(default_factory=list)
    contract_clauses: list[CoverageMatrixEntry] = Field(default_factory=list)
    freshness_requirements: list[CoverageMatrixEntry] = Field(default_factory=list)
    counterevidence_requirements: list[CoverageMatrixEntry] = Field(default_factory=list)
    evidence_types: list[CoverageMatrixEntry] = Field(default_factory=list)
    needs_more_retrieval: bool = False
    next_priority_labels: list[str] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)


class DeepRetrievalStageOutput(BaseModel):
    """Typed stage output returned by the composite deep retrieval skill."""

    model_config = ConfigDict(extra="forbid")

    stage: Literal[
        "plan_queries",
        "discover",
        "rank",
        "build_read_queue",
        "ingest_batch",
        "extract_batch",
        "refresh_coverage",
        "run_wave",
    ]
    query_bucket_count: int = 0
    query_count: int = 0
    discovered_count: int = 0
    deduped_count: int = 0
    read_queue_count: int = 0
    successful_read_count: int = 0
    failed_read_count: int = 0
    source_card_count: int = 0
    extraction_batch_count: int = 0
    coverage_status: str = ""
    artifact_paths: list[str] = Field(default_factory=list)


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


class PhaseTimingEvent(BaseModel):
    """One measured latency sample for a major harness phase."""

    model_config = ConfigDict(extra="forbid")

    phase: Literal["planner", "retrieval", "writer", "evaluator"]
    duration_seconds: float
    detail: str = ""
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
    unresolved_gaps: list[str] = Field(default_factory=list)
    suggested_retrieval_queries: list[str] = Field(default_factory=list)
    wall_clock_exhausted: bool = False
    qa_iteration: int = 0
    evaluator_parse_mode: Literal[
        "direct_structured",
        "normalized_structured",
        "rule_based_fallback",
    ] = "rule_based_fallback"
    raw_feedback: str = ""


class HarnessRequest(BaseModel):
    """Input contract for one harness run."""

    model_config = ConfigDict(extra="forbid")

    user_prompt: str
    runtime: str = "harness"
    research_profile: Literal["standard", "deep"] = "standard"
    wall_clock_budget_seconds: int = 300
    max_steps: int = 18
    max_revision_rounds: int = 6
    max_chars_before_condense: int = 6000
    max_worker_iterations: int = 3
    max_parallel_steps: int = 3
    allow_parallel_steps: bool = True
    reset_context_each_phase: bool = False
    selected_packs: list[str] | None = None
    run_id: str | None = None
    resume_from: str | None = None
    deep_query_target: int = 60
    deep_candidate_target: int = 300
    deep_read_queue_target: int = 120
    deep_read_batch_size: int = 20
    deep_successful_read_target: int = 80

    @model_validator(mode="before")
    @classmethod
    def _apply_profile_defaults(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values
        profile = str(values.get("research_profile", "standard")).strip().lower()
        if profile not in RESEARCH_PROFILES:
            raise ValueError(f"Illegal research_profile: {profile}")
        if profile == "deep":
            values.setdefault("wall_clock_budget_seconds", 1200)
            values.setdefault("max_steps", 120)
            values.setdefault("max_revision_rounds", 20)
            values.setdefault("max_worker_iterations", 6)
            values.setdefault("deep_query_target", 60)
            values.setdefault("deep_candidate_target", 300)
            values.setdefault("deep_read_queue_target", 120)
            values.setdefault("deep_read_batch_size", 20)
            values.setdefault("deep_successful_read_target", 80)
        return values


class HarnessState(BaseModel):
    """Mutable working state carried by the harness runtime."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    request: HarnessRequest
    run_id: str = ""
    run_root: str | None = None
    dossier_paths: dict[str, str] = Field(default_factory=dict)
    started_at_epoch: float = Field(default_factory=time.time)
    elapsed_seconds: float = 0.0
    enabled_packs: list[str] = Field(default_factory=list)
    available_skills: list[SkillSpec] = Field(default_factory=list)
    mission_text: str = ""
    progress_text: str = ""
    research_brief: ResearchBrief | None = None
    research_plan: ResearchPlan | None = None
    research_contract: ResearchContract | None = None
    deep_query_buckets: list[RetrievalQueryBucket] = Field(default_factory=list)
    discovered_sources: list[DiscoveredSource] = Field(default_factory=list)
    read_queue: list[ReadQueueEntry] = Field(default_factory=list)
    read_results: list[ReadResultRecord] = Field(default_factory=list)
    source_cards: list[SourceCard] = Field(default_factory=list)
    fact_index: list[FactIndexRecord] = Field(default_factory=list)
    section_briefs: list[SectionBrief] = Field(default_factory=list)
    coverage_matrix: CoverageMatrix | None = None
    evidence_ledger: list[EvidenceItem] = Field(default_factory=list)
    skill_history: list[SkillResult] = Field(default_factory=list)
    step_results: list[StepExecutionResult] = Field(default_factory=list)
    step_statuses: dict[str, str] = Field(default_factory=dict)
    phase_history: list[PhaseUpdate] = Field(default_factory=list)
    phase_timing_events: list[PhaseTimingEvent] = Field(default_factory=list)
    phase_timing_totals_seconds: dict[str, float] = Field(default_factory=dict)
    latency_bottleneck: Literal["planner", "retrieval", "writer", "evaluator", "unknown"] = "unknown"
    controller_log: list[ControllerDecision] = Field(default_factory=list)
    verification_reports: list[VerificationReport] = Field(default_factory=list)
    pending_follow_up_calls: list[SkillCall] = Field(default_factory=list)
    executed_follow_up_calls: list[SkillCall] = Field(default_factory=list)
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
    qa_iteration_count: int = 0
    deep_retrieval_wave_count: int = 0
    timed_out: bool = False
    timeout_reason: str | None = None
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
