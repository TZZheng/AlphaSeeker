"""Typed contracts for the file-based harness kernel."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import time
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


DOMAIN_PACKS = ("equity", "macro", "commodity")
ALL_PACKS = ("core", *DOMAIN_PACKS)


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
    """One deterministic skill invocation requested by the controller."""

    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class SkillMetrics(BaseModel):
    """Normalized counters emitted by skills for controller reasoning."""

    model_config = ConfigDict(extra="forbid")

    evidence_count: int = 0
    fresh_evidence_count: int = 0
    artifact_count: int = 0
    urls_discovered: int = 0
    urls_read: int = 0
    dated_evidence_count: int = 0
    sections_touched: list[str] = Field(default_factory=list)
    filings_found: int = 0
    extra: dict[str, Any] = Field(default_factory=dict)


class RetrievalQueryBucket(BaseModel):
    """A themed bucket of retrieval queries."""

    model_config = ConfigDict(extra="forbid")

    label: str
    intent: str
    queries: list[str] = Field(default_factory=list)


class DiscoveredSource(BaseModel):
    """One candidate source discovered during retrieval."""

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
    evidence_ids: list[str] = Field(default_factory=list)
    section_labels: list[str] = Field(default_factory=list)
    numbers: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    stance: Literal["supporting", "counterevidence", "neutral"] = "neutral"


class SectionBrief(BaseModel):
    """A compressed section-level brief for the writer and controller."""

    model_config = ConfigDict(extra="forbid")

    section_label: str
    summary: str
    evidence_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    key_facts: list[str] = Field(default_factory=list)
    counterpoints: list[str] = Field(default_factory=list)
    coverage_status: Literal["strong", "partial", "missing"] = "missing"


class CoverageMatrixEntry(BaseModel):
    """One machine-readable coverage row for retrieval and critique."""

    model_config = ConfigDict(extra="forbid")

    coverage_type: Literal["section", "contract", "freshness", "counterevidence", "evidence_type"]
    label: str
    status: Literal["strong", "partial", "missing"]
    evidence_count: int = 0
    evidence_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class CoverageMatrix(BaseModel):
    """Coverage summary derived from retrieval artifacts."""

    model_config = ConfigDict(extra="forbid")

    sections: list[CoverageMatrixEntry] = Field(default_factory=list)
    contract_clauses: list[CoverageMatrixEntry] = Field(default_factory=list)
    freshness_requirements: list[CoverageMatrixEntry] = Field(default_factory=list)
    counterevidence_requirements: list[CoverageMatrixEntry] = Field(default_factory=list)
    evidence_types: list[CoverageMatrixEntry] = Field(default_factory=list)
    needs_more_retrieval: bool = False
    next_priority_labels: list[str] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)


class RetrievalStageOutput(BaseModel):
    """Typed stage output returned by the composite retrieval skill."""

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


class Observation(BaseModel):
    """Structured state fact emitted by skills, critics, or validators."""

    model_config = ConfigDict(extra="forbid")

    id: str
    source: str
    summary: str
    details: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    evidence_ids: list[str] = Field(default_factory=list)
    artifact_paths: list[str] = Field(default_factory=list)
    related_sections: list[str] = Field(default_factory=list)
    related_agenda_items: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=_utc_now_iso)


class SkillResult(BaseModel):
    """Normalized skill execution result captured in the harness trace."""

    model_config = ConfigDict(extra="forbid")

    skill_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    status: Literal["ok", "partial", "failed", "truncated"]
    summary: str
    details: dict[str, Any] = Field(default_factory=dict)
    metrics: SkillMetrics = Field(default_factory=SkillMetrics)
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


AGENT_PRESETS = ("orchestrator", "research", "source_triage", "writer", "synthesizer", "evaluator")
AGENT_STATUSES = ("queued", "running", "waiting", "done", "failed", "blocked", "stale", "cancelled", "refining")
AGENT_TRANSPORTS = ("auto", "minimax_anthropic", "minimax_openai", "text_json")


class AgentCommand(BaseModel):
    """One internal tool call proposed by an agent worker."""

    model_config = ConfigDict(extra="forbid")

    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    note: str = ""


class AgentRecord(BaseModel):
    """Append-only agent registry snapshot."""

    model_config = ConfigDict(extra="forbid")

    agent_id: str
    parent_id: str = ""
    preset: Literal["orchestrator", "research", "source_triage", "writer", "synthesizer", "evaluator"] = "research"
    workspace_path: str
    task_name: str
    description: str
    status: Literal["queued", "running", "waiting", "done", "failed", "blocked", "stale", "cancelled", "refining"] = "queued"
    created_at: str = Field(default_factory=_utc_now_iso)
    updated_at: str = Field(default_factory=_utc_now_iso)
    pid: int | None = None
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None


class AgentEvent(BaseModel):
    """One append-only supervisor or worker event."""

    model_config = ConfigDict(extra="forbid")

    event_type: str
    agent_id: str
    parent_id: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=_utc_now_iso)


class HarnessRequest(BaseModel):
    """Input contract for one file-based harness run."""

    model_config = ConfigDict(extra="forbid")

    user_prompt: str
    runtime: str = "harness"
    run_id: str | None = None
    root_preset: Literal["orchestrator", "research", "source_triage", "writer", "synthesizer", "evaluator"] = "orchestrator"
    agent_transport: Literal["auto", "minimax_anthropic", "minimax_openai", "text_json"] = "auto"
    wall_clock_budget_seconds: int = 1200
    root_wall_clock_seconds: int | None = None
    max_agents_per_run: int = 64
    max_live_agents: int = 16
    max_live_children_per_parent: int = 8
    per_agent_wall_clock_seconds: int = 1800
    stale_heartbeat_seconds: int = 45
    available_skill_packs: list[str] | None = None
    continuous_refinement: bool = False
    resume_from_run_root: str | None = None

    @field_validator("available_skill_packs")
    @classmethod
    def _validate_available_skill_packs(cls, values: list[str] | None) -> list[str] | None:
        if values is None:
            return values
        normalized: list[str] = []
        seen: set[str] = set()
        for pack in values:
            pack_name = pack.strip().lower()
            if pack_name not in ALL_PACKS:
                raise ValueError(f"Illegal skill pack: {pack}")
            if pack_name not in seen:
                normalized.append(pack_name)
                seen.add(pack_name)
        return normalized


class HarnessState(BaseModel):
    """Mutable per-agent skill state persisted to disk between worker turns."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    request: HarnessRequest
    run_id: str = ""
    run_root: str | None = None
    agent_id: str = ""
    workspace_path: str | None = None
    dossier_paths: dict[str, str] = Field(default_factory=dict)
    started_at_epoch: float = Field(default_factory=time.time)
    elapsed_seconds: float = 0.0
    enabled_packs: list[str] = Field(default_factory=list)
    available_skills: list[SkillSpec] = Field(default_factory=list)
    required_sections: list[str] = Field(default_factory=list)
    research_contract: dict[str, Any] | None = None
    query_buckets: list[RetrievalQueryBucket] = Field(default_factory=list)
    discovered_sources: list[DiscoveredSource] = Field(default_factory=list)
    read_queue: list[ReadQueueEntry] = Field(default_factory=list)
    read_results: list[ReadResultRecord] = Field(default_factory=list)
    source_cards: list[SourceCard] = Field(default_factory=list)
    fact_index: list[FactIndexRecord] = Field(default_factory=list)
    section_briefs: list[SectionBrief] = Field(default_factory=list)
    coverage_matrix: CoverageMatrix | None = None
    evidence_ledger: list[EvidenceItem] = Field(default_factory=list)
    skill_history: list[SkillResult] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
    critic_reports: list[dict[str, Any]] = Field(default_factory=list)
    latest_draft: str | None = None
    final_response: str | None = None
    retrieval_wave_count: int = 0
    last_error: str | None = None


class HarnessResponse(BaseModel):
    """Final public response produced by the file-based harness kernel."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["completed", "failed"]
    stop_reason: str | None = None
    run_root: str | None = None
    root_agent_path: str | None = None
    final_report_path: str | None = None
    error: str | None = None
