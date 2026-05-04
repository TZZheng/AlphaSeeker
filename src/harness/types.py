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


_TOOL_VIEW_DESCRIPTION_LIMIT = 600


def _truncate_tool_view_description(text: str) -> str:
    if len(text) <= _TOOL_VIEW_DESCRIPTION_LIMIT:
        return text
    return text[: max(0, _TOOL_VIEW_DESCRIPTION_LIMIT - 18)] + f"... [{len(text)} chars]"


def _line_count(text: str) -> int:
    return len(text.splitlines())


class ToolView(BaseModel):
    """Two views of one tool result: full audit log and compact model-visible state."""

    model_config = ConfigDict(extra="forbid")

    log: dict[str, Any]
    conversation: dict[str, Any]

    @classmethod
    def from_result(
        cls,
        tool_name: str,
        result: dict[str, Any],
        *,
        arguments: dict[str, Any] | None = None,
        canonical_path: str | None = None,
        canonical_fields: dict[str, Any] | None = None,
    ) -> "ToolView":
        status = str(result.get("status") or "ok")
        description = ""
        for key in ("description", "summary", "message", "error"):
            value = result.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                description = _truncate_tool_view_description(text)
                break
        if not description:
            if status in {"error", "failed"}:
                description = f"{tool_name} failed."
            else:
                description = f"{tool_name} completed with status {status}."
        conversation = {
            "tool_name": tool_name,
            "status": status,
            "description": description,
        }
        if tool_name == "write" and status == "ok":
            path = canonical_path or result.get("path")
            if path:
                conversation["path"] = str(path)
            conversation["operation"] = "overwrite"
            if arguments is not None and "content" in arguments:
                conversation["line_count"] = _line_count(str(arguments.get("content") or ""))
        if canonical_fields:
            conversation.update(
                {
                    key: value
                    for key, value in canonical_fields.items()
                    if value is not None
                }
            )
        return cls(
            log=dict(result),
            conversation=conversation,
        )


AGENT_PRESETS = ("orchestrator", "research", "writer", "synthesizer", "evaluator")
AGENT_STATUSES = (
    "queued",
    "running",
    "waiting",
    "done",
    "failed",
    "blocked",
    "stale",
    "cancelled",
    "refining",
    "timed_out",
    "timed_out_with_deliverable",
)
AGENT_TRANSPORTS = ("auto", "minimax_anthropic", "minimax_openai", "anthropic", "openai", "text_json")


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
    preset: Literal["orchestrator", "research", "writer", "synthesizer", "evaluator"] = "research"
    workspace_path: str
    task_name: str
    description: str
    status: Literal[
        "queued",
        "running",
        "waiting",
        "done",
        "failed",
        "blocked",
        "stale",
        "cancelled",
        "refining",
        "timed_out",
        "timed_out_with_deliverable",
    ] = "queued"
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
    root_preset: Literal["orchestrator", "research", "writer", "synthesizer", "evaluator"] = "orchestrator"
    agent_transport: Literal["auto", "minimax_anthropic", "minimax_openai", "anthropic", "openai", "text_json"] = "auto"
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
    commenter_interval_seconds: float | None = None

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
    started_at_epoch: float = Field(default_factory=time.time)
    elapsed_seconds: float = 0.0
    enabled_packs: list[str] = Field(default_factory=list)
    available_skills: list[SkillSpec] = Field(default_factory=list)
    required_sections: list[str] = Field(default_factory=list)
    research_contract: dict[str, Any] | None = None
    evidence_ledger: list[EvidenceItem] = Field(default_factory=list)
    skill_history: list[SkillResult] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
    critic_reports: list[dict[str, Any]] = Field(default_factory=list)
    latest_draft: str | None = None
    final_response: str | None = None
    last_error: str | None = None


class HarnessResponse(BaseModel):
    """Final public response produced by the file-based harness kernel."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["completed", "failed", "time_out", "time_out_with_deliverable"]
    stop_reason: str | None = None
    run_root: str | None = None
    root_agent_path: str | None = None
    final_report_path: str | None = None
    error: str | None = None
