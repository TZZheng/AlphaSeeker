"""Split status derivation for research-platform memo runs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.harness.context_types import OutputStatus
from src.research_platform.memo_artifacts import MemoArtifactSet
from src.research_platform.memo_evaluator import EvaluatorEvaluationRef

RunStatus = Literal["succeeded", "partial", "failed", "timed_out"]
StateStatus = Literal["disabled", "succeeded", "failed", "no_change"]
DeliverableStatus = Literal["pass", "warn", "fail", "missing", "unparseable", "error", "timeout"]


class StatusReason(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    severity: Literal["info", "warn", "error"] = "warn"
    message: str


class ProposalApplyCounts(BaseModel):
    model_config = ConfigDict(extra="forbid")

    applied: int = 0
    rejected: int = 0
    revised: int = 0


class StateValidationMeta(BaseModel):
    model_config = ConfigDict(extra="allow")

    ok: bool | None = None
    errors: list[str] = Field(default_factory=list)


class StateCommitMeta(BaseModel):
    model_config = ConfigDict(extra="allow")

    committed: bool | None = None
    changed: bool | None = None


class ResearchStateRunMeta(BaseModel):
    model_config = ConfigDict(extra="allow")

    enabled: bool = False
    mode: str | None = None
    rolled_back: bool = False
    integrity_errors: list[str] = Field(default_factory=list)
    apply_counts: ProposalApplyCounts | None = None
    validation_report: StateValidationMeta | None = None
    commit: StateCommitMeta | None = None


class SplitMemoStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    raw_harness_status: str | None
    run_status: RunStatus
    state_status: StateStatus
    deliverable_status: DeliverableStatus
    legacy_status: OutputStatus
    status_reasons: list[StatusReason] = Field(default_factory=list)


def normalize_research_state_meta(meta: dict[str, Any] | ResearchStateRunMeta | None) -> ResearchStateRunMeta:
    if meta is None:
        return ResearchStateRunMeta(enabled=False)
    if isinstance(meta, ResearchStateRunMeta):
        return meta
    return ResearchStateRunMeta.model_validate(meta)


def derive_state_status(research_state_meta: dict[str, Any] | ResearchStateRunMeta | None) -> tuple[StateStatus, list[StatusReason]]:
    meta = normalize_research_state_meta(research_state_meta)
    reasons: list[StatusReason] = []
    if not meta.enabled:
        return "disabled", reasons
    if meta.rolled_back or meta.integrity_errors:
        reasons.append(StatusReason(code="research_state_failed", severity="error", message="Research state rolled back or integrity errors were present."))
        return "failed", reasons
    if meta.validation_report and meta.validation_report.ok is False:
        reasons.append(StatusReason(code="research_state_validation_failed", severity="error", message="Research state validation failed."))
        return "failed", reasons
    if meta.apply_counts is not None and (meta.apply_counts.applied + meta.apply_counts.revised) == 0:
        return "no_change", reasons
    if meta.commit is not None and meta.commit.committed is False:
        reasons.append(StatusReason(code="research_state_commit_failed", severity="error", message="Research state commit did not complete."))
        return "failed", reasons
    return "succeeded", reasons


def derive_deliverable_status(evaluator_ref: EvaluatorEvaluationRef | None) -> tuple[DeliverableStatus, list[StatusReason]]:
    if evaluator_ref is None:
        return "missing", [StatusReason(code="missing_evaluator_evaluation", severity="error", message="deliverable_evaluation.json was not produced.")]
    if evaluator_ref.status == "timeout":
        return "timeout", [StatusReason(code="evaluator_timeout", severity="error", message="Post-harness evaluator timed out.")]
    if evaluator_ref.status == "error":
        return "error", [StatusReason(code="evaluator_error", severity="error", message="Post-harness evaluator errored.")]
    if evaluator_ref.status == "unparseable":
        return "unparseable", [StatusReason(code="evaluator_unparseable", severity="error", message="Post-harness evaluator output was not parseable Pydantic JSON.")]
    if evaluator_ref.status == "missing" or evaluator_ref.verdict is None:
        return "missing", [StatusReason(code="missing_evaluator_evaluation", severity="error", message="Post-harness evaluator verdict is missing.")]
    if evaluator_ref.verdict.status == "pass":
        return "pass", []
    if evaluator_ref.verdict.status == "warn":
        return "warn", [StatusReason(code="deliverable_warn", severity="warn", message="Evaluator returned warn.")]
    return "fail", [StatusReason(code="deliverable_fail", severity="error", message="Evaluator returned fail.")]


def derive_run_status(
    *,
    harness_status: str | None,
    artifacts: MemoArtifactSet,
    state_status: StateStatus,
    deliverable_status: DeliverableStatus,
) -> tuple[RunStatus, list[StatusReason]]:
    reasons: list[StatusReason] = []
    if harness_status == "time_out" and not artifacts.has_usable_final:
        reasons.append(StatusReason(code="harness_timeout_no_deliverable", severity="error", message="Harness timed out without usable final.md."))
        return "timed_out", reasons
    if harness_status == "time_out_with_deliverable":
        reasons.append(StatusReason(code="harness_timeout_with_deliverable", severity="warn", message="Harness timed out after preserving a deliverable."))
    elif harness_status not in ("completed", None):
        reasons.append(StatusReason(code="harness_not_completed", severity="warn", message=f"Harness ended with status {harness_status}."))

    if state_status == "failed":
        return "failed", reasons
    if deliverable_status in ("fail", "missing", "unparseable", "error", "timeout"):
        return "failed", reasons
    if not artifacts.has_usable_final:
        return "failed", reasons
    if not artifacts.has_required_artifacts or deliverable_status == "warn" or harness_status in ("failed", "time_out"):
        return "partial", reasons
    return "succeeded", reasons


def derive_legacy_status(run_status: RunStatus, state_status: StateStatus, deliverable_status: DeliverableStatus) -> OutputStatus:
    if run_status == "succeeded" and state_status != "failed" and deliverable_status == "pass":
        return "succeeded"
    if run_status == "partial" or deliverable_status == "warn":
        return "partial"
    return "failed"


def derive_split_status(
    *,
    harness_status: str | None,
    artifacts: MemoArtifactSet,
    research_state_meta: dict[str, Any] | ResearchStateRunMeta | None,
    evaluator_ref: EvaluatorEvaluationRef | None,
) -> SplitMemoStatus:
    state_status, state_reasons = derive_state_status(research_state_meta)
    deliverable_status, deliverable_reasons = derive_deliverable_status(evaluator_ref)
    run_status, run_reasons = derive_run_status(
        harness_status=harness_status,
        artifacts=artifacts,
        state_status=state_status,
        deliverable_status=deliverable_status,
    )
    artifact_reasons = [StatusReason(code=r.code, severity=r.severity, message=r.message) for r in artifacts.reasons]
    reasons = [*artifact_reasons, *state_reasons, *deliverable_reasons, *run_reasons]
    legacy = derive_legacy_status(run_status, state_status, deliverable_status)
    return SplitMemoStatus(
        raw_harness_status=harness_status,
        run_status=run_status,
        state_status=state_status,
        deliverable_status=deliverable_status,
        legacy_status=legacy,
        status_reasons=reasons,
    )
