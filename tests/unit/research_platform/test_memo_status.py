from __future__ import annotations

from src.research_platform.memo_artifacts import MemoArtifactPaths, MemoArtifactSet
from src.research_platform.memo_evaluator import EvaluatorEvaluationRef, MemoEvaluatorVerdict
from src.research_platform.memo_status import (
    derive_legacy_status,
    derive_split_status,
)


def _artifacts(*, usable_final: bool = True, required: bool = True) -> MemoArtifactSet:
    missing = [] if usable_final else ["final.md"]
    if usable_final and not required:
        missing.append("source_use_table.md")
    return MemoArtifactSet(
        paths=MemoArtifactPaths(),
        has_usable_final=usable_final,
        has_required_artifacts=required and usable_final,
        missing_required=missing,
        optional_present=[],
        reasons=[],
    )


def _eval_ref(status: str, verdict_status: str | None = None) -> EvaluatorEvaluationRef:
    verdict = None
    if verdict_status:
        verdict = MemoEvaluatorVerdict(
            status=verdict_status,
            rationale="judge rationale",
            blockers=[],
            warnings=[],
            source_review="sources reviewed",
            logic_review="logic reviewed",
            trace_review="trace reviewed",
            link_review="links reviewed",
            state_linkage_review=None,
            inputs_received=[],
            inputs_missing=[],
            evaluator_model="test-model",
            prompt_sha256="p",
            inputs_sha256="i",
            created_at="2026-05-15T00:00:00Z",
        )
    return EvaluatorEvaluationRef(status=status, path="/tmp/deliverable_evaluation.json", reason=[], verdict=verdict)


def test_missing_evaluation_blocks_legacy_success():
    split = derive_split_status(
        harness_status="completed",
        artifacts=_artifacts(),
        research_state_meta=None,
        evaluator_ref=None,
    )

    assert split.deliverable_status == "missing"
    assert split.legacy_status == "failed"
    assert split.run_status == "failed"


def test_warn_deliverable_maps_legacy_partial():
    split = derive_split_status(
        harness_status="completed",
        artifacts=_artifacts(),
        research_state_meta=None,
        evaluator_ref=_eval_ref("ok", "warn"),
    )

    assert split.deliverable_status == "warn"
    assert split.run_status == "partial"
    assert split.legacy_status == "partial"


def test_timeout_with_deliverable_lenient_rule_succeeds_with_reason():
    split = derive_split_status(
        harness_status="time_out_with_deliverable",
        artifacts=_artifacts(),
        research_state_meta=None,
        evaluator_ref=_eval_ref("ok", "pass"),
    )

    assert split.run_status == "succeeded"
    assert split.legacy_status == "succeeded"
    assert any(reason.code == "harness_timeout_with_deliverable" for reason in split.status_reasons)


def test_time_out_without_usable_final_is_timed_out_and_legacy_failed():
    split = derive_split_status(
        harness_status="time_out",
        artifacts=_artifacts(usable_final=False),
        research_state_meta=None,
        evaluator_ref=_eval_ref("missing", None),
    )

    assert split.run_status == "timed_out"
    assert split.legacy_status == "failed"


def test_research_state_failure_blocks_success():
    split = derive_split_status(
        harness_status="completed",
        artifacts=_artifacts(),
        research_state_meta={"enabled": True, "rolled_back": True, "integrity_errors": ["bad state"]},
        evaluator_ref=_eval_ref("ok", "pass"),
    )

    assert split.state_status == "failed"
    assert split.run_status == "failed"
    assert derive_legacy_status(split.run_status, split.state_status, split.deliverable_status) == "failed"
