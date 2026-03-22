from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pytest

from src.harness.runtime import run_harness
from src.harness.types import (
    ControllerDecision,
    EvidenceItem,
    HarnessRequest,
    SkillResult,
    SkillSpec,
    VerificationReport,
)

pytestmark = pytest.mark.component


def _make_skill(
    name: str,
    pack: str,
    *,
    status: str = "ok",
    summary: str | None = None,
    output_text: str | None = None,
    evidence_suffix: str | None = None,
) -> SkillSpec:
    def _executor(arguments: dict[str, Any], _state) -> SkillResult:
        evidence = []
        if evidence_suffix is not None:
            evidence.append(
                EvidenceItem(
                    skill_name=name,
                    source_type="note",
                    summary=f"{name} evidence {evidence_suffix}",
                    content=f"{name} content {evidence_suffix}",
                )
            )
        return SkillResult(
            skill_name=name,
            arguments=arguments,
            status=status,  # type: ignore[arg-type]
            summary=summary or f"{name} executed",
            output_text=output_text,
            evidence=evidence,
            error=None if status != "failed" else f"{name} failed",
        )

    return SkillSpec(
        name=name,
        description=f"{name} stub skill",
        pack=pack,
        input_schema={"query": "string"},
        executor=_executor,
    )


def _load_trace(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_run_harness_single_domain_with_stubbed_components(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    registry = {"search_and_read": _make_skill("search_and_read", "core", evidence_suffix="one")}
    decisions = iter(
        [
            ControllerDecision(
                action="call_skill",
                rationale="Gather evidence",
                skill_call={"name": "search_and_read", "arguments": {"queries": ["Analyze AAPL"]}},
            ),
            ControllerDecision(action="draft", rationale="Draft now"),
        ]
    )

    response = run_harness(
        HarnessRequest(user_prompt="Analyze AAPL", selected_packs=["core"]),
        controller_fn=lambda _state: next(decisions),
        writer_fn=lambda _state: "# Draft\n\nApple looks interesting [E1].\n\n## Sources\n- [E1]",
        verifier_fn=lambda _state, _draft: VerificationReport(
            decision="pass",
            summary="Looks grounded.",
            grounding="pass",
            completeness="pass",
            numeric_consistency="pass",
            citation_coverage="pass",
            formatting="pass",
        ),
        registry=registry,
    )

    assert response.status == "completed"
    assert "Apple looks interesting" in response.final_response
    assert response.report_path and Path(response.report_path).exists()
    assert response.trace_path and Path(response.trace_path).exists()

    trace = _load_trace(response.trace_path)
    assert trace["enabled_packs"] == ["core"]
    assert trace["skill_history"][0]["skill_name"] == "search_and_read"
    assert trace["evidence_ledger"][0]["id"] == "E1"


def test_run_harness_cross_domain_sequence_records_multiple_skills(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    registry = {
        "search_and_read": _make_skill("search_and_read", "core", evidence_suffix="one"),
        "fetch_financials": _make_skill("fetch_financials", "equity", evidence_suffix="two"),
    }
    decisions = iter(
        [
            ControllerDecision(
                action="call_skill",
                rationale="Collect context",
                skill_call={"name": "search_and_read", "arguments": {"queries": ["AAPL"]}},
            ),
            ControllerDecision(
                action="call_skill",
                rationale="Collect financials",
                skill_call={"name": "fetch_financials", "arguments": {"ticker": "AAPL"}},
            ),
            ControllerDecision(action="draft", rationale="Write answer"),
        ]
    )

    response = run_harness(
        HarnessRequest(user_prompt="Analyze AAPL", selected_packs=["core", "equity"]),
        controller_fn=lambda _state: next(decisions),
        writer_fn=lambda _state: "# Draft\n\nEvidence spans [E1] and [E2].\n\n## Sources\n- [E1]\n- [E2]",
        verifier_fn=lambda _state, _draft: VerificationReport(
            decision="pass",
            summary="Looks grounded.",
            grounding="pass",
            completeness="pass",
            numeric_consistency="pass",
            citation_coverage="pass",
            formatting="pass",
        ),
        registry=registry,
    )

    assert response.skills_used == ["search_and_read", "fetch_financials"]

    trace = _load_trace(response.trace_path or "")
    assert len(trace["skill_history"]) == 2
    assert trace["evidence_ledger"][1]["id"] == "E2"


def test_run_harness_keeps_partial_skill_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    registry = {"search_and_read": _make_skill("search_and_read", "core", status="partial", evidence_suffix="one")}
    decisions = iter(
        [
            ControllerDecision(
                action="call_skill",
                rationale="Gather evidence",
                skill_call={"name": "search_and_read", "arguments": {"queries": ["AAPL"]}},
            ),
            ControllerDecision(action="draft", rationale="Draft with partial evidence"),
        ]
    )

    response = run_harness(
        HarnessRequest(user_prompt="Analyze AAPL", selected_packs=["core"]),
        controller_fn=lambda _state: next(decisions),
        writer_fn=lambda _state: "# Draft\n\nPartial evidence still exists [E1].\n\n## Sources\n- [E1]",
        verifier_fn=lambda _state, _draft: VerificationReport(
            decision="pass",
            summary="Acceptable.",
            grounding="pass",
            completeness="pass",
            numeric_consistency="pass",
            citation_coverage="pass",
            formatting="pass",
        ),
        registry=registry,
    )

    trace = _load_trace(response.trace_path or "")
    assert trace["skill_history"][0]["status"] == "partial"
    assert response.status == "completed"


def test_run_harness_uses_verifier_feedback_for_extra_research(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    registry = {
        "search_and_read": _make_skill("search_and_read", "core", evidence_suffix="one"),
        "fetch_financials": _make_skill("fetch_financials", "equity", evidence_suffix="two"),
    }

    def _controller(state) -> ControllerDecision:
        if not state.skill_history:
            return ControllerDecision(
                action="call_skill",
                rationale="Initial evidence",
                skill_call={"name": "search_and_read", "arguments": {"queries": ["AAPL"]}},
            )
        if not state.latest_draft:
            return ControllerDecision(action="draft", rationale="Draft the first answer")
        if state.verification_reports and state.verification_reports[-1].decision == "revise" and len(state.skill_history) == 1:
            return ControllerDecision(
                action="call_skill",
                rationale="Address critique with more evidence",
                skill_call={"name": "fetch_financials", "arguments": {"ticker": "AAPL"}},
            )
        return ControllerDecision(action="draft", rationale="Redraft with new evidence")

    def _writer(state) -> str:
        if len(state.skill_history) == 1:
            return "# Draft\n\nNeed more support [E1].\n\n## Sources\n- [E1]"
        return "# Draft\n\nNow supported by [E1] and [E2].\n\n## Sources\n- [E1]\n- [E2]"

    verifier_reports = iter(
        [
            VerificationReport(
                decision="revise",
                summary="Need more source coverage.",
                grounding="revise",
                completeness="pass",
                numeric_consistency="pass",
                citation_coverage="revise",
                formatting="pass",
                improvement_instructions=["Add another grounded source."],
            ),
            VerificationReport(
                decision="pass",
                summary="Now grounded.",
                grounding="pass",
                completeness="pass",
                numeric_consistency="pass",
                citation_coverage="pass",
                formatting="pass",
            ),
        ]
    )

    response = run_harness(
        HarnessRequest(user_prompt="Analyze AAPL", selected_packs=["core", "equity"]),
        controller_fn=_controller,
        writer_fn=_writer,
        verifier_fn=lambda _state, _draft: next(verifier_reports),
        registry=registry,
    )

    trace = _load_trace(response.trace_path or "")
    assert response.status == "completed"
    assert len(trace["verification_reports"]) == 2
    assert len(trace["skill_history"]) == 2
    assert trace["revision_count"] == 1


def test_run_harness_enforces_revision_budget_and_condenses_large_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "src.harness.runtime.condense_context",
        lambda text, max_chars, agent, purpose: text[:max_chars],
    )

    registry = {
        "search_and_read": _make_skill(
            "search_and_read",
            "core",
            evidence_suffix="one",
            output_text="x" * 8000,
        )
    }
    decisions = iter(
        [
            ControllerDecision(
                action="call_skill",
                rationale="Gather evidence",
                skill_call={"name": "search_and_read", "arguments": {"queries": ["AAPL"]}},
            ),
            ControllerDecision(action="draft", rationale="Draft"),
            ControllerDecision(action="finalize", rationale="Budget exhausted"),
        ]
    )

    response = run_harness(
        HarnessRequest(
            user_prompt="Analyze AAPL",
            selected_packs=["core"],
            max_revision_rounds=0,
            max_chars_before_condense=6000,
        ),
        controller_fn=lambda _state: next(decisions),
        writer_fn=lambda _state: "# Draft\n\nCited [E1].\n\n## Sources\n- [E1]",
        verifier_fn=lambda _state, _draft: VerificationReport(
            decision="revise",
            summary="Needs more work but revision budget is zero.",
            grounding="revise",
            completeness="pass",
            numeric_consistency="pass",
            citation_coverage="pass",
            formatting="pass",
            improvement_instructions=["Revise if possible."],
        ),
        registry=registry,
    )

    trace = _load_trace(response.trace_path or "")
    assert response.status == "completed"
    assert trace["skill_history"][0]["structured_data"]["condensed"] is True
    assert trace["revision_count"] == 0
