from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from src.harness.runtime import run_harness
from src.harness.types import (
    ContractClause,
    ControllerDecision,
    EvidenceItem,
    HarnessRequest,
    ResearchBrief,
    ResearchContract,
    ResearchPlan,
    ResearchStep,
    SkillResult,
    SkillSpec,
    StepExecutionResult,
    VerificationReport,
)

pytestmark = pytest.mark.component


def _load_json(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _make_skill(name: str, pack: str, evidence_suffix: str) -> SkillSpec:
    def _executor(arguments: dict[str, Any], _state) -> SkillResult:
        return SkillResult(
            skill_name=name,
            arguments=arguments,
            status="ok",
            summary=f"{name} executed",
            evidence=[
                EvidenceItem(
                    skill_name=name,
                    source_type="note",
                    summary=f"{name} evidence {evidence_suffix}",
                    content=f"{name} content {evidence_suffix}",
                )
            ],
        )

    return SkillSpec(
        name=name,
        description=f"{name} stub skill",
        pack=pack,
        input_schema={},
        executor=_executor,
    )


def _planner_for_equity(_state) -> tuple[ResearchBrief, ResearchPlan, ResearchContract]:
    brief = ResearchBrief(
        primary_question="Analyze AAPL",
        sub_questions=["Analyze AAPL"],
        domain_packs=["equity"],
        likely_report_shape=["Summary", "Risks"],
        rationale="stub",
    )
    plan = ResearchPlan(
        primary_question="Analyze AAPL",
        domain_packs=["equity"],
        required_sections=["Summary", "Risks"],
        counterevidence_topics=["Competitor pressure"],
        steps=[
            ResearchStep(
                id="s1",
                objective="Collect context",
                recommended_skill_calls=[{"name": "search_and_read", "arguments": {"queries": ["AAPL"]}}],
                can_run_parallel=True,
            ),
            ResearchStep(
                id="s2",
                objective="Collect financials",
                recommended_skill_calls=[{"name": "fetch_financials", "arguments": {"ticker": "AAPL"}}],
                can_run_parallel=True,
            ),
        ],
        rationale="stub",
    )
    contract = ResearchContract(
        global_clauses=[ContractClause(id="c1", category="citations", text="Cite evidence IDs.")],
        section_clauses=[
            ContractClause(id="c2", category="sections", text="Include Summary", applies_to_sections=["Summary"]),
            ContractClause(id="c3", category="sections", text="Include Risks", applies_to_sections=["Risks"]),
        ],
        counterevidence_clauses=[
            ContractClause(id="c4", category="counterevidence", text="Include competitor pressure.")
        ],
    )
    return brief, plan, contract


def test_run_harness_writes_dossier_and_claim_map(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    registry = {
        "search_and_read": _make_skill("search_and_read", "core", "one"),
        "fetch_financials": _make_skill("fetch_financials", "equity", "two"),
    }
    decisions = iter(
        [
            ControllerDecision(action="run_steps", rationale="Collect", step_ids=["s1", "s2"]),
            ControllerDecision(action="draft", rationale="Write"),
        ]
    )

    response = run_harness(
        HarnessRequest(user_prompt="Analyze AAPL", selected_packs=["equity"]),
        planner_fn=_planner_for_equity,
        controller_fn=lambda _state: next(decisions),
        writer_fn=lambda _state: "# Summary\n\nAAPL is supported [E1] and [E2].\n\n## Risks\n\nCompetitor pressure matters [E2].\n\n## Sources\n- [E1]\n- [E2]",
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

    trace = _load_json(response.trace_path or "")
    assert response.status == "completed"
    assert Path(trace["mission_path"]).exists()
    assert Path(trace["progress_path"]).exists()
    assert Path(trace["plan_path"]).exists()
    assert Path(trace["contract_path"]).exists()
    assert len(trace["step_results"]) == 2
    assert len(trace["claim_map"]) >= 2


def test_run_harness_can_resume_from_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    registry = {"search_and_read": _make_skill("search_and_read", "core", "one")}
    first_decisions = iter(
        [
            ControllerDecision(action="run_step", rationale="Collect", step_id="s1"),
            ControllerDecision(action="finalize", rationale="Stop early"),
        ]
    )

    def _single_step_planner(_state):
        brief = ResearchBrief(primary_question="Analyze AAPL", domain_packs=[], rationale="stub")
        plan = ResearchPlan(
            primary_question="Analyze AAPL",
            domain_packs=[],
            required_sections=["Summary"],
            steps=[
                ResearchStep(
                    id="s1",
                    objective="Collect context",
                    recommended_skill_calls=[{"name": "search_and_read", "arguments": {"queries": ["AAPL"]}}],
                )
            ],
        )
        contract = ResearchContract(global_clauses=[ContractClause(id="c1", category="citations", text="Cite evidence IDs.")])
        return brief, plan, contract

    first_response = run_harness(
        HarnessRequest(user_prompt="Analyze AAPL", selected_packs=[]),
        planner_fn=_single_step_planner,
        controller_fn=lambda _state: next(first_decisions),
        writer_fn=lambda _state: "# Summary\n\nContext [E1].\n\n## Sources\n- [E1]",
        verifier_fn=lambda _state, _draft: VerificationReport(
            decision="pass",
            summary="pass",
            grounding="pass",
            completeness="pass",
            numeric_consistency="pass",
            citation_coverage="pass",
            formatting="pass",
        ),
        registry=registry,
    )
    checkpoint_path = _load_json(first_response.trace_path or "")["checkpoint_paths"][1]

    second_decisions = iter([ControllerDecision(action="draft", rationale="Draft after resume")])
    second_response = run_harness(
        HarnessRequest(
            user_prompt="Analyze AAPL",
            selected_packs=[],
            resume_from_checkpoint=checkpoint_path,
        ),
        controller_fn=lambda _state: next(second_decisions),
        writer_fn=lambda _state: "# Summary\n\nResumed context [E1].\n\n## Sources\n- [E1]",
        verifier_fn=lambda _state, _draft: VerificationReport(
            decision="pass",
            summary="pass",
            grounding="pass",
            completeness="pass",
            numeric_consistency="pass",
            citation_coverage="pass",
            formatting="pass",
        ),
        registry=registry,
    )

    trace = _load_json(second_response.trace_path or "")
    assert second_response.status == "completed"
    assert "s1" in trace["completed_step_ids"]
    assert len(trace["evidence_ledger"]) >= 1


def test_run_harness_parallel_step_batch_is_faster_than_serial_worker_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)

    registry = {"search_and_read": _make_skill("search_and_read", "core", "one")}

    def _planner(_state):
        brief = ResearchBrief(primary_question="Analyze AAPL", domain_packs=[], rationale="stub")
        plan = ResearchPlan(
            primary_question="Analyze AAPL",
            domain_packs=[],
            steps=[
                ResearchStep(id="s1", objective="One", can_run_parallel=True),
                ResearchStep(id="s2", objective="Two", can_run_parallel=True),
            ],
        )
        contract = ResearchContract()
        return brief, plan, contract

    start_times: dict[str, float] = {}

    def _worker(step, _state, _registry):
        start_times[step.id] = time.perf_counter()
        time.sleep(0.25)
        return [], StepExecutionResult(step_id=step.id, status="completed", summary=f"{step.id} done")

    decisions = iter(
        [
            ControllerDecision(action="run_steps", rationale="Parallel", step_ids=["s1", "s2"]),
            ControllerDecision(action="draft", rationale="Write"),
        ]
    )

    response = run_harness(
        HarnessRequest(user_prompt="Analyze AAPL", enable_parallel_steps=True),
        planner_fn=_planner,
        controller_fn=lambda _state: next(decisions),
        worker_fn=_worker,
        writer_fn=lambda _state: "# Summary\n\nDone.\n\n## Sources\n- none",
        verifier_fn=lambda _state, _draft: VerificationReport(
            decision="pass",
            summary="pass",
            grounding="pass",
            completeness="pass",
            numeric_consistency="pass",
            citation_coverage="pass",
            formatting="pass",
        ),
        registry=registry,
    )

    trace = _load_json(response.trace_path or "")
    assert response.status == "completed"
    assert len(trace["step_results"]) == 2
    assert set(start_times) == {"s1", "s2"}
    assert abs(start_times["s1"] - start_times["s2"]) < 0.15
