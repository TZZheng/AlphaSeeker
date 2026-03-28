from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from src.harness.controller import decide_next_step
from src.harness.planner import plan_research
from src.harness.registry import build_skill_registry, get_skills_for_packs
from src.harness.runtime import run_harness
from src.harness.types import (
    ControllerDecision,
    EvidenceItem,
    HarnessRequest,
    HarnessState,
    ReportSectionFeedback,
    ResearchBrief,
    ResearchPlan,
    ResearchStep,
    SkillCall,
    SkillResult,
    SkillSpec,
    VerificationReport,
)
from src.harness.verifier import verify_draft
from src.harness.writer import write_draft

pytestmark = pytest.mark.component


def _make_skill(
    name: str,
    pack: str,
    *,
    sleep_seconds: float = 0.0,
    evidence_suffix: str | None = None,
) -> SkillSpec:
    def _executor(arguments: dict[str, Any], _state) -> SkillResult:
        if sleep_seconds:
            time.sleep(sleep_seconds)
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
            status="ok",
            summary=f"{name} executed",
            evidence=evidence,
        )

    return SkillSpec(
        name=name,
        description=f"{name} stub skill",
        pack=pack,
        input_schema={"query": "string"},
        executor=_executor,
    )


def _planner_core_two_call(_state: HarnessState, _registry: dict[str, object]):
    brief = ResearchBrief(
        primary_question="Analyze AAPL",
        domain_packs=[],
        likely_report_shape=["Executive Summary", "Sources"],
    )
    plan = ResearchPlan(
        primary_question="Analyze AAPL",
        domain_packs=[],
        required_sections=["Executive Summary", "Sources"],
        steps=[
            ResearchStep(
                id="collect_context",
                objective="Collect evidence via two skill calls.",
                recommended_skill_calls=[
                    SkillCall(name="search_and_read", arguments={"queries": ["AAPL"]}),
                    SkillCall(name="search_news", arguments={"query": "AAPL"}),
                ],
                parallel_safe=True,
            )
        ],
    )
    from src.harness.contracts import build_default_contract

    contract = build_default_contract(brief, plan)
    return brief, plan, contract


def test_progress_artifact_records_major_phases(
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
        HarnessRequest(user_prompt="Analyze AAPL", selected_packs=["core"], run_id="phase-log-run"),
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

    progress_text = Path(response.dossier_paths["progress"]).read_text(encoding="utf-8")
    assert "planner:" in progress_text
    assert "controller:" in progress_text
    assert "worker:" in progress_text
    assert "writer:" in progress_text
    assert "evaluator:" in progress_text


def test_controller_schedules_only_unblocked_steps(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.harness.controller as controller_module

    monkeypatch.setattr(
        controller_module,
        "get_llm",
        lambda _model: (_ for _ in ()).throw(RuntimeError("offline")),
    )

    brief, plan, contract = plan_research(
        "How do higher rates affect JPM and bank margins?",
        selected_packs=["equity", "macro"],
    )
    registry = build_skill_registry()
    state = HarnessState(
        request=HarnessRequest(user_prompt="How do higher rates affect JPM and bank margins?"),
        research_brief=brief,
        research_plan=plan,
        research_contract=contract,
        enabled_packs=["core", *plan.domain_packs],
        available_skills=get_skills_for_packs(registry, ["core", *plan.domain_packs]),
        step_statuses={"collect_context": "completed"},
    )

    decision = decide_next_step(state)

    assert decision.action in {"execute_step", "execute_parallel_steps"}
    scheduled = decision.step_ids or ([decision.step_id] if decision.step_id else [])
    assert scheduled
    assert "peer_challenge" not in scheduled
    for step_id in scheduled:
        step = next(step for step in plan.steps if step.id == step_id)
        assert all(
            state.step_statuses.get(dep) in {"completed", "partial", "skipped"}
            for dep in step.depends_on
        )


def test_step_worker_completes_multi_call_step_and_updates_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    registry = {
        "search_and_read": _make_skill("search_and_read", "core", evidence_suffix="one"),
        "search_news": _make_skill("search_news", "core", evidence_suffix="two"),
    }
    decisions = iter(
        [
            ControllerDecision(action="execute_step", rationale="Run the plan.", step_id="collect_context"),
            ControllerDecision(action="draft", rationale="Draft now"),
        ]
    )

    response = run_harness(
        HarnessRequest(user_prompt="Analyze AAPL", selected_packs=["core"], run_id="worker-two-call"),
        planner_fn=_planner_core_two_call,
        controller_fn=lambda _state: next(decisions),
        writer_fn=lambda _state: "# Draft\n\nCombined evidence [E1] and [E2].\n\n## Sources\n- [E1]\n- [E2]",
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

    trace = json.loads(Path(response.trace_path or "").read_text(encoding="utf-8"))
    progress_text = Path(response.dossier_paths["progress"]).read_text(encoding="utf-8")
    assert len(trace["skill_history"]) == 2
    assert trace["step_results"][0]["status"] == "completed"
    assert "collect_context" in progress_text


def test_writer_revises_only_targeted_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.harness.writer as writer_module

    monkeypatch.setattr(
        writer_module,
        "get_llm",
        lambda _model: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    state = HarnessState(
        request=HarnessRequest(user_prompt="Analyze AAPL"),
        latest_draft=(
            "# Response to: Analyze AAPL\n\n"
            "## Executive Summary\nStable section [E1]\n\n"
            "## Risks and Counterevidence\nOld risk text [E2]\n\n"
            "## Sources\n- [E1]\n- [E2]"
        ),
        research_plan=ResearchPlan(
            primary_question="Analyze AAPL",
            domain_packs=[],
            required_sections=["Executive Summary", "Risks and Counterevidence", "Sources"],
        ),
        evidence_ledger=[
            EvidenceItem(id="E1", skill_name="s", source_type="note", summary="Summary evidence"),
            EvidenceItem(id="E2", skill_name="s", source_type="note", summary="Counterevidence risk"),
        ],
        verification_reports=[
            VerificationReport(
                decision="revise",
                summary="Revise the risk section.",
                grounding="pass",
                completeness="revise",
                numeric_consistency="pass",
                citation_coverage="pass",
                formatting="pass",
                report_section_feedback=[
                    ReportSectionFeedback(
                        section_label="Risks and Counterevidence",
                        issue="Weak section",
                        why_it_fails="Too generic.",
                        suggested_fix="Add explicit counterevidence.",
                    )
                ],
            )
        ],
    )

    revised = write_draft(state)

    assert "## Executive Summary\nStable section [E1]" in revised
    assert "Counterevidence matters here" in revised


def test_resume_from_checkpoint_restores_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.harness.controller as controller_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        controller_module,
        "get_llm",
        lambda _model: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    registry = {"search_and_read": _make_skill("search_and_read", "core", evidence_suffix="one")}
    initial_decisions = iter(
        [
            ControllerDecision(action="execute_step", rationale="Run the plan.", step_id="collect_context"),
            ControllerDecision(action="draft", rationale="Draft now"),
        ]
    )

    initial = run_harness(
        HarnessRequest(user_prompt="Analyze AAPL", selected_packs=["core"], run_id="resume-base"),
        planner_fn=lambda state, reg: (
            ResearchBrief(primary_question=state.request.user_prompt, domain_packs=[], likely_report_shape=["Executive Summary", "Sources"]),
            ResearchPlan(
                primary_question=state.request.user_prompt,
                domain_packs=[],
                required_sections=["Executive Summary", "Sources"],
                steps=[
                    ResearchStep(
                        id="collect_context",
                        objective="Collect evidence.",
                        recommended_skill_calls=[
                            SkillCall(name="search_and_read", arguments={"queries": ["AAPL"]})
                        ],
                    )
                ],
            ),
            __import__("src.harness.contracts", fromlist=["build_default_contract"]).build_default_contract(
                ResearchBrief(primary_question=state.request.user_prompt, domain_packs=[], likely_report_shape=["Executive Summary", "Sources"]),
                ResearchPlan(
                    primary_question=state.request.user_prompt,
                    domain_packs=[],
                    required_sections=["Executive Summary", "Sources"],
                    steps=[
                        ResearchStep(
                            id="collect_context",
                            objective="Collect evidence.",
                            recommended_skill_calls=[
                                SkillCall(name="search_and_read", arguments={"queries": ["AAPL"]})
                            ],
                        )
                    ],
                ),
            ),
        ),
        controller_fn=lambda _state: next(initial_decisions),
        writer_fn=lambda _state: "# Draft\n\nGrounded [E1].\n\n## Sources\n- [E1]",
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

    trace_payload = json.loads(Path(initial.trace_path or "").read_text(encoding="utf-8"))
    worker_checkpoint = next(path for path in trace_payload["checkpoint_paths"] if path.endswith("_worker.json"))

    resumed = run_harness(
        HarnessRequest(
            user_prompt="Analyze AAPL",
            selected_packs=["core"],
            resume_from=worker_checkpoint,
            run_id="resume-base",
        ),
        writer_fn=lambda _state: "# Draft\n\nGrounded [E1].\n\n## Sources\n- [E1]",
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

    assert resumed.final_response == initial.final_response
    assert Path(resumed.dossier_paths["mission"]).exists()


def test_parallel_execution_reduces_wall_clock_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    registry = {
        "search_and_read": _make_skill("search_and_read", "core", sleep_seconds=0.2, evidence_suffix="one"),
        "search_news": _make_skill("search_news", "core", sleep_seconds=0.2, evidence_suffix="two"),
    }

    def _parallel_planner(state: HarnessState, _registry: dict[str, object]):
        brief = ResearchBrief(primary_question=state.request.user_prompt, domain_packs=[], likely_report_shape=["Executive Summary", "Sources"])
        plan = ResearchPlan(
            primary_question=state.request.user_prompt,
            domain_packs=[],
            required_sections=["Executive Summary", "Sources"],
            steps=[
                ResearchStep(
                    id="a",
                    objective="Collect A.",
                    recommended_skill_calls=[SkillCall(name="search_and_read", arguments={"queries": ["A"]})],
                    parallel_safe=True,
                ),
                ResearchStep(
                    id="b",
                    objective="Collect B.",
                    recommended_skill_calls=[SkillCall(name="search_news", arguments={"query": "B"})],
                    parallel_safe=True,
                ),
            ],
        )
        from src.harness.contracts import build_default_contract

        return brief, plan, build_default_contract(brief, plan)

    sequential_decisions = iter(
        [
            ControllerDecision(action="execute_step", rationale="Run A.", step_id="a"),
            ControllerDecision(action="execute_step", rationale="Run B.", step_id="b"),
            ControllerDecision(action="draft", rationale="Draft"),
        ]
    )
    start = time.perf_counter()
    run_harness(
        HarnessRequest(user_prompt="Parallel benchmark", selected_packs=["core"], run_id="sequential-run", allow_parallel_steps=False),
        planner_fn=_parallel_planner,
        controller_fn=lambda _state: next(sequential_decisions),
        writer_fn=lambda _state: "# Draft\n\nUsed [E1] and [E2].\n\n## Sources\n- [E1]\n- [E2]",
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
    sequential_elapsed = time.perf_counter() - start

    parallel_decisions = iter(
        [
            ControllerDecision(action="execute_parallel_steps", rationale="Run both.", step_ids=["a", "b"]),
            ControllerDecision(action="draft", rationale="Draft"),
        ]
    )
    start = time.perf_counter()
    run_harness(
        HarnessRequest(user_prompt="Parallel benchmark", selected_packs=["core"], run_id="parallel-run", allow_parallel_steps=True),
        planner_fn=_parallel_planner,
        controller_fn=lambda _state: next(parallel_decisions),
        writer_fn=lambda _state: "# Draft\n\nUsed [E1] and [E2].\n\n## Sources\n- [E1]\n- [E2]",
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
    parallel_elapsed = time.perf_counter() - start

    assert parallel_elapsed < sequential_elapsed


def test_evaluator_follow_up_schema_accepts_section_feedback(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.harness.verifier as verifier_module

    monkeypatch.setattr(
        verifier_module,
        "get_llm",
        lambda _model: (_ for _ in ()).throw(RuntimeError("offline")),
    )

    state = HarnessState(
        request=HarnessRequest(user_prompt="Analyze AAPL"),
        research_plan=ResearchPlan(
            primary_question="Analyze AAPL",
            domain_packs=[],
            required_sections=["Executive Summary", "Risks and Counterevidence", "Sources"],
            counterevidence_topics=["Bear case"],
            steps=[
                ResearchStep(
                    id="collect_context",
                    objective="Collect evidence.",
                    recommended_skill_calls=[SkillCall(name="search_and_read", arguments={"queries": ["AAPL"]})],
                )
            ],
        ),
        evidence_ledger=[
            EvidenceItem(id="E1", skill_name="s", source_type="note", summary="Positive evidence")
        ],
    )

    report = verify_draft(
        state,
        "# Draft\n\n## Executive Summary\nClaim without citations.\n\n## Sources\n- [E1]",
    )

    assert report.report_section_feedback
    assert report.required_follow_up_calls
