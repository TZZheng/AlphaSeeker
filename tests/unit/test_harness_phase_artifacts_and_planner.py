from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.harness.artifacts import initialize_dossier, sync_dossier
from src.harness.benchmark import (
    BenchmarkMetrics,
    BenchmarkResult,
    compare_benchmark_results,
    detect_model_role_changes,
)
from src.harness.claims import build_claim_map
from src.harness.contracts import build_default_contract
from src.harness.planner import plan_research, validate_research_plan
from src.harness.types import (
    EvidenceItem,
    HarnessRequest,
    HarnessState,
    ResearchBrief,
    ResearchPlan,
    ResearchStep,
    SkillCall,
)

pytestmark = pytest.mark.unit


def _basic_brief() -> ResearchBrief:
    return ResearchBrief(
        primary_question="Analyze AAPL valuation and risk.",
        sub_questions=["What matters for valuation?"],
        domain_packs=["equity"],
        likely_report_shape=[
            "Executive Summary",
            "Key Findings",
            "Risks and Counterevidence",
            "Sources",
        ],
    )


def _basic_plan() -> ResearchPlan:
    return ResearchPlan(
        primary_question="Analyze AAPL valuation and risk.",
        domain_packs=["equity"],
        required_sections=[
            "Executive Summary",
            "Key Findings",
            "Risks and Counterevidence",
            "Sources",
        ],
        counterevidence_topics=["Peer and competitor pressure"],
        steps=[
            ResearchStep(
                id="collect_context",
                objective="Collect broad research evidence.",
                recommended_skill_calls=[
                    SkillCall(name="search_and_read", arguments={"queries": ["AAPL risk"]})
                ],
            )
        ],
    )


def test_artifact_writer_creates_required_dossier_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    brief = _basic_brief()
    plan = _basic_plan()
    contract = build_default_contract(brief, plan)
    state = HarnessState(
        request=HarnessRequest(user_prompt="Analyze AAPL valuation and risk."),
        research_brief=brief,
        research_plan=plan,
        research_contract=contract,
    )

    initialize_dossier(state)
    sync_dossier(state)

    required = [
        "mission",
        "progress",
        "research_plan",
        "research_contract",
        "qa_report",
        "findings",
        "claim_map",
    ]
    for key in required:
        path = Path(state.dossier_paths[key])
        assert path.exists(), key
        if path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
        else:
            assert path.read_text(encoding="utf-8").startswith("# ")


def test_plan_validator_rejects_illegal_skill_names_and_cycles() -> None:
    illegal_skill_plan = ResearchPlan(
        primary_question="Analyze AAPL.",
        domain_packs=["equity"],
        steps=[
            ResearchStep(
                id="bad_step",
                objective="Call an illegal skill.",
                recommended_skill_calls=[SkillCall(name="totally_fake_skill", arguments={})],
            )
        ],
    )

    with pytest.raises(ValueError, match="Illegal skill name"):
        validate_research_plan(illegal_skill_plan)

    cyclic_plan = ResearchPlan.model_construct(
        primary_question="Analyze AAPL.",
        domain_packs=["equity"],
        steps=[
            ResearchStep(id="a", objective="A", depends_on=["b"]),
            ResearchStep(id="b", objective="B", depends_on=["a"]),
        ],
    )

    with pytest.raises(ValueError, match="cyclic"):
        validate_research_plan(cyclic_plan)


def test_claim_map_builder_marks_fact_vs_inference() -> None:
    state = HarnessState(
        request=HarnessRequest(user_prompt="Analyze AAPL."),
        evidence_ledger=[
            EvidenceItem(
                id="E1",
                skill_name="search_and_read",
                source_type="url",
                summary="Revenue growth evidence",
            ),
            EvidenceItem(
                id="E2",
                skill_name="search_and_read",
                source_type="url",
                summary="Competitor pressure risk",
            ),
        ],
    )

    draft = """# Response

## Key Findings
Apple revenue growth accelerated in 2025 based on the collected evidence [E1].

## Risks and Counterevidence
Inference: competitor pressure could compress margins if peer offerings improve [E2].
"""

    claims = build_claim_map(state, draft)

    assert len(claims) == 2
    assert claims[0].claim_type == "fact"
    assert claims[1].claim_type == "inference"
    assert claims[1].complicating_evidence_ids == ["E2"]


def test_benchmark_comparison_and_model_change_detection(tmp_path: Path) -> None:
    current = [
        BenchmarkResult(
            case_id="equity_single_name",
            lane_id="weak",
            metrics=BenchmarkMetrics(
                artifact_creation_success=True,
                contract_satisfaction="pass",
                citation_coverage="pass",
                missing_section_count=1,
                freshness_failures=0,
                numeric_inconsistency_count=0,
                counterevidence_gap_count=1,
                runtime_status="completed",
            ),
        )
    ]
    baseline = [
        BenchmarkResult(
            case_id="equity_single_name",
            lane_id="weak",
            metrics=BenchmarkMetrics(
                artifact_creation_success=True,
                contract_satisfaction="pass",
                citation_coverage="pass",
                missing_section_count=2,
                freshness_failures=1,
                numeric_inconsistency_count=0,
                counterevidence_gap_count=2,
                runtime_status="completed",
            ),
        )
    ]

    comparison = compare_benchmark_results(current, baseline)
    assert comparison["equity_single_name:weak"]["missing_section_delta"] == -1
    assert comparison["equity_single_name:weak"]["counterevidence_gap_delta"] == -1

    baseline_path = tmp_path / "baseline_models.yaml"
    current_path = tmp_path / "current_models.yaml"
    baseline_path.write_text("harness:\n  planner: sf/Qwen/Qwen3-8B\n", encoding="utf-8")
    current_path.write_text(
        "harness:\n  planner: kimi-k2.5\n  evaluator: kimi-k2.5\n",
        encoding="utf-8",
    )

    changed = detect_model_role_changes(str(current_path), str(baseline_path))
    assert changed["planner"] == ("sf/Qwen/Qwen3-8B", "kimi-k2.5")
    assert changed["evaluator"] == ("", "kimi-k2.5")


def test_plan_research_respects_selected_pack_override() -> None:
    brief, plan, contract = plan_research(
        "Analyze AAPL with macro context.",
        selected_packs=["core", "equity"],
    )

    assert brief.domain_packs == ["equity"]
    assert plan.domain_packs == ["equity"]
    assert contract.counterevidence_clauses
