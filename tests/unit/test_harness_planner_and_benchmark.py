from __future__ import annotations

import os

import pytest

from src.harness.benchmark import WEAK_MODEL_LANE, harness_model_lane
from src.harness.planner import validate_research_plan
from src.harness.types import ResearchPlan, ResearchStep, SkillCall
from src.harness.writer import build_claim_map
from src.harness.types import HarnessRequest, HarnessState

pytestmark = pytest.mark.unit


def test_validate_research_plan_rejects_unknown_skill() -> None:
    plan = ResearchPlan(
        primary_question="Analyze AAPL",
        domain_packs=["equity"],
        steps=[
            ResearchStep(
                id="s1",
                objective="Collect profile",
                recommended_skill_calls=[SkillCall(name="unknown_skill", arguments={})],
            )
        ],
    )

    with pytest.raises(ValueError):
        validate_research_plan(plan, {"fetch_company_profile"})


def test_validate_research_plan_rejects_cyclic_dependencies() -> None:
    plan = ResearchPlan(
        primary_question="Analyze AAPL",
        domain_packs=["equity"],
        steps=[
            ResearchStep(id="s1", objective="One", depends_on=["s2"]),
            ResearchStep(id="s2", objective="Two", depends_on=["s1"]),
        ],
    )

    with pytest.raises(ValueError):
        validate_research_plan(plan, set())


def test_build_claim_map_marks_fact_and_inference() -> None:
    state = HarnessState(request=HarnessRequest(user_prompt="Analyze AAPL"))
    draft = """# Summary

Apple reported revenue resilience [E1].

## Risks

Inference: margins may compress if competition intensifies [E2].
"""

    claims = build_claim_map(state, draft)

    assert len(claims) == 2
    assert claims[0].claim_type == "fact"
    assert claims[0].supporting_evidence_ids == ["E1"]
    assert claims[1].claim_type == "inference"
    assert claims[1].supporting_evidence_ids == ["E2"]


def test_harness_model_lane_sets_and_restores_env_vars() -> None:
    key = "ALPHASEEKER_MODEL_HARNESS_CONTROLLER"
    original = os.environ.get(key)

    with harness_model_lane(WEAK_MODEL_LANE):
        assert os.environ[key] == WEAK_MODEL_LANE.model_name

    assert os.environ.get(key) == original
