from __future__ import annotations

import pytest

from src.harness.types import ControllerDecision, EvidenceItem, HarnessRequest, HarnessState
from src.harness import verifier as harness_verifier
from src.harness.verifier import verify_draft

pytestmark = pytest.mark.unit


def test_controller_decision_requires_skill_call_for_call_skill() -> None:
    with pytest.raises(ValueError):
        ControllerDecision(action="call_skill", rationale="missing payload")


def test_fallback_verifier_requests_revision_without_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(harness_verifier, "get_llm", lambda _model: (_ for _ in ()).throw(RuntimeError("offline")))
    state = HarnessState(request=HarnessRequest(user_prompt="Analyze AAPL"))

    report = verify_draft(state, "# Draft\n\nNo evidence.")

    assert report.decision == "revise"
    assert report.grounding == "fail"
    assert report.citation_coverage == "fail"


def test_fallback_verifier_passes_grounded_draft_with_citations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(harness_verifier, "get_llm", lambda _model: (_ for _ in ()).throw(RuntimeError("offline")))
    state = HarnessState(
        request=HarnessRequest(user_prompt="Analyze AAPL"),
        evidence_ledger=[
            EvidenceItem(
                id="E1",
                skill_name="search_and_read",
                source_type="url",
                summary="AAPL article",
                sources=["https://example.com/aapl"],
            )
        ],
    )

    report = verify_draft(
        state,
        "# Draft\n\nApple looks resilient based on recent reporting [E1].\n\n## Sources\n- [E1]",
    )

    assert report.decision == "pass"
    assert report.grounding == "pass"
    assert report.citation_coverage == "pass"
