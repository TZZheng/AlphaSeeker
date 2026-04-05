from __future__ import annotations

from pathlib import Path

import pytest

from src.harness.retrieval import refresh_reduction_state
from src.harness.types import HarnessRequest, HarnessState, SourceCard


def test_refresh_reduction_state_uses_required_sections_and_writes_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    workspace = tmp_path / "data" / "harness_runs" / "reduction" / "agents" / "agent_root"
    workspace.mkdir(parents=True, exist_ok=True)

    state = HarnessState(
        request=HarnessRequest(user_prompt="Analyze AAPL"),
        run_id="reduction",
        run_root=str(tmp_path / "data" / "harness_runs" / "reduction"),
        agent_id="agent_root",
        workspace_path=str(workspace),
        enabled_packs=["core", "equity"],
        required_sections=["Executive Summary", "Valuation and Scenarios"],
        source_cards=[
            SourceCard(
                source_id="SRC1",
                title="Apple update",
                canonical_url="https://example.com/aapl",
                domain="example.com",
                summary="Apple margins expanded and valuation remains debated.",
                extracted_facts=["Apple margins expanded in the latest quarter."],
                section_relevance=["Executive Summary", "Valuation and Scenarios"],
                supporting_evidence=["Apple margins expanded in the latest quarter."],
                evidence_ids=["E1"],
            )
        ],
    )

    refresh_reduction_state(state)

    assert state.fact_index
    assert state.section_briefs
    assert state.coverage_matrix is not None
    assert Path(state.dossier_paths["fact_index"]).exists()
    assert Path(state.dossier_paths["coverage_matrix"]).exists()
