from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.harness.artifacts import (
    agent_workspace_paths,
    create_agent_workspace,
    initialize_run_root,
    snapshot_final_report_if_changed,
    write_text_atomic,
)
from src.harness.evaluator import (
    EvidenceContext,
    ReportVersion,
    _markdown_section_index,
    default_eval_contract,
    evaluate_harness_run,
    score_report_version,
)
from src.harness.presets import visible_skills_for_preset
from src.harness.registry import build_skill_registry, get_skills_for_packs
from src.harness.types import HarnessRequest

pytestmark = pytest.mark.unit


def _make_root_run(tmp_path: Path) -> Path:
    request = HarnessRequest(
        user_prompt=(
            "Write an investment memo on XOM covering valuation, crude oil, "
            "macro backdrop, bull case, bear case, and 12-month risk/reward."
        ),
        run_id="eval-fixture",
    )
    run_root, root_agent_id = initialize_run_root(request)
    create_agent_workspace(
        run_root,
        agent_id=root_agent_id,
        parent_id="",
        preset="orchestrator",
        task_name="Root Task",
        description=request.user_prompt,
        task_markdown=request.user_prompt,
        tools_markdown="tools",
    )
    paths = agent_workspace_paths(run_root, root_agent_id)
    write_text_atomic(paths["publish_summary"], "# Summary\n\nRoot summary.\n")
    write_text_atomic(paths["publish_index"], "- final.md: Final report\n- scratch/evidence.md: Evidence\n")
    write_text_atomic(paths["scratch_root"] / "evidence.md", "XOM evidence: price and WTI as of test date.\n")

    write_text_atomic(paths["publish_final"], "# Final\n\nXOM is a hold. Valuation evidence is thin.\n")
    snapshot_final_report_if_changed(run_root, root_agent_id, trigger={"tool": "write"})
    write_text_atomic(
        paths["publish_final"],
        "# Final\n\nXOM is a buy. It cites WTI but removes valuation support.\n",
    )
    snapshot_final_report_if_changed(run_root, root_agent_id, trigger={"tool": "write"})
    return run_root


def _judge_stub(_system_prompt: str, user_prompt: str) -> str:
    if "Generate an evaluation contract" in user_prompt:
        return json.dumps(
            {
                "prompt_specific_requirements": ["Cover XOM valuation and 12-month risk/reward."],
                "freshness_requirements": ["Use current market data as-of dates."],
                "critical_fail_rules": ["A contradicted key number caps the score below 6."],
            }
        )
    if "Evaluate report version 1" in user_prompt:
        return json.dumps(
            {
                "overall_score": 7.4,
                "area_scores": {
                    "factual_correctness": {"score": 7.5, "rationale": "No contradiction found.", "critical_issues": []},
                    "freshness": {"score": 7.0, "rationale": "As-of dates are light.", "critical_issues": []},
                    "evidence_grounding": {"score": 7.2, "rationale": "Some run evidence is cited.", "critical_issues": []},
                    "logical_soundness": {"score": 7.0, "rationale": "Hold view is plausible.", "critical_issues": []},
                    "completeness": {"score": 7.8, "rationale": "Covers most requirements.", "critical_issues": []},
                    "numerical_discipline": {"score": 7.0, "rationale": "Few numbers.", "critical_issues": []},
                    "decision_usefulness": {"score": 7.3, "rationale": "Clear view.", "critical_issues": []},
                },
                "claim_checks": [
                    {
                        "claim": "XOM is a hold",
                        "category": "qualitative_judgment",
                        "status": "partially_supported",
                        "importance": "high",
                        "evidence": "Report and evidence context",
                        "notes": "",
                    }
                ],
                "major_conclusions": [],
                "critical_fail_flags": [],
                "decision_usefulness_notes": ["Decision view is explicit."],
                "summary": "Acceptable first version.",
            }
        )
    if "Evaluate report version 2" in user_prompt:
        return json.dumps(
            {
                "overall_score": 7.1,
                "area_scores": {
                    "factual_correctness": {"score": 6.8, "rationale": "No contradiction found.", "critical_issues": []},
                    "freshness": {"score": 6.8, "rationale": "As-of dates are still light.", "critical_issues": []},
                    "evidence_grounding": {"score": 5.8, "rationale": "Valuation support was removed.", "critical_issues": ["Missing valuation evidence."]},
                    "logical_soundness": {"score": 6.2, "rationale": "Buy view lacks a bridge.", "critical_issues": []},
                    "completeness": {"score": 7.0, "rationale": "Covers requested topics.", "critical_issues": []},
                    "numerical_discipline": {"score": 6.5, "rationale": "Few numbers.", "critical_issues": []},
                    "decision_usefulness": {"score": 7.0, "rationale": "Clear but less supported.", "critical_issues": []},
                },
                "claim_checks": [
                    {
                        "claim": "XOM is a buy",
                        "category": "investment_conclusion",
                        "status": "unsupported",
                        "importance": "critical",
                        "evidence": "No valuation bridge",
                        "notes": "",
                    }
                ],
                "major_conclusions": [],
                "critical_fail_flags": [],
                "decision_usefulness_notes": ["Decision view is explicit but weakly supported."],
                "summary": "Second version regresses on grounding.",
            }
        )
    if "Compare adjacent report versions" in user_prompt:
        return json.dumps(
            {
                "winner": "older",
                "margin": 1.0,
                "regressions": [
                    {
                        "area": "evidence_grounding",
                        "severity": "material",
                        "reason": "Newer version removes valuation support.",
                    }
                ],
                "improvements": [],
                "introduced_errors": [],
                "removed_evidence": ["Valuation support"],
                "summary": "The newer version is weaker.",
            }
        )
    raise AssertionError(f"Unexpected judge prompt: {user_prompt[:120]}")


def test_evaluate_harness_run_writes_version_and_trajectory_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    run_root = _make_root_run(tmp_path)

    result = evaluate_harness_run(
        run_root,
        case_id="xom_eval",
        eval_id="eval_test",
        judge_fn=_judge_stub,
        evaluator_model="stub-model",
    )

    output_root = Path(result.output_root)
    assert (output_root / "eval_contract.json").exists()
    assert (output_root / "evidence_context.md").exists()
    assert (output_root / "versions" / "v0001.md").exists()
    assert (output_root / "versions" / "v0001.eval.json").exists()
    assert (output_root / "versions" / "v0002.eval.json").exists()
    assert (output_root / "pairwise_v0001_v0002.json").exists()
    assert (output_root / "trajectory.json").exists()

    trajectory = json.loads((output_root / "trajectory.json").read_text(encoding="utf-8"))
    assert trajectory["versions_evaluated"] == 2
    assert trajectory["best_version"] == 1
    assert trajectory["final_version"] == 2
    assert trajectory["regressions"] == 1
    assert trajectory["pairwise_win_rate_vs_previous"] == 0.0
    assert result.pairwise_evaluations[0].is_regression is True


def test_evaluate_harness_run_uses_current_final_when_manifest_is_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    request = HarnessRequest(user_prompt="Analyze XOM valuation.", run_id="manifest-empty")
    run_root, root_agent_id = initialize_run_root(request)
    create_agent_workspace(
        run_root,
        agent_id=root_agent_id,
        parent_id="",
        preset="orchestrator",
        task_name="Root Task",
        description=request.user_prompt,
        task_markdown=request.user_prompt,
        tools_markdown="tools",
    )
    paths = agent_workspace_paths(run_root, root_agent_id)
    write_text_atomic(paths["publish_final"], "# Final\n\nXOM valuation draft.\n")

    result = evaluate_harness_run(
        run_root,
        case_id="fallback",
        eval_id="fallback_eval",
        judge_fn=_judge_stub,
        evaluator_model="stub-model",
    )

    assert len(result.versions) == 1
    assert result.versions[0].trigger_operation == "fallback_snapshot"
    assert Path(result.versions[0].eval_snapshot_path).exists()


def test_score_report_version_normalizes_fractional_judge_scores(tmp_path: Path) -> None:
    report_path = tmp_path / "v0001.md"
    write_text_atomic(report_path, "# Final\n\nXOM is a hold with cited evidence.\n")
    contract = default_eval_contract("fractional", "Analyze XOM valuation.", model="stub-model")

    def judge(_system_prompt: str, _user_prompt: str) -> str:
        area_scores = {
            area.name: {"score": 0.85, "rationale": "Good.", "critical_issues": []}
            for area in contract.areas
        }
        return json.dumps(
            {
                "overall_score": 0.85,
                "area_scores": area_scores,
                "claim_checks": [],
                "major_conclusions": [
                    {
                        "conclusion": "XOM is a hold.",
                        "supporting_claims": ["Valuation evidence is cited."],
                        "counterarguments_considered": [],
                        "missing_links": [],
                        "logic_score": 0.9,
                    }
                ],
                "critical_fail_flags": [],
                "decision_usefulness_notes": [],
                "summary": "Fractional scores should be treated as 0-1 scale.",
            }
        )

    result = score_report_version(
        case_id="fractional",
        eval_id="fractional_eval",
        prompt="Analyze XOM valuation.",
        contract=contract,
        version=ReportVersion(
            version=1,
            source_snapshot_path=str(report_path),
            eval_snapshot_path=str(report_path),
            sha256="",
            chars=report_path.stat().st_size,
            created_at="2026-04-30T00:00:00Z",
        ),
        evidence_context=EvidenceContext(
            run_root=str(tmp_path),
            rendered="Evidence: XOM valuation support.",
            included_files=[],
            omitted_file_count=0,
            chars=32,
        ),
        judge_fn=judge,
        model_name="stub-model",
    )

    assert result.raw_model_score == 8.5
    assert result.weighted_score == 8.5
    assert result.overall_score == 8.5
    assert result.major_conclusions[0].logic_score == 9.0


def test_markdown_section_index_tracks_nested_heading_ranges(tmp_path: Path) -> None:
    report_path = tmp_path / "report.md"
    write_text_atomic(
        report_path,
        "# Executive Summary\n"
        "summary\n"
        "## Valuation\n"
        "valuation\n"
        "### Multiples\n"
        "multiples\n"
        "## Risk/Reward\n"
        "risk\n",
    )

    sections = _markdown_section_index(report_path)

    assert sections[0] == {"level": 1, "title": "Executive Summary", "start_line": 1, "end_line": 8}
    assert sections[1] == {"level": 2, "title": "Valuation", "start_line": 3, "end_line": 6}
    assert sections[2] == {"level": 3, "title": "Multiples", "start_line": 5, "end_line": 6}
    assert sections[3] == {"level": 2, "title": "Risk/Reward", "start_line": 7, "end_line": 8}


def test_evaluator_preset_sees_full_enabled_skill_set() -> None:
    registry = build_skill_registry()
    skills = get_skills_for_packs(registry, ["core", "equity", "macro", "commodity"])
    visible_names = {spec.name for spec in visible_skills_for_preset(preset="evaluator", available_skills=skills)}

    assert "read" in visible_names
    assert "grep" in visible_names
    assert "get_current_datetime" in visible_names
    assert "search_web" in visible_names
    assert "search_news" in visible_names
    assert "read_web_pages" in visible_names
    assert "condense_context" in visible_names
    assert "fetch_company_profile" in visible_names
    assert "fetch_financials" in visible_names
    assert "fetch_market_data" in visible_names
    assert "fetch_macro_indicators" in visible_names
    assert "fetch_eia_inventory" in visible_names
