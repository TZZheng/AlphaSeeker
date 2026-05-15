from __future__ import annotations

import json
from pathlib import Path

from src.research_platform.memo_artifacts import MemoArtifactPaths, MemoArtifactSet
from src.research_platform.memo_evaluator import (
    MemoEvaluatorVerdict,
    build_evaluator_inputs,
    run_and_write_memo_evaluation,
    run_memo_evaluator,
)


def _verdict(status: str = "pass") -> dict[str, object]:
    return {
        "status": status,
        "rationale": "The memo is acceptable for the mocked test.",
        "blockers": [] if status != "fail" else ["unsupported conclusion"],
        "warnings": [] if status == "pass" else ["minor caveat"],
        "source_review": "Source usage reviewed.",
        "logic_review": "Logic reviewed.",
        "trace_review": "Trace reviewed.",
        "link_review": "Links reviewed.",
        "state_linkage_review": None,
        "inputs_received": ["final_md", "source_index_md", "source_bodies", "source_use_table_md"],
        "inputs_missing": [],
    }


def _artifact_set(final_path: Path | None, source_use_table: Path | None, tool_log: Path | None = None) -> MemoArtifactSet:
    return MemoArtifactSet(
        paths=MemoArtifactPaths(
            product_final_path=str(final_path) if final_path else None,
            product_source_use_table_path=str(source_use_table) if source_use_table else None,
            harness_tool_calls_path=str(tool_log) if tool_log else None,
        ),
        has_usable_final=final_path is not None,
        has_required_artifacts=final_path is not None and source_use_table is not None,
        missing_required=[],
        optional_present=[],
        reasons=[],
    )


def test_build_evaluator_inputs_includes_sources_tool_log_and_present_optional_traces(tmp_path):
    final_path = tmp_path / "final.md"
    final_path.write_text("# Final\n" + ("memo body " * 100), encoding="utf-8")
    source_use = tmp_path / "source_use_table.md"
    source_use.write_text("| Claim | Source |\n| --- | --- |\n| A | S1 |\n", encoding="utf-8")
    source_index = tmp_path / "source_index.md"
    source_index.write_text("# Source Index\nS1: Annual filing", encoding="utf-8")
    source_body = tmp_path / "S1.md"
    source_body.write_text("annual filing body", encoding="utf-8")
    tool_log = tmp_path / "tool_calls.jsonl"
    tool_log.write_text('{"tool":"sec"}\n', encoding="utf-8")
    optional = tmp_path / "execution_plan.md"
    optional.write_text("plan", encoding="utf-8")

    inputs = build_evaluator_inputs(
        run_id="run-1",
        ticker="XOM",
        memo_dir=tmp_path,
        source_index_path=source_index,
        source_body_paths={"S1": source_body},
        artifacts=_artifact_set(final_path, source_use, tool_log),
        optional_trace_paths={"execution_plan.md": optional, "missing.md": tmp_path / "missing.md"},
    )

    assert inputs.final_md.startswith("# Final")
    assert inputs.source_index_md.startswith("# Source Index")
    assert inputs.source_bodies == {"S1": "annual filing body"}
    assert inputs.harness_tool_calls_jsonl == '{"tool":"sec"}\n'
    assert inputs.optional_traces == {"execution_plan.md": "plan"}
    assert "missing.md" not in inputs.optional_traces
    assert "harness_tool_calls_jsonl" in inputs.inputs_received


def test_build_evaluator_inputs_truncates_oversized_source_bodies_with_warning(tmp_path):
    final_path = tmp_path / "final.md"
    final_path.write_text("# Final\n" + ("memo body " * 100), encoding="utf-8")
    source_use = tmp_path / "source_use_table.md"
    source_use.write_text("table", encoding="utf-8")
    source_index = tmp_path / "source_index.md"
    source_index.write_text("index", encoding="utf-8")
    body = tmp_path / "S1.md"
    body.write_text("x" * 200, encoding="utf-8")

    inputs = build_evaluator_inputs(
        run_id="run-1",
        ticker="XOM",
        memo_dir=tmp_path,
        source_index_path=source_index,
        source_body_paths={"S1": body},
        artifacts=_artifact_set(final_path, source_use),
        max_source_body_chars=50,
        max_total_source_body_chars=80,
    )

    assert len(inputs.source_bodies["S1"]) < 100
    assert inputs.truncation_warnings


def test_evaluator_pass_with_missing_critical_input_downgrades_to_warn(tmp_path):
    inputs = build_evaluator_inputs(
        run_id="run-1",
        ticker="XOM",
        memo_dir=tmp_path,
        source_index_path=tmp_path / "missing-source-index.md",
        source_body_paths={},
        artifacts=_artifact_set(None, None),
    )

    verdict = run_memo_evaluator(inputs, judge_fn=lambda _system, _user: json.dumps(_verdict("pass")), model_name="judge-model")

    assert verdict.status == "warn"
    assert any("missing critical evaluator input" in warning for warning in verdict.warnings)


def test_missing_only_tool_log_does_not_auto_downgrade_pass(tmp_path):
    final_path = tmp_path / "final.md"
    final_path.write_text("# Final\n" + ("memo body " * 100), encoding="utf-8")
    source_use = tmp_path / "source_use_table.md"
    source_use.write_text("table", encoding="utf-8")
    source_index = tmp_path / "source_index.md"
    source_index.write_text("index", encoding="utf-8")
    body = tmp_path / "S1.md"
    body.write_text("source body", encoding="utf-8")
    inputs = build_evaluator_inputs(
        run_id="run-1",
        ticker="XOM",
        memo_dir=tmp_path,
        source_index_path=source_index,
        source_body_paths={"S1": body},
        artifacts=_artifact_set(final_path, source_use, None),
    )

    verdict = run_memo_evaluator(inputs, judge_fn=lambda _system, _user: json.dumps(_verdict("pass")), model_name="judge-model")

    assert "harness_tool_calls_jsonl" in inputs.inputs_missing
    assert verdict.status == "pass"


def test_evaluator_repairs_one_malformed_json_response(tmp_path):
    final_path = tmp_path / "final.md"
    final_path.write_text("# Final\n" + ("memo body " * 100), encoding="utf-8")
    source_use = tmp_path / "source_use_table.md"
    source_use.write_text("table", encoding="utf-8")
    source_index = tmp_path / "source_index.md"
    source_index.write_text("index", encoding="utf-8")
    body = tmp_path / "S1.md"
    body.write_text("source body", encoding="utf-8")
    inputs = build_evaluator_inputs(
        run_id="run-1",
        ticker="XOM",
        memo_dir=tmp_path,
        source_index_path=source_index,
        source_body_paths={"S1": body},
        artifacts=_artifact_set(final_path, source_use),
    )
    calls = {"n": 0}

    def judge(_system: str, _user: str) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            return "not json"
        return json.dumps(_verdict("pass"))

    verdict = run_memo_evaluator(inputs, judge_fn=judge, model_name="judge-model")

    assert calls["n"] == 2
    assert isinstance(verdict, MemoEvaluatorVerdict)
    assert verdict.status == "pass"


def test_run_and_write_maps_unparseable_evaluator_to_ref_without_success(tmp_path):
    final_path = tmp_path / "final.md"
    final_path.write_text("# Final\n" + ("memo body " * 100), encoding="utf-8")
    source_use = tmp_path / "source_use_table.md"
    source_use.write_text("table", encoding="utf-8")
    source_index = tmp_path / "source_index.md"
    source_index.write_text("index", encoding="utf-8")
    inputs = build_evaluator_inputs(
        run_id="run-1",
        ticker="XOM",
        memo_dir=tmp_path,
        source_index_path=source_index,
        source_body_paths={},
        artifacts=_artifact_set(final_path, source_use),
    )

    ref = run_and_write_memo_evaluation(inputs, memo_dir=tmp_path, judge_fn=lambda _system, _user: "nope", model_name="judge-model")

    assert ref.status == "unparseable"
    assert ref.verdict is None
    assert Path(ref.path).exists()
