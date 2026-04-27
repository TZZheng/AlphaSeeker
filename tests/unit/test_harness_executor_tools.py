from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.harness.artifacts import (
    agent_workspace_paths,
    create_agent_workspace,
    initialize_run_root,
    read_jsonl,
    registry_paths,
    update_agent_record,
    write_status,
    write_text_atomic,
)
from src.harness.executor import _HANDLERS, create_or_load_session, execute_model_tool
from src.harness.presets import default_tool_allowlist, visible_skills_for_preset
from src.harness.prompt_builder import render_task_markdown, render_tools_markdown
from src.harness.registry import build_skill_registry, get_skills_for_packs
from src.harness.tool_catalog import harness_tool_definitions
from src.harness.types import AGENT_PRESETS, HarnessRequest, SkillMetrics, SkillResult, SkillSpec


def _create_basic_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    run_id: str,
    user_prompt: str,
    preset: str = "orchestrator",
):
    monkeypatch.chdir(tmp_path)
    request = HarnessRequest(user_prompt=user_prompt, run_id=run_id)
    run_root, root_agent_id = initialize_run_root(request)
    registry = build_skill_registry()
    create_agent_workspace(
        run_root,
        agent_id=root_agent_id,
        parent_id="",
        preset=preset,
        task_name="Root Task",
        description=user_prompt,
        task_markdown=render_task_markdown(request.user_prompt),
        tools_markdown=render_tools_markdown(
            preset=preset,
            available_tools=default_tool_allowlist(preset),
            available_skills=[],
        ),
    )
    session = create_or_load_session(
        request=request,
        run_root=str(run_root),
        agent_id=root_agent_id,
        preset=preset,
        registry_map=registry,
    )
    return run_root, root_agent_id, session


def _final_report_snapshot_rows(run_root: Path) -> list[dict[str, object]]:
    return read_jsonl(registry_paths(run_root)["final_report_versions"])


def test_all_executor_handlers_are_exposed_by_a_preset_allowlist() -> None:
    exposed_tools: set[str] = set()
    for preset in AGENT_PRESETS:
        exposed_tools.update(default_tool_allowlist(preset))

    assert set(_HANDLERS) <= exposed_tools


def test_spawn_subagent_rejects_unknown_preset_and_lists_legal_presets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    request = HarnessRequest(user_prompt="Delegate a task", run_id="executor-spawn")
    run_root, root_agent_id = initialize_run_root(request)
    registry = build_skill_registry()
    create_agent_workspace(
        run_root,
        agent_id=root_agent_id,
        parent_id="",
        preset="orchestrator",
        task_name="Root Task",
        description="Delegate work.",
        task_markdown=render_task_markdown(request.user_prompt),
        tools_markdown=render_tools_markdown(
            preset="orchestrator",
            available_tools=default_tool_allowlist("orchestrator"),
            available_skills=[],
        ),
    )
    session = create_or_load_session(
        request=request,
        run_root=str(run_root),
        agent_id=root_agent_id,
        preset="orchestrator",
        registry_map=registry,
    )

    with pytest.raises(ValueError, match="Legal presets: 'orchestrator', 'research', 'source_triage', 'writer', 'synthesizer', 'evaluator'"):
        execute_model_tool(
            session,
            "spawn_subagent",
            {"task_name": "child", "description": "Do work", "preset": "analysis"},
        )


def test_spawn_subagent_can_record_expected_publish_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    request = HarnessRequest(user_prompt="Delegate a task", run_id="executor-spawn-outputs")
    run_root, root_agent_id = initialize_run_root(request)
    registry = build_skill_registry()
    create_agent_workspace(
        run_root,
        agent_id=root_agent_id,
        parent_id="",
        preset="orchestrator",
        task_name="Root Task",
        description="Delegate work.",
        task_markdown=render_task_markdown(request.user_prompt),
        tools_markdown=render_tools_markdown(
            preset="orchestrator",
            available_tools=default_tool_allowlist("orchestrator"),
            available_skills=[],
        ),
    )
    session = create_or_load_session(
        request=request,
        run_root=str(run_root),
        agent_id=root_agent_id,
        preset="orchestrator",
        registry_map=registry,
    )

    result = execute_model_tool(
        session,
        "spawn_subagent",
        {
            "task_name": "child",
            "description": "Do focused work",
            "preset": "research",
            "expected_publish_files": ["publish/financials_valuation.md"],
        },
    )

    child_task = (agent_workspace_paths(run_root, result["agent_id"])["task"]).read_text(encoding="utf-8")

    assert result["expected_publish_files"] == ["publish/financials_valuation.md"]
    assert "## Expected Published Outputs" in child_task
    assert "`publish/financials_valuation.md`" in child_task


def test_skill_results_return_lean_content_and_artifact_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    request = HarnessRequest(user_prompt="Run a deterministic skill", run_id="executor-skill-paths")
    run_root, root_agent_id = initialize_run_root(request)
    registry = build_skill_registry()
    create_agent_workspace(
        run_root,
        agent_id=root_agent_id,
        parent_id="",
        preset="research",
        task_name="Root Task",
        description="Run a skill.",
        task_markdown=render_task_markdown(request.user_prompt),
        tools_markdown=render_tools_markdown(
            preset="research",
            available_tools=default_tool_allowlist("research"),
            available_skills=[],
        ),
    )
    external_artifact = tmp_path / "external_artifact.txt"
    external_artifact.write_text("artifact body\n", encoding="utf-8")

    def _fake_skill(arguments: dict[str, object], _state) -> SkillResult:
        return SkillResult(
            skill_name="fake_skill",
            arguments=dict(arguments),
            status="ok",
            summary="Fake skill completed.",
            details={"kind": "fake"},
            metrics=SkillMetrics(evidence_count=1, artifact_count=1),
            output_text="Hello from fake skill.",
            artifacts=[str(external_artifact)],
        )

    registry["fake_skill"] = SkillSpec(
        name="fake_skill",
        description="Fake skill for testing exact output paths.",
        pack="core",
        input_schema={},
        produces_artifacts=True,
        executor=_fake_skill,
    )

    session = create_or_load_session(
        request=request,
        run_root=str(run_root),
        agent_id=root_agent_id,
        preset="research",
        registry_map=registry,
    )

    result = execute_model_tool(session, "fake_skill", {})
    skill_output_root = next(agent_workspace_paths(run_root, root_agent_id)["skills_artifacts_root"].glob("*_fake_skill"))

    assert result == {
        "skill_name": "fake_skill",
        "status": "ok",
        "summary": "Fake skill completed.",
        "content": "Hello from fake skill.",
        "artifact_paths": [str(external_artifact)],
    }
    assert (skill_output_root / "output.md").read_text(encoding="utf-8") == "Hello from fake skill."
    assert (skill_output_root / "summary.md").read_text(encoding="utf-8") == "Fake skill completed.\n"
    assert (skill_output_root / "details.json").exists()


def test_search_web_result_exposes_compact_urls_without_snippets(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, root_agent_id, session = _create_basic_session(
        monkeypatch,
        tmp_path,
        run_id="executor-search-web",
        user_prompt="Search for XOM macro sources.",
        preset="research",
    )
    search_results = [
        {
            "title": "Fed policy update",
            "href": "https://example.com/fed",
            "date": "2025-01-02",
            "body": "Long preview text that should stay out of compact rows.",
        },
        {
            "title": "Oil outlook",
            "href": "https://example.com/oil",
            "body": "Another preview.",
        },
    ]
    monkeypatch.setattr(
        "src.harness.skills.core.search_web",
        lambda _query, max_results=8: search_results[:max_results],
    )

    result = execute_model_tool(session, "search_web", {"query": "xom macro", "max_results": 2})

    assert result["results"] == [
        {"title": "Fed policy update", "url": "https://example.com/fed", "date": "2025-01-02"},
        {"title": "Oil outlook", "url": "https://example.com/oil", "date": ""},
    ]
    assert all("snippet" not in row for row in result["results"])
    artifact_path = Path(result["results_path"])
    output_path = next(agent_workspace_paths(run_root, root_agent_id)["skills_artifacts_root"].glob("*_search_web/output.md"))
    output_text = output_path.read_text(encoding="utf-8")

    assert artifact_path.exists()
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == search_results
    assert result["artifact_paths"] == [str(artifact_path)]
    assert not {
        "primary_artifact_path",
        "output_files",
        "output_root",
        "summary_path",
        "details_path",
        "evidence_path",
        "artifacts_manifest_path",
        "artifact_count",
        "evidence_count",
        "details",
    } & set(result)
    assert f"Results artifact: {artifact_path}" in output_text
    assert "| 1 | Fed policy update | https://example.com/fed | 2025-01-02 |" in output_text
    assert "| 2 | Oil outlook | https://example.com/oil |  |" in output_text
    assert "Long preview text" not in output_text


def test_publish_tools_normalize_publish_prefix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    request = HarnessRequest(user_prompt="Publish a summary", run_id="executor-publish-prefix")
    run_root, root_agent_id = initialize_run_root(request)
    registry = build_skill_registry()
    create_agent_workspace(
        run_root,
        agent_id=root_agent_id,
        parent_id="",
        preset="orchestrator",
        task_name="Root Task",
        description="Publish work.",
        task_markdown=render_task_markdown(request.user_prompt),
        tools_markdown=render_tools_markdown(
            preset="orchestrator",
            available_tools=default_tool_allowlist("orchestrator"),
            available_skills=[],
        ),
    )
    session = create_or_load_session(
        request=request,
        run_root=str(run_root),
        agent_id=root_agent_id,
        preset="orchestrator",
        registry_map=registry,
    )

    write_result = execute_model_tool(
        session,
        "write_file",
        {"path": "publish/summary.md", "content": "Hello\n"},
    )
    read_result = execute_model_tool(
        session,
        "read_file",
        {"path": write_result["path"]},
    )

    assert write_result["path"].endswith("/publish/summary.md")
    assert "/publish/publish/" not in write_result["path"]
    assert read_result["content"].startswith("Hello")


def test_write_file_schema_requires_path_and_content() -> None:
    schema = harness_tool_definitions()["write_file"]["input_schema"]

    assert schema["required"] == ["path", "content"]


def test_write_file_rejects_missing_publish_content(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, _, session = _create_basic_session(
        monkeypatch,
        tmp_path,
        run_id="executor-write-file-missing-content",
        user_prompt="Publish a final memo",
    )

    with pytest.raises(ValueError, match="write_file requires content"):
        execute_model_tool(
            session,
            "write_file",
            {"path": "publish/final.md"},
        )


def test_write_file_rejects_empty_publish_content(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, _, session = _create_basic_session(
        monkeypatch,
        tmp_path,
        run_id="executor-write-file-empty-publish",
        user_prompt="Publish a final memo",
    )

    with pytest.raises(ValueError, match="non-empty content for publish paths"):
        execute_model_tool(
            session,
            "write_file",
            {"path": "publish/final.md", "content": ""},
        )


def test_write_file_allows_empty_scratch_content(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, root_agent_id, session = _create_basic_session(
        monkeypatch,
        tmp_path,
        run_id="executor-write-file-empty-scratch",
        user_prompt="Clear a scratch note",
    )

    result = execute_model_tool(
        session,
        "write_file",
        {"path": "scratch/notes.md", "content": ""},
    )

    note_path = agent_workspace_paths(run_root, root_agent_id)["scratch_root"] / "notes.md"

    assert result["path"] == str(note_path)
    assert note_path.read_text(encoding="utf-8") == ""


def test_root_write_file_snapshots_final_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, _root_agent_id, session = _create_basic_session(
        monkeypatch,
        tmp_path,
        run_id="executor-final-snapshot-write",
        user_prompt="Publish a final memo",
    )
    content = "# Final\n\nDraft memo.\n"

    write_result = execute_model_tool(
        session,
        "write_file",
        {"path": "publish/final.md", "content": content},
    )

    rows = _final_report_snapshot_rows(run_root)
    assert len(rows) == 1
    row = rows[0]
    assert row["version"] == 1
    assert row["agent_id"] == "agent_root"
    assert row["source_path"] == write_result["path"]
    assert row["sha256"] == hashlib.sha256(content.encode("utf-8")).hexdigest()
    assert row["chars"] == len(content)
    assert row["trigger_tool"] == "write_file"
    assert row["trigger_operation"] == ""
    assert Path(str(row["snapshot_path"])).read_text(encoding="utf-8") == content
    events = read_jsonl(registry_paths(run_root)["events_registry"])
    assert any(event.get("event_type") == "final_report_snapshot_created" for event in events)


def test_identical_root_final_write_does_not_create_duplicate_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, _root_agent_id, session = _create_basic_session(
        monkeypatch,
        tmp_path,
        run_id="executor-final-snapshot-duplicate",
        user_prompt="Publish a final memo",
    )
    content = "# Final\n\nSame memo.\n"

    execute_model_tool(session, "write_file", {"path": "publish/final.md", "content": content})
    execute_model_tool(session, "write_file", {"path": "publish/final.md", "content": content})

    rows = _final_report_snapshot_rows(run_root)
    assert len(rows) == 1
    assert Path(str(rows[0]["snapshot_path"])).name == "v0001.md"


def test_editing_root_final_creates_next_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, _root_agent_id, session = _create_basic_session(
        monkeypatch,
        tmp_path,
        run_id="executor-final-snapshot-edit",
        user_prompt="Revise a final memo",
        preset="writer",
    )

    execute_model_tool(session, "write_file", {"path": "publish/final.md", "content": "# Final\n\nDraft.\n"})
    execute_model_tool(
        session,
        "edit_file",
        {
            "path": "publish/final.md",
            "operation": "replace",
            "target_text": "Draft.",
            "content": "Revised.",
        },
    )

    rows = _final_report_snapshot_rows(run_root)
    assert [row["version"] for row in rows] == [1, 2]
    assert Path(str(rows[1]["snapshot_path"])).name == "v0002.md"
    assert Path(str(rows[1]["snapshot_path"])).read_text(encoding="utf-8") == "# Final\n\nRevised.\n"
    assert rows[1]["trigger_tool"] == "edit_file"
    assert rows[1]["trigger_operation"] == "replace"


def test_updating_publish_summary_does_not_snapshot_final_report(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, _root_agent_id, session = _create_basic_session(
        monkeypatch,
        tmp_path,
        run_id="executor-final-snapshot-summary",
        user_prompt="Publish a summary",
    )

    execute_model_tool(session, "write_file", {"path": "publish/summary.md", "content": "# Summary\n"})

    assert _final_report_snapshot_rows(run_root) == []


def test_child_final_write_does_not_snapshot_report_versions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, _root_agent_id, session = _create_basic_session(
        monkeypatch,
        tmp_path,
        run_id="executor-final-snapshot-child",
        user_prompt="Delegate a child final",
    )
    child = execute_model_tool(
        session,
        "spawn_subagent",
        {
            "task_name": "Child final",
            "description": "Publish child final.",
            "preset": "writer",
        },
    )
    child_session = create_or_load_session(
        request=session.request,
        run_root=str(run_root),
        agent_id=str(child["agent_id"]),
        preset="writer",
        registry_map=session.registry_map,
    )

    execute_model_tool(child_session, "write_file", {"path": "publish/final.md", "content": "# Child\n"})

    assert _final_report_snapshot_rows(run_root) == []


def test_failed_final_edits_do_not_create_new_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, _root_agent_id, session = _create_basic_session(
        monkeypatch,
        tmp_path,
        run_id="executor-final-snapshot-failed-edit",
        user_prompt="Revise a final memo",
        preset="writer",
    )
    execute_model_tool(session, "write_file", {"path": "publish/final.md", "content": "# Final\n\nStable.\n"})

    with pytest.raises(ValueError, match="target_text not found"):
        execute_model_tool(
            session,
            "edit_file",
            {
                "path": "publish/final.md",
                "operation": "replace",
                "target_text": "Missing",
                "content": "Replacement",
            },
        )
    with pytest.raises(ValueError, match="context was not found"):
        execute_model_tool(
            session,
            "apply_patch",
            {
                "patch": "\n".join(
                    [
                        "*** Begin Patch",
                        "*** Update File: publish/final.md",
                        "@@",
                        " Missing context",
                        "-Old",
                        "+New",
                        "*** End Patch",
                    ]
                ),
            },
        )

    rows = _final_report_snapshot_rows(run_root)
    assert len(rows) == 1
    assert Path(str(rows[0]["snapshot_path"])).read_text(encoding="utf-8") == "# Final\n\nStable.\n"


def test_context_files_are_copied_for_read_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    request = HarnessRequest(user_prompt="Delegate with context", run_id="executor-context-copy")
    run_root, root_agent_id = initialize_run_root(request)
    registry = build_skill_registry()
    create_agent_workspace(
        run_root,
        agent_id=root_agent_id,
        parent_id="",
        preset="orchestrator",
        task_name="Root Task",
        description="Delegate with context.",
        task_markdown=render_task_markdown(request.user_prompt),
        tools_markdown=render_tools_markdown(
            preset="orchestrator",
            available_tools=default_tool_allowlist("orchestrator"),
            available_skills=visible_skills_for_preset(
                preset="orchestrator",
                available_skills=get_skills_for_packs(registry, ["core"]),
            ),
        ),
    )
    source_file = tmp_path / "source-note.md"
    source_file.write_text("AlphaSeeker context note\n", encoding="utf-8")
    session = create_or_load_session(
        request=request,
        run_root=str(run_root),
        agent_id=root_agent_id,
        preset="orchestrator",
        registry_map=registry,
    )

    result = execute_model_tool(
        session,
        "spawn_subagent",
        {
            "task_name": "child",
            "description": "Read passed context",
            "preset": "research",
            "context_files": [str(source_file)],
        },
    )

    child_context_root = agent_workspace_paths(run_root, result["agent_id"])["context_root"]
    copied = list(child_context_root.iterdir())

    assert len(copied) == 1
    assert copied[0].read_text(encoding="utf-8") == "AlphaSeeker context note\n"


def test_search_in_files_returns_match_locations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    request = HarnessRequest(user_prompt="Search local files", run_id="executor-search-files")
    run_root, root_agent_id = initialize_run_root(request)
    registry = build_skill_registry()
    create_agent_workspace(
        run_root,
        agent_id=root_agent_id,
        parent_id="",
        preset="research",
        task_name="Root Task",
        description="Search files.",
        task_markdown=render_task_markdown(request.user_prompt),
        tools_markdown=render_tools_markdown(
            preset="research",
            available_tools=default_tool_allowlist("research"),
            available_skills=visible_skills_for_preset(
                preset="research",
                available_skills=get_skills_for_packs(registry, ["core"]),
            ),
        ),
    )
    notes_dir = agent_workspace_paths(run_root, root_agent_id)["scratch_root"] / "notes"
    notes_dir.mkdir()
    hit_file = notes_dir / "memo.md"
    hit_file.write_text("Apple valuation is sensitive to services mix.\n", encoding="utf-8")
    miss_file = notes_dir / "other.md"
    miss_file.write_text("Unrelated line.\n", encoding="utf-8")

    session = create_or_load_session(
        request=request,
        run_root=str(run_root),
        agent_id=root_agent_id,
        preset="research",
        registry_map=registry,
    )

    result = execute_model_tool(
        session,
        "search_in_files",
        {
            "pattern": "services mix",
            "paths": [str(notes_dir)],
            "max_results": 5,
        },
    )

    assert result["status"] == "ok"
    assert result["summary"].startswith("Found 1 match")
    assert "services mix" in result["content"]
    assert str(hit_file) in result["content"]
    assert "output_files" not in result
    assert "details_path" not in result


def test_bash_rg_discovers_matching_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import shutil
    if not shutil.which("rg"):
        pytest.skip("ripgrep not installed")

    monkeypatch.chdir(tmp_path)
    request = HarnessRequest(user_prompt="Find markdown files", run_id="executor-glob-files")
    run_root, root_agent_id = initialize_run_root(request)
    registry = build_skill_registry()
    create_agent_workspace(
        run_root,
        agent_id=root_agent_id,
        parent_id="",
        preset="research",
        task_name="Root Task",
        description="Discover files.",
        task_markdown=render_task_markdown(request.user_prompt),
        tools_markdown=render_tools_markdown(
            preset="research",
            available_tools=default_tool_allowlist("research"),
            available_skills=visible_skills_for_preset(
                preset="research",
                available_skills=get_skills_for_packs(registry, ["core"]),
            ),
        ),
    )
    notes_dir = agent_workspace_paths(run_root, root_agent_id)["scratch_root"] / "notes"
    notes_dir.mkdir()
    (notes_dir / "memo.md").write_text("memo\n", encoding="utf-8")
    (notes_dir / "draft.txt").write_text("draft\n", encoding="utf-8")

    session = create_or_load_session(
        request=request,
        run_root=str(run_root),
        agent_id=root_agent_id,
        preset="research",
        registry_map=registry,
    )

    result = execute_model_tool(
        session,
        "bash",
        {
            "argv": ["rg", "--files", "-g", "*.md", str(notes_dir)],
        },
    )

    assert result["ok"] is True
    assert "memo.md" in result["stdout"]


def test_search_visibility_excludes_private_and_default_artifact_roots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, root_agent_id, session = _create_basic_session(
        monkeypatch,
        tmp_path,
        run_id="executor-search-visibility",
        user_prompt="Search visible files only",
        preset="research",
    )
    paths = agent_workspace_paths(run_root, root_agent_id)
    private_file = paths["harness_logs_root"] / "private.md"
    private_file.write_text("needle-private\n", encoding="utf-8")
    artifact_file = paths["skills_artifacts_root"] / "001_tool" / "output.md"
    artifact_file.parent.mkdir(parents=True, exist_ok=True)
    artifact_file.write_text("needle-artifact\n", encoding="utf-8")

    default_result = execute_model_tool(
        session,
        "search_in_files",
        {"pattern": "needle-artifact", "max_results": 5},
    )
    exact_artifact_result = execute_model_tool(
        session,
        "search_in_files",
        {"pattern": "needle-artifact", "paths": [str(artifact_file)], "max_results": 5},
    )

    read_private_result = execute_model_tool(session, "read_file", {"path": str(private_file)})

    assert default_result["summary"].startswith("Found 0 match")
    assert default_result.get("content", "") == ""
    assert exact_artifact_result["summary"].startswith("Found 1 match")
    assert str(artifact_file) in exact_artifact_result["content"]
    assert read_private_result["status"] == "failed"
    assert "harness-private" in read_private_result["summary"]


def test_bash_ls_and_rg_do_not_reveal_private_or_artifact_roots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, root_agent_id, session = _create_basic_session(
        monkeypatch,
        tmp_path,
        run_id="executor-bash-visibility",
        user_prompt="Inspect visible files",
        preset="research",
    )
    paths = agent_workspace_paths(run_root, root_agent_id)
    (paths["harness_logs_root"] / "private.md").write_text("secret-token\n", encoding="utf-8")
    artifact_file = paths["skills_artifacts_root"] / "001_tool" / "output.md"
    artifact_file.parent.mkdir(parents=True, exist_ok=True)
    artifact_file.write_text("artifact-token\n", encoding="utf-8")

    ls_result = execute_model_tool(session, "bash", {"argv": ["ls"]})
    rg_result = execute_model_tool(session, "bash", {"argv": ["rg", "secret-token|artifact-token"]})

    assert ls_result["ok"] is True
    assert "_harness" not in ls_result["stdout"]
    assert "artifacts" not in ls_result["stdout"]
    assert "scratch/" in ls_result["stdout"]
    assert "_harness" not in rg_result["stdout"]
    assert "secret-token" not in rg_result["stdout"]
    assert "artifact-token" not in rg_result["stdout"]


def test_bash_mutations_reject_private_and_artifact_destinations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, root_agent_id, session = _create_basic_session(
        monkeypatch,
        tmp_path,
        run_id="executor-bash-mutation-visibility",
        user_prompt="Reject hidden mutations",
        preset="research",
    )
    paths = agent_workspace_paths(run_root, root_agent_id)
    write_result = execute_model_tool(session, "write_file", {"path": "scratch/source.md", "content": "source\n"})

    with pytest.raises(ValueError, match="harness-private"):
        execute_model_tool(session, "bash", {"argv": ["mkdir", str(paths["harness_root"] / "new")]})
    with pytest.raises(ValueError, match="publish/ or scratch"):
        execute_model_tool(session, "bash", {"argv": ["mkdir", str(paths["artifacts_root"] / "new")]})
    with pytest.raises(ValueError, match="publish/ or scratch"):
        execute_model_tool(
            session,
            "bash",
            {"argv": ["cp", write_result["path"], str(paths["artifacts_root"] / "copy.md")]},
        )
    with pytest.raises(ValueError, match="harness-private"):
        execute_model_tool(
            session,
            "bash",
            {"argv": ["mv", write_result["path"], str(paths["harness_root"] / "moved.md")]},
        )


def test_bash_copy_and_move_stay_inside_visible_workspace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    request = HarnessRequest(user_prompt="Copy a file", run_id="executor-bash-copy-move")
    run_root, root_agent_id = initialize_run_root(request)
    registry = build_skill_registry()
    create_agent_workspace(
        run_root,
        agent_id=root_agent_id,
        parent_id="",
        preset="research",
        task_name="Root Task",
        description="Copy and move files.",
        task_markdown=render_task_markdown(request.user_prompt),
        tools_markdown=render_tools_markdown(
            preset="research",
            available_tools=default_tool_allowlist("research"),
            available_skills=visible_skills_for_preset(
                preset="research",
                available_skills=get_skills_for_packs(registry, ["core"]),
            ),
        ),
    )
    session = create_or_load_session(
        request=request,
        run_root=str(run_root),
        agent_id=root_agent_id,
        preset="research",
        registry_map=registry,
    )

    written = execute_model_tool(
        session,
        "write_file",
        {"path": "publish/summary.md", "content": "hello\n"},
    )
    copied_path = Path(agent_workspace_paths(run_root, root_agent_id)["scratch_root"] / "summary_copy.md")
    moved_path = Path(agent_workspace_paths(run_root, root_agent_id)["scratch_root"] / "summary_renamed.md")

    copy_result = execute_model_tool(
        session,
        "bash",
        {
            "argv": ["cp", written["path"], str(copied_path)],
        },
    )
    move_result = execute_model_tool(
        session,
        "bash",
        {
            "argv": ["mv", str(copied_path), str(moved_path)],
        },
    )
    read_result = execute_model_tool(
        session,
        "read_file",
        {"path": str(moved_path)},
    )

    assert copy_result["ok"] is True
    assert move_result["ok"] is True
    assert read_result["content"] == "hello\n"


def test_bash_rejects_paths_outside_visible_workspace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    request = HarnessRequest(user_prompt="Reject outside path", run_id="executor-bash-scope")
    run_root, root_agent_id = initialize_run_root(request)
    registry = build_skill_registry()
    create_agent_workspace(
        run_root,
        agent_id=root_agent_id,
        parent_id="",
        preset="research",
        task_name="Root Task",
        description="Reject outside paths.",
        task_markdown=render_task_markdown(request.user_prompt),
        tools_markdown=render_tools_markdown(
            preset="research",
            available_tools=default_tool_allowlist("research"),
            available_skills=visible_skills_for_preset(
                preset="research",
                available_skills=get_skills_for_packs(registry, ["core"]),
            ),
        ),
    )
    session = create_or_load_session(
        request=request,
        run_root=str(run_root),
        agent_id=root_agent_id,
        preset="research",
        registry_map=registry,
    )
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("outside\n", encoding="utf-8")

    with pytest.raises(ValueError, match="agent-visible file surface"):
        execute_model_tool(
            session,
            "bash",
            {
                "argv": ["cp", str(outside), str(agent_workspace_paths(run_root, root_agent_id)["scratch_root"] / "copy.txt")],
            },
        )


def test_bash_sleep_records_standard_bash_event(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, root_agent_id, session = _create_basic_session(
        monkeypatch,
        tmp_path,
        run_id="executor-bash-sleep",
        user_prompt="Sleep briefly",
        preset="research",
    )

    result = execute_model_tool(
        session,
        "bash",
        {"argv": ["sleep", "0"]},
    )
    events = read_jsonl(registry_paths(run_root)["events_registry"])
    bash_events = [event for event in events if event.get("event_type") == "bash_executed"]

    assert result["ok"] is True
    assert result["summary"] == "Slept for 0.0 second(s)."
    assert bash_events
    assert bash_events[-1]["details"] == {
        "argv": ["sleep", "0"],
            "cwd": str(agent_workspace_paths(run_root, root_agent_id)["workspace"]),
        "returncode": 0,
    }


def test_read_file_supports_line_slices(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    request = HarnessRequest(user_prompt="Read part of a file", run_id="executor-read-lines")
    run_root, root_agent_id = initialize_run_root(request)
    registry = build_skill_registry()
    create_agent_workspace(
        run_root,
        agent_id=root_agent_id,
        parent_id="",
        preset="research",
        task_name="Root Task",
        description="Read lines.",
        task_markdown=render_task_markdown(request.user_prompt),
        tools_markdown=render_tools_markdown(
            preset="research",
            available_tools=default_tool_allowlist("research"),
            available_skills=visible_skills_for_preset(
                preset="research",
                available_skills=get_skills_for_packs(registry, ["core"]),
            ),
        ),
    )
    note = agent_workspace_paths(run_root, root_agent_id)["scratch_root"] / "note.md"
    note.write_text("line1\nline2\nline3\nline4\n", encoding="utf-8")

    session = create_or_load_session(
        request=request,
        run_root=str(run_root),
        agent_id=root_agent_id,
        preset="research",
        registry_map=registry,
    )

    result = execute_model_tool(
        session,
        "read_file",
        {
            "path": str(note),
            "start_line": 2,
            "max_lines": 2,
        },
    )

    assert result["status"] == "truncated"
    assert result["content"] == "line2\nline3\n"


def test_edit_file_replaces_anchored_text_in_publish(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    request = HarnessRequest(user_prompt="Edit publish output", run_id="executor-edit-publish")
    run_root, root_agent_id = initialize_run_root(request)
    registry = build_skill_registry()
    create_agent_workspace(
        run_root,
        agent_id=root_agent_id,
        parent_id="",
        preset="writer",
        task_name="Root Task",
        description="Edit publish output.",
        task_markdown=render_task_markdown(request.user_prompt),
        tools_markdown=render_tools_markdown(
            preset="writer",
            available_tools=default_tool_allowlist("writer"),
            available_skills=visible_skills_for_preset(
                preset="writer",
                available_skills=get_skills_for_packs(registry, ["core"]),
            ),
        ),
    )
    session = create_or_load_session(
        request=request,
        run_root=str(run_root),
        agent_id=root_agent_id,
        preset="writer",
        registry_map=registry,
    )

    write_result = execute_model_tool(
        session,
        "write_file",
        {"path": "publish/summary.md", "content": "alpha\nbeta\ngamma\n"},
    )
    edit_result = execute_model_tool(
        session,
        "edit_file",
        {
            "path": "publish/summary.md",
            "operation": "replace",
            "target_text": "beta",
            "content": "BETA",
        },
    )
    read_result = execute_model_tool(
        session,
        "read_file",
        {"path": write_result["path"]},
    )

    assert edit_result["match_count"] == 1
    assert read_result["content"] == "alpha\nBETA\ngamma\n"


def test_edit_file_inserts_into_scratch_without_full_rewrite(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    request = HarnessRequest(user_prompt="Edit scratch note", run_id="executor-edit-scratch")
    run_root, root_agent_id = initialize_run_root(request)
    registry = build_skill_registry()
    create_agent_workspace(
        run_root,
        agent_id=root_agent_id,
        parent_id="",
        preset="research",
        task_name="Root Task",
        description="Edit scratch note.",
        task_markdown=render_task_markdown(request.user_prompt),
        tools_markdown=render_tools_markdown(
            preset="research",
            available_tools=default_tool_allowlist("research"),
            available_skills=visible_skills_for_preset(
                preset="research",
                available_skills=get_skills_for_packs(registry, ["core"]),
            ),
        ),
    )
    session = create_or_load_session(
        request=request,
        run_root=str(run_root),
        agent_id=root_agent_id,
        preset="research",
        registry_map=registry,
    )

    write_result = execute_model_tool(
        session,
        "write_file",
        {"path": "scratch/notes.md", "content": "top\nbottom\n"},
    )
    execute_model_tool(
        session,
        "edit_file",
        {
            "path": "scratch/notes.md",
            "operation": "insert_before",
            "target_text": "bottom",
            "content": "middle\n",
        },
    )
    read_result = execute_model_tool(
        session,
        "read_file",
        {"path": write_result["path"]},
    )

    assert read_result["content"] == "top\nmiddle\nbottom\n"


def test_apply_patch_replaces_multiline_paragraph_in_publish(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, root_agent_id, session = _create_basic_session(
        monkeypatch,
        tmp_path,
        run_id="executor-apply-patch-paragraph",
        user_prompt="Patch publish output",
        preset="writer",
    )
    write_result = execute_model_tool(
        session,
        "write_file",
        {
            "path": "publish/summary.md",
            "content": "Intro\nThe old thesis line 1.\nThe old thesis line 2.\nOutro\n",
        },
    )

    patch_result = execute_model_tool(
        session,
        "apply_patch",
        {
            "patch": "\n".join(
                [
                    "*** Begin Patch",
                    "*** Update File: publish/summary.md",
                    "@@",
                    " Intro",
                    "-The old thesis line 1.",
                    "-The old thesis line 2.",
                    "+The new thesis line 1.",
                    "+The new thesis line 2.",
                    " Outro",
                    "*** End Patch",
                ]
            ),
        },
    )
    read_result = execute_model_tool(session, "read_file", {"path": write_result["path"]})

    assert patch_result["operation"] == "patch"
    assert patch_result["hunks_applied"] == 1
    assert read_result["content"] == "Intro\nThe new thesis line 1.\nThe new thesis line 2.\nOutro\n"
    assert patch_result["description"] == "Intro"


def test_apply_patch_supports_two_hunks_in_one_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, root_agent_id, session = _create_basic_session(
        monkeypatch,
        tmp_path,
        run_id="executor-apply-patch-two-hunks",
        user_prompt="Patch two sections",
        preset="research",
    )
    write_result = execute_model_tool(
        session,
        "write_file",
        {
            "path": "scratch/notes.md",
            "content": "alpha\nbeta\ngamma\ndelta\nepsilon\n",
        },
    )

    patch_result = execute_model_tool(
        session,
        "apply_patch",
        {
            "patch": "\n".join(
                [
                    "*** Begin Patch",
                    "*** Update File: scratch/notes.md",
                    "@@",
                    " alpha",
                    "-beta",
                    "+BETA",
                    " gamma",
                    "@@",
                    " delta",
                    "-epsilon",
                    "+EPSILON",
                    "*** End Patch",
                ]
            ),
        },
    )
    read_result = execute_model_tool(session, "read_file", {"path": write_result["path"]})

    assert patch_result["hunks_applied"] == 2
    assert read_result["content"] == "alpha\nBETA\ngamma\ndelta\nEPSILON\n"


def test_apply_patch_rejects_ambiguous_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, root_agent_id, session = _create_basic_session(
        monkeypatch,
        tmp_path,
        run_id="executor-apply-patch-ambiguous",
        user_prompt="Patch ambiguous note",
        preset="research",
    )
    execute_model_tool(
        session,
        "write_file",
        {
            "path": "scratch/notes.md",
            "content": "start\nshared\nold\nend\nstart\nshared\nold\nend\n",
        },
    )

    with pytest.raises(ValueError) as exc_info:
        execute_model_tool(
            session,
            "apply_patch",
            {
                "patch": "\n".join(
                    [
                        "*** Begin Patch",
                        "*** Update File: scratch/notes.md",
                        "@@",
                        " start",
                        " shared",
                        "-old",
                        "+new",
                        " end",
                        "*** End Patch",
                    ]
                ),
            },
        )
    message = str(exc_info.value)
    assert "matched multiple locations" in message
    assert "read_file(path='scratch/notes.md')" in message
    assert "more specific surrounding lines" in message


def test_apply_patch_rejects_missing_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, root_agent_id, session = _create_basic_session(
        monkeypatch,
        tmp_path,
        run_id="executor-apply-patch-missing",
        user_prompt="Patch missing note",
        preset="research",
    )
    execute_model_tool(
        session,
        "write_file",
        {"path": "scratch/notes.md", "content": "alpha\nbeta\ngamma\n"},
    )

    with pytest.raises(ValueError) as exc_info:
        execute_model_tool(
            session,
            "apply_patch",
            {
                "patch": "\n".join(
                    [
                        "*** Begin Patch",
                        "*** Update File: scratch/notes.md",
                        "@@",
                        " alpha",
                        "-missing",
                        "+replacement",
                        " gamma",
                        "*** End Patch",
                    ]
                ),
            },
        )
    message = str(exc_info.value)
    assert "context was not found" in message
    assert "read_file(path='scratch/notes.md')" in message
    assert "exact current lines" in message


def test_apply_patch_is_atomic_when_a_later_hunk_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, root_agent_id, session = _create_basic_session(
        monkeypatch,
        tmp_path,
        run_id="executor-apply-patch-atomic",
        user_prompt="Patch atomically",
        preset="research",
    )
    write_result = execute_model_tool(
        session,
        "write_file",
        {"path": "scratch/notes.md", "content": "alpha\nbeta\ngamma\nomega\n"},
    )

    with pytest.raises(ValueError, match="context was not found"):
        execute_model_tool(
            session,
            "apply_patch",
            {
                "patch": "\n".join(
                    [
                        "*** Begin Patch",
                        "*** Update File: scratch/notes.md",
                        "@@",
                        " alpha",
                        "-beta",
                        "+BETA",
                        " gamma",
                        "@@",
                        " missing",
                        "+line",
                        "*** End Patch",
                    ]
                ),
            },
        )

    read_result = execute_model_tool(session, "read_file", {"path": write_result["path"]})
    assert read_result["content"] == "alpha\nbeta\ngamma\nomega\n"
