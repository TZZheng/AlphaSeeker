from __future__ import annotations

from pathlib import Path

import pytest

from src.harness.artifacts import (
    agent_workspace_paths,
    create_agent_workspace,
    initialize_run_root,
    update_agent_record,
    write_status,
    write_text_atomic,
)
from src.harness import executor as executor_module
from src.harness.executor import create_or_load_session, execute_model_tool
from src.harness.presets import default_tool_allowlist, render_root_task_markdown, render_tools_markdown, visible_skills_for_preset
from src.harness.registry import build_skill_registry, get_skills_for_packs
from src.harness.types import HarnessRequest, SkillMetrics, SkillResult, SkillSpec


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
        task_markdown=render_root_task_markdown(request.user_prompt),
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
        task_markdown=render_root_task_markdown(request.user_prompt),
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


def test_skill_results_return_exact_artifact_and_output_paths(
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
        task_markdown=render_root_task_markdown(request.user_prompt),
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

    assert result["primary_artifact_path"] == str(external_artifact)
    assert result["artifact_paths"] == [str(external_artifact)]
    assert result["summary_path"].endswith("summary.md")
    assert result["details_path"].endswith("details.json")
    assert any(path.endswith("output.md") for path in result["output_files"])
    assert any(path.endswith("summary.md") for path in result["output_files"])
    assert any(path.endswith("details.json") for path in result["output_files"])


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
        task_markdown=render_root_task_markdown(request.user_prompt),
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
        task_markdown=render_root_task_markdown(request.user_prompt),
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
        task_markdown=render_root_task_markdown(request.user_prompt),
        tools_markdown=render_tools_markdown(
            preset="research",
            available_tools=default_tool_allowlist("research"),
            available_skills=visible_skills_for_preset(
                preset="research",
                available_skills=get_skills_for_packs(registry, ["core"]),
            ),
        ),
    )
    notes_dir = tmp_path / "notes"
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
    assert result["details_path"].endswith("details.json")
    assert result["output_root"].endswith("search_in_files")
    assert any(path.endswith("output.md") for path in result["output_files"])


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
        task_markdown=render_root_task_markdown(request.user_prompt),
        tools_markdown=render_tools_markdown(
            preset="research",
            available_tools=default_tool_allowlist("research"),
            available_skills=visible_skills_for_preset(
                preset="research",
                available_skills=get_skills_for_packs(registry, ["core"]),
            ),
        ),
    )
    notes_dir = tmp_path / "notes"
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


def test_bash_copy_and_move_stay_inside_project_root(
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
        task_markdown=render_root_task_markdown(request.user_prompt),
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


def test_bash_rejects_paths_outside_project_root(
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
        task_markdown=render_root_task_markdown(request.user_prompt),
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

    with pytest.raises(ValueError, match="escapes the project root"):
        execute_model_tool(
            session,
            "bash",
            {
                "argv": ["cp", str(outside), str(agent_workspace_paths(run_root, root_agent_id)["scratch_root"] / "copy.txt")],
            },
        )


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
        task_markdown=render_root_task_markdown(request.user_prompt),
        tools_markdown=render_tools_markdown(
            preset="research",
            available_tools=default_tool_allowlist("research"),
            available_skills=visible_skills_for_preset(
                preset="research",
                available_skills=get_skills_for_packs(registry, ["core"]),
            ),
        ),
    )
    note = tmp_path / "note.md"
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
        task_markdown=render_root_task_markdown(request.user_prompt),
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
        task_markdown=render_root_task_markdown(request.user_prompt),
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
