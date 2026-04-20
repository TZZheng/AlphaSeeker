from __future__ import annotations

from pathlib import Path

import pytest

from src.harness.agent_worker import _publish_outputs_satisfy_completion
from src.harness.artifacts import agent_workspace_paths, create_agent_workspace, initialize_run_root, write_text_atomic
from src.harness import prompt_builder as prompt_builder_module
from src.harness.presets import default_tool_allowlist, visible_skills_for_preset
from src.harness.prompt_builder import build_agent_prompt_bundle, render_task_markdown, render_tools_markdown
from src.harness.prompt_builder import build_commenter_prompt_bundle
from src.harness.registry import build_skill_registry, get_skills_for_packs
from src.harness.types import HarnessRequest


def _create_workspace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    preset: str = "orchestrator",
    user_prompt: str = "Analyze AAPL",
) -> tuple[Path, str, HarnessRequest]:
    monkeypatch.chdir(tmp_path)
    request = HarnessRequest(user_prompt=user_prompt, run_id="prompt-bundle-test")
    run_root, agent_id = initialize_run_root(request)
    registry = build_skill_registry()
    create_agent_workspace(
        run_root,
        agent_id=agent_id,
        parent_id="",
        preset=preset,
        task_name="Root Task",
        description=user_prompt,
        task_markdown=render_task_markdown(request.user_prompt),
        tools_markdown=render_tools_markdown(
            preset=preset,
            available_tools=default_tool_allowlist(preset),
            available_skills=visible_skills_for_preset(
                preset=preset,
                available_skills=get_skills_for_packs(registry, ["core", "equity"]),
            ),
        ),
    )
    return Path(run_root), agent_id, request


def _build_bundle(
    request: HarnessRequest,
    run_root: Path,
    agent_id: str,
    *,
    preset: str,
    previous_error: str | None = None,
    comment_feed: str | None = None,
    soft_stop_active: bool = False,
):
    registry = build_skill_registry()
    visible_skills = visible_skills_for_preset(
        preset=preset,
        available_skills=get_skills_for_packs(registry, ["core", "equity"]),
    )
    return build_agent_prompt_bundle(
        request=request,
        run_root=str(run_root),
        agent_id=agent_id,
        preset=preset,
        response_mode="native_tools",
        available_tools=default_tool_allowlist(preset),
        available_skills=visible_skills,
        previous_error=previous_error,
        comment_feed=comment_feed,
        soft_stop_active=soft_stop_active,
    )


def test_harness_request_defaults_are_kernel_focused() -> None:
    request = HarnessRequest(user_prompt="Analyze AAPL")

    assert request.root_preset == "orchestrator"
    assert request.agent_transport == "auto"
    assert request.wall_clock_budget_seconds == 1200
    assert request.root_wall_clock_seconds is None
    assert request.max_agents_per_run == 64
    assert request.max_live_agents == 16
    assert request.max_live_children_per_parent == 8
    assert request.per_agent_wall_clock_seconds == 1800
    assert request.resume_from_run_root is None


def test_render_tools_markdown_lists_visible_runtime_surface() -> None:
    registry = build_skill_registry()
    skills = visible_skills_for_preset(
        preset="research",
        available_skills=get_skills_for_packs(registry, ["core", "equity"]),
    )

    text = render_tools_markdown(
        preset="research",
        available_tools=default_tool_allowlist("research"),
        available_skills=skills,
    )

    assert "# Runtime Interface" in text
    assert "`spawn_subagent`" in text
    assert "`bash`" in text
    assert "`apply_patch`" in text
    assert "edit_file(path=..., ...)" in text
    assert "short exact replacements or inserts" in text
    assert "apply_patch(patch=...)" in text
    assert "localized multi-line edits" in text
    assert "*** Begin Patch" in text
    assert "*** Update File: publish/example.md" in text
    assert "space for unchanged context lines" in text
    assert "If `apply_patch` fails because the context is missing or ambiguous" in text
    assert "read_file(path=...)" in text
    assert "write_file(path=..., content=...)" in text
    assert "replacing most of a file or creating a new one" in text
    assert "fetch_company_profile" in text
    assert "search_web_resources" not in text
    assert "retrieve_sources" not in text
    assert "Response Mode" in text


def test_orchestrator_tools_markdown_hides_direct_skills() -> None:
    registry = build_skill_registry()
    skills = visible_skills_for_preset(
        preset="orchestrator",
        available_skills=get_skills_for_packs(registry, ["core", "equity"]),
    )
    text = render_tools_markdown(
        preset="orchestrator",
        available_tools=default_tool_allowlist("orchestrator"),
        available_skills=skills,
    )

    assert "fetch_company_profile" not in text
    assert "read_file" in text
    assert "search_in_files" in text
    assert "`evaluator`" in text


def test_evaluator_preset_gets_primitive_core_skills() -> None:
    registry = build_skill_registry()
    skills = visible_skills_for_preset(
        preset="evaluator",
        available_skills=get_skills_for_packs(registry, ["core", "equity"]),
    )

    names = {spec.name for spec in skills}

    assert "read_file" in names
    assert "search_in_files" in names
    assert "get_current_datetime" in names
    assert "search_web" in names
    assert "read_web_pages" in names
    assert "condense_context" in names
    assert "fetch_company_profile" not in names


def test_invalid_skill_pack_is_rejected() -> None:
    with pytest.raises(ValueError):
        HarnessRequest(user_prompt="Analyze AAPL", available_skill_packs=["core", "illegal"])


def test_prompt_bundle_separates_system_and_user_channels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, agent_id, request = _create_workspace(monkeypatch, tmp_path, preset="research")
    paths = agent_workspace_paths(run_root, agent_id)
    write_text_atomic(paths["prompt_memory"], "Carry forward the valuation framing.\n")
    write_text_atomic(paths["history_summary"], "## Decisions\n- Prior synthesis already narrowed the scope.\n")

    bundle = _build_bundle(request, run_root, agent_id, preset="research")

    assert "# Runtime Interface" in bundle.system_prompt
    assert "# Task Assignment" not in bundle.system_prompt
    assert "Carry forward the valuation framing." in bundle.system_prompt
    assert "# Compacted History" in bundle.system_prompt
    assert "Prior synthesis already narrowed the scope." in bundle.system_prompt
    assert "# Task Assignment" in bundle.user_prompt
    assert "# Runtime Snapshot" in bundle.user_prompt
    assert "# Runtime History / Delta" in bundle.user_prompt
    assert "Carry forward the valuation framing." not in bundle.user_prompt
    assert "Prior synthesis already narrowed the scope." not in bundle.user_prompt


def test_agent_user_prompt_includes_runtime_capacity_fields_without_time_counts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, agent_id, request = _create_workspace(monkeypatch, tmp_path, preset="orchestrator")

    bundle = _build_bundle(request, run_root, agent_id, preset="orchestrator")

    assert "## Agent Identity" in bundle.user_prompt
    assert "- root_orchestrator: yes" in bundle.user_prompt
    assert "Runtime Capacity Snapshot" in bundle.user_prompt
    assert "remaining agent slots" in bundle.user_prompt
    assert "remaining live child slots" in bundle.user_prompt
    assert "remaining agent seconds" not in bundle.user_prompt
    assert "remaining root seconds" not in bundle.user_prompt
    assert "remaining run seconds" not in bundle.user_prompt


def test_agent_user_prompt_lists_context_as_artifact_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, agent_id, request = _create_workspace(monkeypatch, tmp_path, preset="research")
    source_file = tmp_path / "context-note.md"
    source_file.write_text("Context note\n", encoding="utf-8")
    create_agent_workspace(
        run_root,
        agent_id="agent_context",
        parent_id="",
        preset="research",
        task_name="Root Task",
        description="Analyze AAPL",
        task_markdown=render_task_markdown(request.user_prompt),
        tools_markdown=render_tools_markdown(
            preset="research",
            available_tools=default_tool_allowlist("research"),
            available_skills=[],
        ),
        context_files=[str(source_file)],
    )

    bundle = _build_bundle(request, run_root, "agent_context", preset="research")

    assert str(run_root / "agents" / "agent_context" / "context") in bundle.user_prompt
    assert "Read listed context file paths with `read_file(path=...)`." in bundle.user_prompt


def test_prompt_bundle_explicit_false_soft_stop_matches_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, agent_id, request = _create_workspace(monkeypatch, tmp_path, preset="orchestrator")

    baseline = _build_bundle(request, run_root, agent_id, preset="orchestrator")
    explicit_false = _build_bundle(
        request,
        run_root,
        agent_id,
        preset="orchestrator",
        soft_stop_active=False,
    )

    assert explicit_false.user_prompt == baseline.user_prompt


def test_prompt_bundle_soft_stop_appends_role_specific_guidance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, agent_id, request = _create_workspace(monkeypatch, tmp_path, preset="orchestrator")
    create_agent_workspace(
        run_root,
        agent_id="agent_child",
        parent_id=agent_id,
        preset="research",
        task_name="Child Task",
        description="Analyze AAPL child",
        task_markdown=render_task_markdown(request.user_prompt),
        tools_markdown=render_tools_markdown(
            preset="research",
            available_tools=default_tool_allowlist("research"),
            available_skills=[],
        ),
    )

    root_bundle = _build_bundle(
        request,
        run_root,
        agent_id,
        preset="orchestrator",
        soft_stop_active=True,
    )
    child_bundle = _build_bundle(
        request,
        run_root,
        "agent_child",
        preset="research",
        soft_stop_active=True,
    )

    shared_text = (
        "Soft-stop mode is active. Stop exploration and do not open new workstreams unless strictly necessary."
    )
    root_text = (
        "Spend the remaining turns improving publish/final.md, publish/summary.md, "
        "and publish/artifact_index.md so they stay readable if execution stops at any time."
    )
    child_text = (
        "Spend the remaining turns improving your current publish/ outputs so your parent "
        "can use them if execution stops at any time."
    )

    assert "## Soft-Stop Mode" in root_bundle.user_prompt
    assert shared_text in root_bundle.user_prompt
    assert root_text in root_bundle.user_prompt
    assert child_text not in root_bundle.user_prompt
    assert "## Soft-Stop Mode" in child_bundle.user_prompt
    assert shared_text in child_bundle.user_prompt
    assert child_text in child_bundle.user_prompt
    assert root_text not in child_bundle.user_prompt


def test_agent_system_prompt_loads_markdown_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prompts_root = tmp_path / "prompts"
    (prompts_root / "actors").mkdir(parents=True)
    (prompts_root / "roles").mkdir(parents=True)
    (prompts_root / "response_modes").mkdir(parents=True)
    (prompts_root / "system.md").write_text("SYSTEM FILE\n", encoding="utf-8")
    (prompts_root / "actors" / "agent_base.md").write_text("AGENT BASE FILE\n", encoding="utf-8")
    (prompts_root / "roles" / "research.md").write_text("ROLE FILE\n", encoding="utf-8")
    (prompts_root / "tools.md").write_text(
        "TOOLS FILE\n\n{{agent_tools}}\n\n{{deterministic_skills}}\n\n{{child_presets}}\n\n{{response_mode}}\n",
        encoding="utf-8",
    )
    (prompts_root / "task.md").write_text("TASK FILE\n\n{{user_prompt}}\n\n{{child_presets}}\n", encoding="utf-8")
    (prompts_root / "response_modes" / "native_tools.md").write_text("NATIVE MODE FILE\n", encoding="utf-8")
    monkeypatch.setattr(prompt_builder_module, "PROMPTS_ROOT", prompts_root)

    run_root, agent_id, request = _create_workspace(monkeypatch, tmp_path, preset="research")
    bundle = _build_bundle(request, run_root, agent_id, preset="research")

    assert "SYSTEM FILE" in bundle.system_prompt
    assert "AGENT BASE FILE" in bundle.system_prompt
    assert "ROLE FILE" in bundle.system_prompt
    assert "TOOLS FILE" in bundle.system_prompt
    assert "NATIVE MODE FILE" in bundle.system_prompt
    assert "`spawn_subagent`" in bundle.system_prompt


def test_commenter_prompt_bundle_uses_commenter_specific_layers_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prompts_root = tmp_path / "prompts"
    (prompts_root / "actors").mkdir(parents=True)
    (prompts_root / "internal").mkdir(parents=True)
    (prompts_root / "response_modes").mkdir(parents=True)
    (prompts_root / "system.md").write_text("SYSTEM FILE\n", encoding="utf-8")
    (prompts_root / "actors" / "commenter_base.md").write_text("COMMENTER BASE FILE\n", encoding="utf-8")
    (prompts_root / "internal" / "commenter_review.md").write_text("COMMENTER ROLE FILE\n", encoding="utf-8")
    (prompts_root / "commenter_interface.md").write_text(
        "COMMENTER INTERFACE FILE\n\n{{commenter_tools}}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(prompt_builder_module, "PROMPTS_ROOT", prompts_root)

    bundle = build_commenter_prompt_bundle(
        tools_available=True,
        user_prompt="Review the latest draft.",
    )

    assert "SYSTEM FILE" not in bundle.system_prompt
    assert "COMMENTER BASE FILE" in bundle.system_prompt
    assert "COMMENTER ROLE FILE" in bundle.system_prompt
    assert "COMMENTER INTERFACE FILE" in bundle.system_prompt
    assert "`read_file`" in bundle.system_prompt


def test_runtime_history_changes_user_prompt_not_system_prompt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, agent_id, request = _create_workspace(monkeypatch, tmp_path, preset="research")

    base = _build_bundle(request, run_root, agent_id, preset="research")
    changed = _build_bundle(
        request,
        run_root,
        agent_id,
        preset="research",
        previous_error="Need to call a valid tool.",
        comment_feed="Comment Feed\nCheck the weaker source.\n",
    )

    assert base.system_prompt == changed.system_prompt
    assert base.user_prompt != changed.user_prompt
    assert "Need to call a valid tool." in changed.user_prompt
    assert "Comment Feed" in changed.user_prompt
    assert "- tool=" not in changed.user_prompt


def test_tools_markdown_changes_system_prompt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, agent_id, request = _create_workspace(monkeypatch, tmp_path, preset="research")

    native_bundle = build_agent_prompt_bundle(
        request=request,
        run_root=str(run_root),
        agent_id=agent_id,
        preset="research",
        response_mode="native_tools",
        available_tools=default_tool_allowlist("research"),
        available_skills=[],
    )
    json_bundle = build_agent_prompt_bundle(
        request=request,
        run_root=str(run_root),
        agent_id=agent_id,
        preset="research",
        response_mode="text_json",
        available_tools=default_tool_allowlist("research"),
        available_skills=[],
    )

    assert native_bundle.system_prompt != json_bundle.system_prompt
    assert "Return exactly one JSON object" in json_bundle.system_prompt


def test_research_system_prompt_uses_child_completion_not_global_final_requirement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, agent_id, request = _create_workspace(monkeypatch, tmp_path, preset="research")

    bundle = _build_bundle(request, run_root, agent_id, preset="research")

    assert "write the final answer to `publish/final.md`" not in bundle.system_prompt
    assert "If you are a child agent, set `status` to `done`" in bundle.system_prompt


def test_orchestrator_prompt_encourages_synthesis_from_partial_child_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, agent_id, request = _create_workspace(monkeypatch, tmp_path, preset="orchestrator")

    bundle = _build_bundle(request, run_root, agent_id, preset="orchestrator")

    assert "Treat child outputs as inputs to judge" in bundle.system_prompt
    assert "The first decomposition is only a draft" in bundle.system_prompt


def test_child_completion_does_not_require_publish_final(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    request = HarnessRequest(user_prompt="Analyze AAPL", run_id="child-completion")
    run_root, root_agent_id = initialize_run_root(request)
    create_agent_workspace(
        run_root,
        agent_id=root_agent_id,
        parent_id="",
        preset="orchestrator",
        task_name="Root Task",
        description="Analyze AAPL",
        task_markdown=render_task_markdown(request.user_prompt),
        tools_markdown=render_tools_markdown(
            preset="orchestrator",
            available_tools=default_tool_allowlist("orchestrator"),
            available_skills=[],
        ),
    )
    create_agent_workspace(
        run_root,
        agent_id="agent_child",
        parent_id=root_agent_id,
        preset="research",
        task_name="Child Task",
        description="Write a topic-specific output.",
        task_markdown="# Task Assignment\n\n## Task Name\nChild Task\n",
        tools_markdown=render_tools_markdown(
            preset="research",
            available_tools=default_tool_allowlist("research"),
            available_skills=[],
        ),
    )

    root_paths = agent_workspace_paths(run_root, root_agent_id)
    child_paths = agent_workspace_paths(run_root, "agent_child")
    write_text_atomic(root_paths["publish_summary"], "# Summary\n")
    write_text_atomic(root_paths["publish_index"], "- final.md\n")
    assert _publish_outputs_satisfy_completion(str(run_root), root_agent_id) is False
    write_text_atomic(root_paths["publish_final"], "# Final\n")
    assert _publish_outputs_satisfy_completion(str(run_root), root_agent_id) is True

    write_text_atomic(child_paths["publish_summary"], "# Child Summary\n")
    write_text_atomic(child_paths["publish_root"] / "financials_valuation.md", "# Topic Output\n")
    assert _publish_outputs_satisfy_completion(str(run_root), "agent_child") is True
