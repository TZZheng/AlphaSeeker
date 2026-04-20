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
