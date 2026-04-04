from __future__ import annotations

import pytest

from src.harness.agent_worker import _build_initial_user_prompt, _publish_outputs_satisfy_completion
from src.harness.artifacts import agent_workspace_paths, create_agent_workspace, initialize_run_root, write_text_atomic
from src.harness import presets as presets_module
from src.harness.presets import (
    default_tool_allowlist,
    preset_system_prompt,
    preset_system_prompt_with_context,
    render_tools_markdown,
    visible_skills_for_preset,
)
from src.harness.registry import build_skill_registry, get_skills_for_packs
from src.harness.types import HarnessRequest


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


def test_render_tools_markdown_lists_agent_tools_and_skills() -> None:
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

    assert "`spawn_subagent`" in text
    assert "fetch_company_profile" in text
    assert "search_in_files" in text
    assert "search_web_resources" not in text
    assert "read_file" in text
    assert "read_web_pages" in text
    assert "retrieve_sources" not in text
    assert "Subagent Presets" in text
    assert "`orchestrator`" in text


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

    assert "`spawn_subagent`" in text
    assert "fetch_company_profile" not in text
    assert "read_file" in text
    assert "get_current_datetime" in text
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
    assert "search_news" in names
    assert "read_web_pages" in names
    assert "condense_context" in names
    assert "fetch_company_profile" not in names


def test_invalid_skill_pack_is_rejected() -> None:
    with pytest.raises(ValueError):
        HarnessRequest(user_prompt="Analyze AAPL", available_skill_packs=["core", "illegal"])


def test_initial_worker_prompt_includes_runtime_capacity_fields_without_time_counts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    request = HarnessRequest(
        user_prompt="Analyze AAPL",
        run_id="prompt-budget",
        root_wall_clock_seconds=900,
    )
    run_root, root_agent_id = initialize_run_root(request)
    create_agent_workspace(
        run_root,
        agent_id=root_agent_id,
        parent_id="",
        preset="orchestrator",
        task_name="Root Task",
        description="Analyze AAPL",
        task_markdown="# Root Task\n",
        tools_markdown=render_tools_markdown(
            preset="orchestrator",
            available_tools=default_tool_allowlist("orchestrator"),
            available_skills=[],
        ),
    )

    prompt = _build_initial_user_prompt(str(run_root), root_agent_id)

    assert "Agent Identity" in prompt
    assert "- root_orchestrator: yes" in prompt
    assert "Runtime Capacity Snapshot" in prompt
    assert "remaining agent slots" in prompt
    assert "remaining live child slots" in prompt
    assert "effective agent time limit (seconds)" not in prompt
    assert "remaining agent seconds" not in prompt
    assert "remaining root seconds" not in prompt
    assert "remaining run seconds" not in prompt


def test_initial_worker_prompt_lists_context_as_artifact_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.chdir(tmp_path)
    request = HarnessRequest(user_prompt="Analyze AAPL", run_id="prompt-context")
    run_root, root_agent_id = initialize_run_root(request)
    source_file = tmp_path / "context-note.md"
    source_file.write_text("Context note\n", encoding="utf-8")
    create_agent_workspace(
        run_root,
        agent_id=root_agent_id,
        parent_id="",
        preset="research",
        task_name="Root Task",
        description="Analyze AAPL",
        task_markdown="# Root Task\n",
        tools_markdown=render_tools_markdown(
            preset="research",
            available_tools=default_tool_allowlist("research"),
            available_skills=[],
        ),
        context_files=[str(source_file)],
    )

    prompt = _build_initial_user_prompt(str(run_root), root_agent_id)

    assert str(run_root / "agents" / root_agent_id / "context") in prompt
    assert "Read listed context file paths with `read_file(path=...)`." in prompt


def test_prompts_push_deadline_aware_partial_synthesis_and_incremental_publish() -> None:
    orchestrator_text = preset_system_prompt("orchestrator")
    research_text = preset_system_prompt("research")
    evaluator_text = preset_system_prompt("evaluator")

    assert "quality of the final answer" in orchestrator_text
    assert "Choose the number and boundaries of workstreams yourself" in orchestrator_text
    assert "Before finalizing, ask whether the current answer is coherent" in orchestrator_text
    assert "publish early findings to `publish/summary.md`" in research_text
    assert "high-confidence findings, open conflicts, and weak-source" in research_text
    assert "You are not the root orchestrator." in research_text
    assert "Audit a draft, research output, or synthesis" in evaluator_text
    assert "paired commenter may sometimes provide outside-angle advisory comments" in orchestrator_text
    assert "If time is short" not in orchestrator_text
    assert "time is getting short" not in research_text


def test_system_prompt_is_loaded_from_markdown_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    prompts_root = tmp_path / "prompts"
    roles_root = prompts_root / "roles"
    roles_root.mkdir(parents=True)
    (prompts_root / "system.md").write_text("SYSTEM FILE\n", encoding="utf-8")
    (prompts_root / "environment.md").write_text(
        "ENV FILE\n\n{{agent_tools}}\n\n{{deterministic_skills}}\n\n{{child_presets}}\n\n{{response_mode}}\n",
        encoding="utf-8",
    )
    response_modes_root = prompts_root / "response_modes"
    response_modes_root.mkdir()
    (response_modes_root / "native_tools.md").write_text("NATIVE MODE FILE\n", encoding="utf-8")
    (roles_root / "research.md").write_text("ROLE FILE\n", encoding="utf-8")
    monkeypatch.setattr(presets_module, "PROMPTS_ROOT", prompts_root)

    text = preset_system_prompt_with_context(
        "research",
        available_tools=["spawn_subagent"],
        available_skills=[],
    )

    assert "SYSTEM FILE" in text
    assert "ROLE FILE" in text
    assert "ENV FILE" in text
    assert "NATIVE MODE FILE" in text
    assert "`spawn_subagent`" in text


def test_shared_system_prompt_no_longer_requires_publish_final_for_all_agents() -> None:
    research_text = preset_system_prompt("research")

    assert "write the final answer to `publish/final.md`" not in research_text
    assert "If you are a child agent, `done` requires at least one non-empty published output file" in research_text


def test_prompt_environment_mentions_search_before_read() -> None:
    research_text = preset_system_prompt("research")

    assert "search_in_files(pattern=..., paths=[...])" in research_text
    assert "read_file(path=..., max_chars=..., start_char=...)" in research_text
    assert "read_web_pages(urls=[...])" in research_text
    assert "Refresh `publish/summary.md` after meaningful progress" in research_text
    assert "If you are a child agent, `done` requires at least one non-empty published output file" in research_text


def test_orchestrator_prompt_encourages_synthesis_from_partial_child_outputs() -> None:
    orchestrator_text = preset_system_prompt("orchestrator")

    assert "Treat child outputs as inputs to judge" in orchestrator_text
    assert "The first decomposition is only a draft" in orchestrator_text


def test_child_completion_does_not_require_publish_final(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
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
        task_markdown="# Root Task\n",
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
        task_markdown="# Child Task\n",
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
