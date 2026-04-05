from __future__ import annotations

from pathlib import Path

import pytest

from src.harness.artifacts import (
    agent_workspace_paths,
    create_agent_workspace,
    initialize_run_root,
    latest_agent_records,
    load_request,
    refresh_progress_view,
    save_skill_state,
)
from src.harness.presets import default_tool_allowlist, render_root_task_markdown, render_tools_markdown
from src.harness.registry import build_skill_registry, get_skills_for_packs
from src.harness.types import HarnessRequest, HarnessState


def test_initialize_run_root_and_workspace_layout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    request = HarnessRequest(user_prompt="Analyze AAPL", run_id="artifact-layout")
    run_root, root_agent_id = initialize_run_root(request)
    registry = build_skill_registry()

    create_agent_workspace(
        run_root,
        agent_id=root_agent_id,
        parent_id="",
        preset="research",
        task_name="Root Task",
        description="Analyze AAPL",
        task_markdown=render_root_task_markdown(request.user_prompt),
        tools_markdown=render_tools_markdown(
            preset="research",
            available_tools=default_tool_allowlist("research"),
            available_skills=get_skills_for_packs(registry, ["core", "equity"]),
        ),
    )
    refresh_progress_view(run_root)

    paths = agent_workspace_paths(run_root, root_agent_id)
    assert Path(run_root, "registry", "agents.jsonl").exists()
    assert Path(run_root, "registry", "events.jsonl").exists()
    assert Path(run_root, "request.json").exists()
    assert paths["task"].exists()
    assert paths["tools"].exists()
    assert paths["state_root"].exists()
    assert paths["transcript"].exists()
    assert paths["tool_history"].exists()
    assert load_request(run_root).user_prompt == "Analyze AAPL"
    assert latest_agent_records(run_root)[root_agent_id].status == "queued"


def test_save_skill_state_persists_agent_runtime(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    request = HarnessRequest(user_prompt="Analyze AAPL", run_id="skill-state")
    run_root, root_agent_id = initialize_run_root(request)
    registry = build_skill_registry()
    create_agent_workspace(
        run_root,
        agent_id=root_agent_id,
        parent_id="",
        preset="research",
        task_name="Root Task",
        description="Analyze AAPL",
        task_markdown=render_root_task_markdown(request.user_prompt),
        tools_markdown=render_tools_markdown(
            preset="research",
            available_tools=default_tool_allowlist("research"),
            available_skills=get_skills_for_packs(registry, ["core"]),
        ),
    )

    state = HarnessState(
        request=request,
        run_id="skill-state",
        run_root=str(run_root),
        agent_id=root_agent_id,
        workspace_path=str(agent_workspace_paths(run_root, root_agent_id)["workspace"]),
        enabled_packs=["core"],
    )
    save_skill_state(state)

    assert agent_workspace_paths(run_root, root_agent_id)["skill_state"].exists()
