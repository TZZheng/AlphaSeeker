from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from src.harness.artifacts import (
    agent_workspace_paths,
    create_agent_workspace,
    initialize_run_root,
    latest_agent_records,
    load_request,
    write_heartbeat,
    write_status,
)
from src.harness.presets import default_tool_allowlist
from src.harness.prompt_builder import render_task_markdown, render_tools_markdown
from src.harness.registry import build_skill_registry, get_skills_for_packs
from src.harness.runtime import run_harness
from src.harness.types import HarnessRequest

pytestmark = pytest.mark.component


class _FakeProcess:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.returncode: int | None = None

    def terminate(self) -> None:
        if self.returncode is None:
            self.returncode = -15

    def kill(self) -> None:
        if self.returncode is None:
            self.returncode = -9

    async def wait(self) -> int:
        while self.returncode is None:
            await asyncio.sleep(0.01)
        return self.returncode


def _publish_done(run_root: str, agent_id: str, body: str) -> None:
    paths = agent_workspace_paths(run_root, agent_id)
    paths["publish_root"].mkdir(parents=True, exist_ok=True)
    paths["publish_summary"].write_text(f"{agent_id} summary\n", encoding="utf-8")
    paths["publish_index"].write_text("- final.md: Final output\n", encoding="utf-8")
    paths["publish_final"].write_text(body, encoding="utf-8")
    write_status(run_root, agent_id, "done")


def test_recursive_child_launch_reaches_three_agent_levels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    registry = build_skill_registry()
    counter = {"value": 4000}

    async def _launcher(run_root: str, agent_id: str):
        counter["value"] += 1
        process = _FakeProcess(counter["value"])

        async def _runner() -> None:
            if agent_id == "agent_root":
                write_status(run_root, agent_id, "waiting")
                create_agent_workspace(
                    run_root,
                    agent_id="agent_child",
                    parent_id=agent_id,
                    preset="research",
                    task_name="Child Research",
                    description="Delegate a middle-layer task.",
                    task_markdown="# Child Research\n\nDelegate again.\n",
                    tools_markdown=render_tools_markdown(
                        preset="research",
                        available_tools=default_tool_allowlist("research"),
                        available_skills=get_skills_for_packs(registry, ["core"]),
                    ),
                )
                while not agent_workspace_paths(run_root, "agent_child")["publish_final"].exists():
                    await asyncio.sleep(0.05)
                _publish_done(run_root, agent_id, "# Root\n\nRoot result.\n")
            elif agent_id == "agent_child":
                write_status(run_root, agent_id, "waiting")
                create_agent_workspace(
                    run_root,
                    agent_id="agent_grandchild",
                    parent_id=agent_id,
                    preset="writer",
                    task_name="Grandchild Draft",
                    description="Produce the leaf answer.",
                    task_markdown="# Grandchild\n\nWrite the leaf answer.\n",
                    tools_markdown=render_tools_markdown(
                        preset="writer",
                        available_tools=default_tool_allowlist("writer"),
                        available_skills=get_skills_for_packs(registry, ["core"]),
                    ),
                )
                while not agent_workspace_paths(run_root, "agent_grandchild")["publish_final"].exists():
                    await asyncio.sleep(0.05)
                _publish_done(run_root, agent_id, "# Child\n\nChild result.\n")
            else:
                await asyncio.sleep(0.1)
                _publish_done(run_root, agent_id, "# Grandchild\n\nLeaf result.\n")
            process.returncode = 0

        asyncio.create_task(_runner())
        return process

    response = run_harness(HarnessRequest(user_prompt="Analyze AAPL"), launch_agent_process=_launcher)

    assert response.status == "completed"
    records = latest_agent_records(response.run_root or "")
    assert {"agent_root", "agent_child", "agent_grandchild"} <= set(records)
    assert records["agent_grandchild"].parent_id == "agent_child"


def test_resume_marks_stale_child_without_relaunching_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    request = HarnessRequest(user_prompt="Resume kernel test")
    run_root, _ = initialize_run_root(request)
    registry = build_skill_registry()
    create_agent_workspace(
        run_root,
        agent_id="agent_root",
        parent_id="",
        preset="research",
        task_name="Root Task",
        description="Resume the run.",
        task_markdown=render_task_markdown(request.user_prompt),
        tools_markdown=render_tools_markdown(
            preset="research",
            available_tools=default_tool_allowlist("research"),
            available_skills=get_skills_for_packs(registry, ["core"]),
        ),
    )
    create_agent_workspace(
        run_root,
        agent_id="agent_child",
        parent_id="agent_root",
        preset="writer",
        task_name="Stale Child",
        description="This child should become stale.",
        task_markdown="# Child\n\nStale child.\n",
        tools_markdown=render_tools_markdown(
            preset="writer",
            available_tools=default_tool_allowlist("writer"),
            available_skills=get_skills_for_packs(registry, ["core"]),
        ),
    )
    write_status(str(run_root), "agent_child", "running")
    write_heartbeat(str(run_root), "agent_child", "2000-01-01T00:00:00+00:00")

    async def _launcher(run_root_str: str, agent_id: str):
        process = _FakeProcess(5001)

        async def _runner() -> None:
            await asyncio.sleep(0.05)
            _publish_done(run_root_str, agent_id, "# Root\n\nResumed result.\n")
            process.returncode = 0

        asyncio.create_task(_runner())
        return process

    response = run_harness(
        HarnessRequest(user_prompt="ignored", resume_from_run_root=str(run_root), stale_heartbeat_seconds=1),
        launch_agent_process=_launcher,
    )

    assert response.status == "completed"
    records = latest_agent_records(str(run_root))
    assert records["agent_child"].status == "stale"
    assert records["agent_root"].status == "done"
