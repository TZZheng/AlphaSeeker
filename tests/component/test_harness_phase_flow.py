from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from src.harness.artifacts import agent_workspace_paths, create_agent_workspace, latest_agent_records, write_text_atomic, write_status
from src.harness.presets import default_tool_allowlist
from src.harness.prompt_builder import render_tools_markdown
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


def _publish_done(run_root: str, agent_id: str, *, body: str) -> None:
    paths = agent_workspace_paths(run_root, agent_id)
    write_text_atomic(paths["publish_summary"], f"{agent_id} summary\n")
    write_text_atomic(paths["publish_index"], "- final.md: Final output\n")
    write_text_atomic(paths["publish_final"], body)
    write_status(run_root, agent_id, "done")


def test_root_can_spawn_child_and_finish_after_child_publish(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    registry = build_skill_registry()
    counter = {"value": 3000}

    async def _launcher(run_root: str, agent_id: str):
        counter["value"] += 1
        process = _FakeProcess(counter["value"])

        async def _runner() -> None:
            if agent_id == "agent_root":
                write_status(run_root, agent_id, "waiting")
                child_id = "agent_child"
                create_agent_workspace(
                    run_root,
                    agent_id=child_id,
                    parent_id=agent_id,
                    preset="writer",
                    task_name="Child Draft",
                    description="Draft a short child answer.",
                    task_markdown="# Child\n\nWrite a short child answer.\n",
                    tools_markdown=render_tools_markdown(
                        preset="writer",
                        available_tools=default_tool_allowlist("writer"),
                        available_skills=get_skills_for_packs(registry, ["core"]),
                    ),
                )
                child_final = agent_workspace_paths(run_root, child_id)["publish_final"]
                while not child_final.exists():
                    await asyncio.sleep(0.05)
                _publish_done(run_root, agent_id, body="# Root\n\nIntegrated child result.\n")
            else:
                await asyncio.sleep(0.1)
                _publish_done(run_root, agent_id, body="# Child\n\nChild result.\n")
            process.returncode = 0

        asyncio.create_task(_runner())
        return process

    response = run_harness(
        HarnessRequest(user_prompt="Analyze AAPL"),
        launch_agent_process=_launcher,
    )

    assert response.status == "completed"
    records = latest_agent_records(response.run_root or "")
    assert {"agent_root", "agent_child"} <= set(records)
    assert records["agent_child"].parent_id == "agent_root"
    assert Path(response.final_report_path or "").exists()
