from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import src.harness.runtime as runtime_module
from src.harness.artifacts import agent_workspace_paths, create_agent_workspace, latest_agent_records, write_text_atomic, write_status
from src.harness.presets import default_tool_allowlist
from src.harness.prompt_builder import render_tools_markdown
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


def _complete_agent(run_root: str, agent_id: str, *, body: str) -> None:
    paths = agent_workspace_paths(run_root, agent_id)
    write_text_atomic(paths["publish_summary"], f"# Summary\n\n{agent_id} finished.\n")
    write_text_atomic(paths["publish_index"], "- final.md: Final answer\n")
    write_text_atomic(paths["publish_final"], body)
    write_status(run_root, agent_id, "done")


def test_run_harness_completes_with_root_publish(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    counter = {"value": 1000}

    async def _launcher(run_root: str, agent_id: str):
        counter["value"] += 1
        process = _FakeProcess(counter["value"])

        async def _runner() -> None:
            await asyncio.sleep(0.05)
            _complete_agent(run_root, agent_id, body="# Final\n\nRoot answer.\n")
            process.returncode = 0

        asyncio.create_task(_runner())
        return process

    response = run_harness(
        HarnessRequest(user_prompt="Analyze AAPL"),
        launch_agent_process=_launcher,
    )

    assert response.status == "completed"
    assert response.stop_reason == "done"
    assert response.final_report_path is not None
    assert Path(response.final_report_path).exists()
    assert "Root answer." in Path(response.final_report_path).read_text(encoding="utf-8")


def test_run_harness_fails_when_root_never_publishes_final(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    async def _launcher(run_root: str, agent_id: str):
        process = _FakeProcess(2001)

        async def _runner() -> None:
            await asyncio.sleep(0.05)
            write_status(run_root, agent_id, "failed")
            process.returncode = 1

        asyncio.create_task(_runner())
        return process

    response = run_harness(
        HarnessRequest(user_prompt="Analyze AAPL"),
        launch_agent_process=_launcher,
    )

    assert response.status == "failed"
    assert response.final_report_path is None
    assert response.error is not None


def test_run_harness_stops_on_run_wall_clock_when_root_never_finishes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runtime_module, "SOFT_STOP_GRACE_SECONDS", 0.2)

    async def _launcher(_run_root: str, _agent_id: str):
        return _FakeProcess(2002)

    response = run_harness(
        HarnessRequest(
            user_prompt="Analyze AAPL",
            wall_clock_budget_seconds=2,
            root_wall_clock_seconds=1,
            per_agent_wall_clock_seconds=1,
        ),
        launch_agent_process=_launcher,
    )

    assert response.status == "time_out"
    assert response.stop_reason == "wall_clock_budget_exhausted"
    assert response.error is not None


def test_run_harness_returns_timeout_with_deliverable_when_root_publish_exists(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runtime_module, "POLL_INTERVAL_SECONDS", 0.05)
    monkeypatch.setattr(runtime_module, "SOFT_STOP_GRACE_SECONDS", 0.2)

    async def _launcher(run_root: str, agent_id: str):
        process = _FakeProcess(2040)

        async def _runner() -> None:
            write_status(run_root, agent_id, "running")
            await asyncio.sleep(1.05)
            paths = agent_workspace_paths(run_root, agent_id)
            write_text_atomic(paths["publish_summary"], "# Summary\n\nReadable draft exists.\n")
            write_text_atomic(paths["publish_index"], "- final.md: Draft final answer\n")
            write_text_atomic(paths["publish_final"], "# Final\n\nReadable draft before timeout.\n")
            await asyncio.sleep(5.0)

        asyncio.create_task(_runner())
        return process

    response = run_harness(
        HarnessRequest(
            user_prompt="Analyze AAPL",
            wall_clock_budget_seconds=1,
            root_wall_clock_seconds=5,
            per_agent_wall_clock_seconds=5,
        ),
        launch_agent_process=_launcher,
    )

    assert response.status == "time_out_with_deliverable"
    assert response.stop_reason == "wall_clock_budget_exhausted"
    assert response.final_report_path is not None
    assert "Readable draft before timeout." in Path(response.final_report_path).read_text(encoding="utf-8")


def test_run_harness_allows_root_to_finish_during_soft_stop_grace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runtime_module, "POLL_INTERVAL_SECONDS", 0.05)
    monkeypatch.setattr(runtime_module, "SOFT_STOP_GRACE_SECONDS", 0.6)

    async def _launcher(run_root: str, agent_id: str):
        process = _FakeProcess(2050)

        async def _runner() -> None:
            write_status(run_root, agent_id, "running")
            await asyncio.sleep(1.15)
            _complete_agent(run_root, agent_id, body="# Final\n\nRoot answer after soft stop.\n")
            process.returncode = 0

        asyncio.create_task(_runner())
        return process

    response = run_harness(
        HarnessRequest(
            user_prompt="Analyze AAPL",
            wall_clock_budget_seconds=1,
            root_wall_clock_seconds=5,
            per_agent_wall_clock_seconds=5,
        ),
        launch_agent_process=_launcher,
    )

    assert response.status == "completed"
    assert response.stop_reason == "done"
    assert response.final_report_path is not None
    assert "soft stop" in Path(response.final_report_path).read_text(encoding="utf-8")


def test_root_is_not_limited_by_child_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    async def _launcher(run_root: str, agent_id: str):
        process = _FakeProcess(2100 if agent_id == "agent_root" else 2101)

        async def _runner() -> None:
            await asyncio.sleep(1.2)
            _complete_agent(run_root, agent_id, body="# Final\n\nRoot answer.\n")
            process.returncode = 0

        asyncio.create_task(_runner())
        return process

    response = run_harness(
        HarnessRequest(
            user_prompt="Analyze AAPL",
            wall_clock_budget_seconds=5,
            root_wall_clock_seconds=3,
            per_agent_wall_clock_seconds=1,
        ),
        launch_agent_process=_launcher,
    )

    assert response.status == "completed"
    assert response.final_report_path is not None


def test_run_harness_caps_global_live_agents(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    counter = {"value": 3000}
    active_agents: set[str] = set()
    max_active = {"value": 0}

    async def _launcher(run_root: str, agent_id: str):
        counter["value"] += 1
        process = _FakeProcess(counter["value"])
        active_agents.add(agent_id)
        max_active["value"] = max(max_active["value"], len(active_agents))

        async def _runner() -> None:
            if agent_id == "agent_root":
                write_status(run_root, agent_id, "waiting")
                await asyncio.sleep(0.02)
                create_agent_workspace(
                    run_root,
                    agent_id="agent_child_1",
                    parent_id=agent_id,
                    preset="research",
                    task_name="Child 1",
                    description="First child",
                    task_markdown="# Child 1\n",
                    tools_markdown=render_tools_markdown(
                        preset="research",
                        available_tools=default_tool_allowlist("research"),
                        available_skills=[],
                    ),
                )
                create_agent_workspace(
                    run_root,
                    agent_id="agent_child_2",
                    parent_id=agent_id,
                    preset="research",
                    task_name="Child 2",
                    description="Second child",
                    task_markdown="# Child 2\n",
                    tools_markdown=render_tools_markdown(
                        preset="research",
                        available_tools=default_tool_allowlist("research"),
                        available_skills=[],
                    ),
                )
                while not (
                    agent_workspace_paths(run_root, "agent_child_1")["publish_final"].exists()
                    and agent_workspace_paths(run_root, "agent_child_2")["publish_final"].exists()
                ):
                    await asyncio.sleep(0.05)
                _complete_agent(run_root, agent_id, body="# Final\n\nRoot answer.\n")
                active_agents.discard(agent_id)
            else:
                await asyncio.sleep(0.15)
                active_agents.discard(agent_id)
                _complete_agent(run_root, agent_id, body=f"# Final\n\n{agent_id} answer.\n")
            process.returncode = 0

        asyncio.create_task(_runner())
        return process

    response = run_harness(
        HarnessRequest(
            user_prompt="Analyze AAPL",
            max_live_agents=2,
            max_live_children_per_parent=2,
            wall_clock_budget_seconds=10,
        ),
        launch_agent_process=_launcher,
    )

    assert response.status == "completed"
    assert max_active["value"] <= 2


def test_root_failure_cancels_live_descendants(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    async def _launcher(run_root: str, agent_id: str):
        process = _FakeProcess(3100 if agent_id == "agent_root" else 3101)

        async def _runner() -> None:
            if agent_id == "agent_root":
                write_status(run_root, agent_id, "waiting")
                await asyncio.sleep(0.02)
                create_agent_workspace(
                    run_root,
                    agent_id="agent_child",
                    parent_id=agent_id,
                    preset="research",
                    task_name="Child",
                    description="Long child",
                    task_markdown="# Child\n",
                    tools_markdown=render_tools_markdown(
                        preset="research",
                        available_tools=default_tool_allowlist("research"),
                        available_skills=[],
                    ),
                )
                await asyncio.sleep(0.05)
                write_status(run_root, agent_id, "failed")
                process.returncode = 1
            else:
                await asyncio.sleep(5.0)

        asyncio.create_task(_runner())
        return process

    response = run_harness(
        HarnessRequest(
            user_prompt="Analyze AAPL",
            wall_clock_budget_seconds=10,
            root_wall_clock_seconds=5,
            per_agent_wall_clock_seconds=30,
        ),
        launch_agent_process=_launcher,
    )

    assert response.status == "failed"
    records = latest_agent_records(response.run_root)
    assert records["agent_child"].status == "cancelled"


def test_run_harness_schedules_commenter_refreshes_on_state_change(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.harness.runtime.POLL_INTERVAL_SECONDS", 0.05)
    monkeypatch.setattr("src.harness.runtime.COMMENTER_REFRESH_INTERVAL_SECONDS", 0.0)
    commenter_calls: list[dict[str, object]] = []

    def _fake_refresh(run_root: str, agent_id: str, request, *, model_name=None, transport_name=None, observation_snapshot=None):
        commenter_calls.append(
            {
                "agent_id": agent_id,
                "snapshot": observation_snapshot,
            }
        )
        return 0

    monkeypatch.setattr("src.harness.runtime.refresh_commenter_for_agent", _fake_refresh)

    async def _launcher(run_root: str, agent_id: str):
        process = _FakeProcess(4001)

        async def _runner() -> None:
            await asyncio.sleep(0.08)
            write_text_atomic(
                agent_workspace_paths(run_root, agent_id)["scratch_root"] / "note.md",
                "new scratch state\n",
            )
            await asyncio.sleep(0.08)
            _complete_agent(run_root, agent_id, body="# Final\n\nRoot answer.\n")
            process.returncode = 0

        asyncio.create_task(_runner())
        return process

    response = run_harness(
        HarnessRequest(
            user_prompt="Analyze AAPL",
            wall_clock_budget_seconds=5,
        ),
        launch_agent_process=_launcher,
    )

    assert response.status == "completed"
    assert any(call["agent_id"] == "agent_root" for call in commenter_calls)
    root_snapshots = [call["snapshot"] for call in commenter_calls if call["agent_id"] == "agent_root"]
    assert all(isinstance(snapshot, dict) for snapshot in root_snapshots)
    assert all("changed_entries" in snapshot for snapshot in root_snapshots if isinstance(snapshot, dict))
    assert any(
        entry.get("display_path") == "scratch/note.md"
        for snapshot in root_snapshots
        if isinstance(snapshot, dict)
        for entry in list(snapshot.get("changed_entries") or [])
    )
