from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from src.harness.artifacts import agent_workspace_paths, write_text_atomic, write_status
from src.harness.runtime import run_harness
from src.harness.types import HarnessRequest

pytestmark = pytest.mark.live


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


def test_harness_smoke_with_injected_kernel_launcher(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    async def _launcher(run_root: str, agent_id: str):
        process = _FakeProcess(9001)

        async def _runner() -> None:
            await asyncio.sleep(0.05)
            paths = agent_workspace_paths(run_root, agent_id)
            write_text_atomic(paths["publish_summary"], "root summary\n")
            write_text_atomic(paths["publish_index"], "- final.md: Final response\n")
            write_text_atomic(paths["publish_final"], "# Final\n\nKernel smoke output.\n")
            write_status(run_root, agent_id, "done")
            process.returncode = 0

        asyncio.create_task(_runner())
        return process

    response = run_harness(
        HarnessRequest(user_prompt="Analyze AAPL", run_id="live-smoke"),
        launch_agent_process=_launcher,
    )

    assert response.status == "completed"
    assert Path(response.final_report_path or "").exists()
