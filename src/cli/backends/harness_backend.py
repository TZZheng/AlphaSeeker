"""Harness backend thread — wraps _supervise_async for the CLI dashboard."""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.harness.artifacts import (
    agent_workspace_paths,
    build_run_root,
    latest_agent_records,
    read_heartbeat,
    read_jsonl,
    read_status,
)
from src.harness.types import AgentRecord, HarnessRequest, HarnessResponse


@dataclass
class DashboardSnapshot:
    """Point-in-time state snapshot for the TUI dashboard."""

    agents: dict[str, AgentRecord]
    elapsed_seconds: float
    evidence_count: int
    status: str  # "running" | "completed" | "failed"
    final_report: str | None = None
    run_root: str | None = None


class HarnessBackend:
    """
    Thread-safe wrapper around the harness runtime for use by the CLI dashboard.

    Runs ``_supervise_async`` in a background thread and exposes
    a ``poll()`` method that reads the run_root filesystem to produce
    ``DashboardSnapshot`` objects consumed by the TUI.
    """

    def __init__(self, request: HarnessRequest):
        self._request = request
        self._response: HarnessResponse | None = None
        self._thread: threading.Thread | None = None
        self._started_at: float | None = None
        self._done = threading.Event()
        self._lock = threading.Lock()
        # Compute run_root deterministically so the dashboard can start polling
        # before _supervise_async finishes initializing.
        self._run_root_path: str = str(build_run_root(request))

    @property
    def run_root(self) -> str:
        """The filesystem path where this run's artifacts are written."""
        return self._run_root_path

    def start(self) -> None:
        """Start the harness runtime in a background thread."""
        self._started_at = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        import os
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        from src.harness.runtime import _supervise_async
        try:
            self._response = asyncio.run(_supervise_async(self._request))
        except Exception as exc:
            self._response = HarnessResponse(
                status="failed",
                error=f"Uncaught exception in harness backend: {exc}",
            )
        finally:
            self._done.set()

    def poll(self, run_root: str) -> DashboardSnapshot:
        """
        Poll the run_root filesystem and return a DashboardSnapshot.

        Safe to call from the main (TUI) thread while the backend runs.
        """
        elapsed = time.time() - (self._started_at or time.time())
        agents = latest_agent_records(run_root)
        status = read_status(run_root, "agent_root")

        evidence_count = 0
        ledger_path = Path(run_root) / "evidence_ledger.jsonl"
        if ledger_path.exists():
            evidence_count = len(read_jsonl(ledger_path))

        final_report: str | None = _best_available_report(run_root, status)

        return DashboardSnapshot(
            agents={k: v for k, v in agents.items()},
            elapsed_seconds=elapsed,
            evidence_count=evidence_count,
            status=status,
            final_report=final_report,
            run_root=run_root,
        )

    def wait(self, timeout: float | None = None) -> HarnessResponse | None:
        """Block until the harness run completes. Returns the final response."""
        if self._thread is None:
            return None
        self._done.wait(timeout=timeout)
        return self._response

    def request_soft_stop(self) -> None:
        """Write the stop-request sentinel file so _monitor_agents triggers a graceful shutdown."""
        stop_file = Path(self._run_root_path) / "stop_requested"
        try:
            stop_file.touch()
        except OSError:
            pass  # run_root may not exist yet if start was very recent

    @property
    def is_done(self) -> bool:
        return self._done.is_set()

    @property
    def response(self) -> HarnessResponse | None:
        return self._response


def _best_available_report(run_root: str, status: str) -> str | None:
    """
    Return the best available report content for the done screen.

    Priority:
    1. agent_root/publish/final.md   — complete synthesized report
    2. agent_root/publish/summary.md — partial summary from root agent
    3. Concatenated child-agent research files — raw research if root timed out
    """
    if not run_root:
        return None

    root_pub = Path(run_root) / "agents" / "agent_root" / "publish"

    # 1. Final report (preferred, any non-empty content counts)
    final_path = root_pub / "final.md"
    if final_path.exists():
        content = final_path.read_text(encoding="utf-8").strip()
        if content:
            return content

    # 2. Root summary
    summary_path = root_pub / "summary.md"
    if summary_path.exists():
        content = summary_path.read_text(encoding="utf-8").strip()
        if content:
            return f"*(Run ended with status: {status} — showing partial summary)*\n\n{content}"

    # 3. Collect child agent research as fallback
    agents_dir = Path(run_root) / "agents"
    if not agents_dir.exists():
        return None
    sections: list[str] = []
    for agent_dir in sorted(agents_dir.iterdir()):
        if not agent_dir.is_dir() or agent_dir.name == "agent_root":
            continue
        for fname in ("final.md", "summary.md"):
            pub_path = agent_dir / "publish" / fname
            if not pub_path.exists():
                continue
            try:
                text = pub_path.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if text:
                sections.append(f"## {agent_dir.name} — {fname}\n\n{text}")
                break  # one file per agent is enough

    if sections:
        header = f"*(Run ended with status: {status} — showing available child-agent research)*\n\n"
        return header + "\n\n---\n\n".join(sections)

    return None
