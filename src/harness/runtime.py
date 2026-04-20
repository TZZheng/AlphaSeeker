"""Async supervisor kernel for the file-based harness runtime."""

from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import time
from typing import Any

from src.harness.artifacts import (
    agent_workspace_paths,
    append_event,
    build_run_root,
    create_agent_workspace,
    initialize_run_root,
    latest_agent_records,
    load_commenter_state,
    load_request,
    read_heartbeat,
    read_jsonl,
    read_status,
    refresh_progress_view,
    save_commenter_state,
    stale_agents,
    update_agent_record,
    write_status,
    write_text_atomic,
)
from src.harness.commenter import (
    COMMENTER_REFRESH_INTERVAL_SECONDS,
    build_commenter_observation_snapshot,
    compute_commenter_observation_fingerprint,
    refresh_commenter_for_agent,
)
from src.harness.presets import default_tool_allowlist, visible_skills_for_preset
from src.harness.prompt_builder import render_task_markdown, render_tools_markdown
from src.harness.registry import build_skill_registry, get_skills_for_packs
from src.harness.types import AgentEvent, HarnessRequest, HarnessResponse, SkillSpec


TERMINAL_STATUSES = {"done", "failed", "blocked", "stale", "cancelled"}
POLL_INTERVAL_SECONDS = 1.0
PROCESS_KILL_GRACE_SECONDS = 2.0
SOFT_STOP_GRACE_SECONDS = 180.0


@dataclass
class ManagedProcess:
    agent_id: str
    process: asyncio.subprocess.Process
    launched_at_epoch: float


@dataclass
class SupervisorState:
    request: HarnessRequest
    run_root: str
    root_agent_id: str
    initial_agent_ids: set[str]
    live: dict[str, ManagedProcess]
    launcher: Any
    run_started_at: float
    commenter_tasks: dict[str, asyncio.Task[None]]
    stop_reason: str | None = None
    error: str | None = None
    stop_requested: bool = False
    soft_stop_requested: bool = False
    soft_stop_started_at: float | None = None
    root_last_reviewed_fingerprint: str = ""
    root_finished_refining_at: float | None = None


def _iso_to_epoch(value: str | None) -> float | None:
    if not value:
        return None
    try:
        normalized = value.replace("Z", "+00:00")
        return datetime.fromisoformat(normalized).timestamp()
    except ValueError:
        return None


async def _default_launch_agent_process(run_root: str, agent_id: str) -> asyncio.subprocess.Process:
    # Redirect subprocess stdout/stderr to a per-agent log file so that
    # print/warning output from agent workers doesn't corrupt the TUI.
    log_path = Path(run_root) / "agents" / agent_id / "scratch" / "worker.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(str(log_path), "a", encoding="utf-8")
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "src.harness.agent_worker",
        "--run-root",
        run_root,
        "--agent-id",
        agent_id,
        cwd=str(Path.cwd()),
        stdout=log_fh,
        stderr=log_fh,
    )
    log_fh.close()  # Parent closes its reference; child still has the fd
    return proc


def _resolve_request(request: HarnessRequest) -> HarnessRequest:
    if request.resume_from_run_root:
        stored = load_request(request.resume_from_run_root)
        return stored.model_copy(update={"resume_from_run_root": request.resume_from_run_root})
    if request.available_skill_packs:
        return request
    return request.model_copy(update={"available_skill_packs": ["core", "equity", "macro", "commodity"]})


def _root_skills(registry_map: dict[str, SkillSpec], request: HarnessRequest) -> list[SkillSpec]:
    return visible_skills_for_preset(
        preset=request.root_preset,
        available_skills=get_skills_for_packs(registry_map, request.available_skill_packs or ["core"]),
    )


def _ensure_root_workspace(
    request: HarnessRequest,
    *,
    run_root: str,
    root_agent_id: str,
    registry_map: dict[str, SkillSpec],
) -> None:
    workspace = agent_workspace_paths(run_root, root_agent_id)["workspace"]
    if workspace.exists():
        return
    create_agent_workspace(
        run_root,
        agent_id=root_agent_id,
        parent_id="",
        preset=request.root_preset,
        task_name="Root Task",
        description=request.user_prompt.strip()[:160],
        task_markdown=render_task_markdown(request.user_prompt),
        tools_markdown=render_tools_markdown(
            preset=request.root_preset,
            available_tools=default_tool_allowlist(request.root_preset),
            available_skills=_root_skills(registry_map, request),
        ),
    )


def _root_agent_id(run_root: str) -> str:
    records = latest_agent_records(run_root)
    if "agent_root" in records:
        return "agent_root"
    for record in records.values():
        if not record.parent_id:
            return record.agent_id
    return "agent_root"


def _sync_registry_from_files(run_root: str) -> dict[str, Any]:
    records = latest_agent_records(run_root)
    changed: list[str] = []
    for agent_id, record in records.items():
        status = read_status(run_root, agent_id)
        pid_text = agent_workspace_paths(run_root, agent_id)["pid"].read_text(encoding="utf-8").strip() if agent_workspace_paths(run_root, agent_id)["pid"].exists() else ""
        pid = int(pid_text) if pid_text.isdigit() else None
        if status != record.status or pid != record.pid:
            update_agent_record(
                run_root,
                agent_id=agent_id,
                status=status,
                pid=pid,
                started_at=record.started_at or (datetime.now(timezone.utc).isoformat() if pid else None),
                finished_at=(datetime.now(timezone.utc).isoformat() if status in TERMINAL_STATUSES else record.finished_at),
                error=record.error,
            )
            changed.append(agent_id)
    if changed:
        refresh_progress_view(run_root)
    return {"records": latest_agent_records(run_root), "changed": changed}


async def _terminate_process(managed: ManagedProcess) -> None:
    if managed.process.returncode is not None:
        return
    managed.process.terminate()
    try:
        await asyncio.wait_for(managed.process.wait(), timeout=PROCESS_KILL_GRACE_SECONDS)
    except asyncio.TimeoutError:
        managed.process.kill()
        await managed.process.wait()


async def _cancel_descendants(shared: SupervisorState, *, reason: str) -> None:
    snapshot = latest_agent_records(shared.run_root)
    for agent_id, record in snapshot.items():
        if agent_id == shared.root_agent_id or record.status in TERMINAL_STATUSES:
            continue
        managed = shared.live.pop(agent_id, None)
        if managed is not None:
            await _terminate_process(managed)
        write_status(shared.run_root, agent_id, "cancelled")
        update_agent_record(
            shared.run_root,
            agent_id=agent_id,
            status="cancelled",
            finished_at=datetime.now(timezone.utc).isoformat(),
            error=reason,
        )
        append_event(
            shared.run_root,
            AgentEvent(
                event_type="agent_cancelled",
                agent_id=agent_id,
                parent_id=record.parent_id,
                details={"reason": reason},
            ),
        )


async def _finalize_root_after_forced_stop(shared: SupervisorState) -> None:
    root_status = read_status(shared.run_root, shared.root_agent_id)
    if root_status in TERMINAL_STATUSES:
        return
    managed = shared.live.pop(shared.root_agent_id, None)
    if managed is not None:
        await _terminate_process(managed)
    write_status(shared.run_root, shared.root_agent_id, "failed")
    update_agent_record(
        shared.run_root,
        agent_id=shared.root_agent_id,
        status="failed",
        finished_at=datetime.now(timezone.utc).isoformat(),
        error=shared.error or "Harness run stopped before the root agent completed.",
    )
    append_event(
        shared.run_root,
        AgentEvent(
            event_type="root_stop_forced",
            agent_id=shared.root_agent_id,
            details={"stop_reason": shared.stop_reason or "unknown", "error": shared.error or ""},
        ),
    )


async def _launch_queued_agents(shared: SupervisorState) -> None:
    while not shared.stop_requested:
        if shared.soft_stop_requested:
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
            continue
        snapshot = latest_agent_records(shared.run_root)
        live_agent_ids = {
            agent_id
            for agent_id, record in snapshot.items()
            if record.status in {"running", "waiting"}
        }
        live_agent_ids.update(shared.live)
        for record in snapshot.values():
            if shared.stop_requested:
                break
            if record.status != "queued" or record.agent_id in shared.live:
                continue
            if shared.request.resume_from_run_root and record.agent_id in shared.initial_agent_ids and record.agent_id != shared.root_agent_id:
                continue
            if len(live_agent_ids) >= shared.request.max_live_agents:
                break
            parent_live = sum(
                1
                for agent_id in live_agent_ids
                if snapshot.get(agent_id) is not None and snapshot[agent_id].parent_id == record.parent_id
            )
            if record.parent_id and parent_live >= shared.request.max_live_children_per_parent:
                continue
            process = await shared.launcher(shared.run_root, record.agent_id)
            shared.live[record.agent_id] = ManagedProcess(
                agent_id=record.agent_id,
                process=process,
                launched_at_epoch=time.time(),
            )
            live_agent_ids.add(record.agent_id)
            update_agent_record(
                shared.run_root,
                agent_id=record.agent_id,
                status="running",
                pid=process.pid,
                started_at=datetime.now(timezone.utc).isoformat(),
            )
            append_event(
                shared.run_root,
                AgentEvent(
                    event_type="worker_launched",
                    agent_id=record.agent_id,
                    parent_id=record.parent_id,
                    details={"pid": process.pid},
                ),
            )
        await asyncio.sleep(POLL_INTERVAL_SECONDS)


def _push_child_done_to_parent_queue(run_root: str, child_id: str, status: str, error: str = "") -> None:
    """Append a child_done event to each living parent's events queue."""
    records = latest_agent_records(run_root)
    child_record = records.get(child_id)
    if not child_record or not child_record.parent_id:
        return
    parent_id = child_record.parent_id
    # Skip if parent is already terminal
    parent_status = read_status(run_root, parent_id)
    if parent_status in TERMINAL_STATUSES:
        return
    # Append to parent's queue
    parent_paths = agent_workspace_paths(run_root, parent_id)
    queue_path = parent_paths["events_queue"]
    event = {
        "type": "child_done",
        "child_id": child_id,
        "status": status,
        "error": error,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(queue_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def _read_last_commented_fingerprint(run_root: str, agent_id: str) -> str:
    """Read the last_commented_fingerprint from commenter's state."""
    state = load_commenter_state(run_root, agent_id) or {}
    return str(state.get("last_commented_fingerprint") or "")


STOP_REQUESTED_FILE = "stop_requested"


async def _monitor_agents(shared: SupervisorState) -> None:
    while not shared.stop_requested:
        now_epoch = time.time()
        elapsed = now_epoch - shared.run_started_at

        # Check for an externally-written stop-request sentinel (e.g. from the TUI).
        stop_file = Path(shared.run_root) / STOP_REQUESTED_FILE
        if stop_file.exists() and not shared.soft_stop_requested:
            elapsed = shared.request.wall_clock_budget_seconds  # trigger soft-stop block below

        if elapsed >= shared.request.wall_clock_budget_seconds and not shared.soft_stop_requested:
            shared.soft_stop_requested = True
            shared.soft_stop_started_at = now_epoch
            append_event(
                shared.run_root,
                AgentEvent(
                    event_type="run_soft_stop_requested",
                    agent_id=shared.root_agent_id,
                    details={
                        "reason": "wall_clock_budget_reached" if not stop_file.exists() else "user_requested_stop",
                        "grace_seconds": SOFT_STOP_GRACE_SECONDS,
                    },
                ),
            )
        if shared.soft_stop_requested and shared.soft_stop_started_at is not None:
            if now_epoch - shared.soft_stop_started_at >= SOFT_STOP_GRACE_SECONDS:
                shared.stop_reason = "wall_clock_budget_exhausted"
                shared.error = "Harness wall-clock budget exhausted."
                shared.stop_requested = True
                break

        snapshot = _sync_registry_from_files(shared.run_root)["records"]

        for stale_id in stale_agents(
            shared.run_root,
            stale_after_seconds=shared.request.stale_heartbeat_seconds,
            now_epoch=now_epoch,
        ):
            if read_status(shared.run_root, stale_id) != "stale":
                write_status(shared.run_root, stale_id, "stale")
                update_agent_record(
                    shared.run_root,
                    agent_id=stale_id,
                    status="stale",
                    finished_at=datetime.now(timezone.utc).isoformat(),
                    error="Heartbeat stale.",
                )
                append_event(
                    shared.run_root,
                    AgentEvent(
                        event_type="heartbeat_stale",
                        agent_id=stale_id,
                        details={"heartbeat": read_heartbeat(shared.run_root, stale_id)},
                    ),
                )
                await asyncio.to_thread(
                    _push_child_done_to_parent_queue,
                    shared.run_root,
                    stale_id,
                    "stale",
                    "Heartbeat stale.",
                )
                managed = shared.live.pop(stale_id, None)
                if managed is not None:
                    await _terminate_process(managed)

        for agent_id, managed in list(shared.live.items()):
            process = managed.process
            if process.returncode is None:
                continue
            shared.live.pop(agent_id, None)
            current_status = read_status(shared.run_root, agent_id)
            if current_status not in TERMINAL_STATUSES:
                write_status(shared.run_root, agent_id, "failed")
                update_agent_record(
                    shared.run_root,
                    agent_id=agent_id,
                    status="failed",
                    finished_at=datetime.now(timezone.utc).isoformat(),
                    error=f"Worker exited with code {process.returncode}.",
                )
                terminal_status = "failed"
                terminal_error = f"Worker exited with code {process.returncode}."
            else:
                terminal_status = current_status
                terminal_error = ""
            append_event(
                shared.run_root,
                AgentEvent(
                    event_type="worker_exited",
                    agent_id=agent_id,
                    details={"returncode": process.returncode},
                ),
            )
            await asyncio.to_thread(
                _push_child_done_to_parent_queue,
                shared.run_root,
                agent_id,
                terminal_status,
                terminal_error,
            )

        # Handle root agent completion and refinement loop
        root_status = read_status(shared.run_root, shared.root_agent_id)
        if root_status == "refining" and shared.root_agent_id not in shared.live:
            # Root finished but is waiting for commenter to produce new content
            if shared.root_finished_refining_at is not None:
                elapsed = now_epoch - shared.root_finished_refining_at
                if elapsed >= COMMENTER_REFRESH_INTERVAL_SECONDS:
                    current_fp = _read_last_commented_fingerprint(shared.run_root, shared.root_agent_id)
                    if current_fp != shared.root_last_reviewed_fingerprint:
                        # New comments — re-launch root for another pass
                        shared.root_last_reviewed_fingerprint = current_fp
                        write_status(shared.run_root, shared.root_agent_id, "queued")
                        process = await shared.launcher(shared.run_root, shared.root_agent_id)
                        shared.live[shared.root_agent_id] = ManagedProcess(
                            agent_id=shared.root_agent_id,
                            process=process,
                            launched_at_epoch=now_epoch,
                        )
                        write_status(shared.run_root, shared.root_agent_id, "running")
                        update_agent_record(
                            shared.run_root,
                            agent_id=shared.root_agent_id,
                            status="running",
                            pid=process.pid,
                            started_at=datetime.now(timezone.utc).isoformat(),
                        )
                        shared.root_finished_refining_at = None
                    # else: no new comments yet, keep waiting
        elif root_status in TERMINAL_STATUSES and shared.root_agent_id not in shared.live:
            if root_status == "done" and shared.request.continuous_refinement:
                # Root finished a pass — enter refinement wait state
                fp = _read_last_commented_fingerprint(shared.run_root, shared.root_agent_id)
                shared.root_last_reviewed_fingerprint = fp
                shared.root_finished_refining_at = now_epoch
                write_status(shared.run_root, shared.root_agent_id, "refining")
                # Do NOT stop — wait for commenter to produce new comments
            else:
                shared.stop_reason = root_status
                shared.stop_requested = True
                break

        await asyncio.sleep(POLL_INTERVAL_SECONDS)


async def _run_commenter_refresh(
    shared: SupervisorState,
    agent_id: str,
    observation_snapshot: dict[str, Any],
) -> None:
    try:
        written = await asyncio.to_thread(
            refresh_commenter_for_agent,
            shared.run_root,
            agent_id,
            shared.request,
            observation_snapshot=observation_snapshot,
        )
        append_event(
            shared.run_root,
            AgentEvent(
                event_type="commenter_refreshed",
                agent_id=agent_id,
                details={"comments_written": written},
            ),
        )
    except Exception as exc:
        state = load_commenter_state(shared.run_root, agent_id) or {}
        state["last_attempted_at"] = datetime.now(timezone.utc).isoformat()
        state["last_error"] = f"{type(exc).__name__}: {exc}"
        save_commenter_state(shared.run_root, agent_id, state)
        append_event(
            shared.run_root,
            AgentEvent(
                event_type="commenter_failed",
                agent_id=agent_id,
                details={"error": state["last_error"]},
            ),
        )


async def _monitor_commenters(shared: SupervisorState) -> None:
    while not shared.stop_requested:
        snapshot = latest_agent_records(shared.run_root)
        now_epoch = time.time()

        for agent_id, task in list(shared.commenter_tasks.items()):
            if not task.done():
                continue
            shared.commenter_tasks.pop(agent_id, None)
            with contextlib.suppress(asyncio.CancelledError):
                await task

        for agent_id, record in snapshot.items():
            if record.status in TERMINAL_STATUSES:
                task = shared.commenter_tasks.pop(agent_id, None)
                if task is not None:
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
                continue
            if agent_id in shared.commenter_tasks:
                continue
            fingerprint = compute_commenter_observation_fingerprint(shared.run_root, agent_id)
            state = load_commenter_state(shared.run_root, agent_id) or {}
            last_observed = str(state.get("last_observed_fingerprint") or "")
            if fingerprint != last_observed:
                state["last_observed_fingerprint"] = fingerprint
                state["last_changed_at"] = datetime.now(timezone.utc).isoformat()
                save_commenter_state(shared.run_root, agent_id, state)
            last_attempted_at = (
                _iso_to_epoch(str(state.get("last_attempted_at") or ""))
                or _iso_to_epoch(str(state.get("last_refreshed_at") or ""))
                or 0.0
            )
            last_commented = str(state.get("last_commented_fingerprint") or "")
            # Skip if agent hasn't actually started running yet
            agent_started_at = _iso_to_epoch(record.started_at)
            if agent_started_at is None:
                continue
            if (
                fingerprint
                and fingerprint != last_commented
                and now_epoch - last_attempted_at >= COMMENTER_REFRESH_INTERVAL_SECONDS
                and now_epoch - agent_started_at >= COMMENTER_REFRESH_INTERVAL_SECONDS
            ):
                observation_snapshot = build_commenter_observation_snapshot(
                    shared.run_root,
                    agent_id,
                    base_manifest=state.get("last_commented_manifest"),
                )
                state["last_attempted_at"] = datetime.now(timezone.utc).isoformat()
                save_commenter_state(shared.run_root, agent_id, state)
                shared.commenter_tasks[agent_id] = asyncio.create_task(
                    _run_commenter_refresh(shared, agent_id, observation_snapshot)
                )

        await asyncio.sleep(POLL_INTERVAL_SECONDS)


async def _supervise_async(
    request: HarnessRequest,
    *,
    launch_agent_process: Any = None,
    registry_map: dict[str, SkillSpec] | None = None,
) -> HarnessResponse:
    resolved_request = _resolve_request(request)
    registry = registry_map or build_skill_registry()

    if resolved_request.resume_from_run_root:
        run_root = str(build_run_root(resolved_request))
        root_agent_id = _root_agent_id(run_root)
    else:
        run_root_path, root_agent_id = initialize_run_root(resolved_request)
        run_root = str(run_root_path)
        _ensure_root_workspace(resolved_request, run_root=run_root, root_agent_id=root_agent_id, registry_map=registry)

    launcher = launch_agent_process or _default_launch_agent_process
    initial_agent_ids = set(latest_agent_records(run_root))
    shared = SupervisorState(
        request=resolved_request,
        run_root=run_root,
        root_agent_id=root_agent_id,
        initial_agent_ids=initial_agent_ids,
        live={},
        launcher=launcher,
        run_started_at=time.time(),
        commenter_tasks={},
    )

    launcher_task = asyncio.create_task(_launch_queued_agents(shared))
    monitor_task = asyncio.create_task(_monitor_agents(shared))
    commenter_task = asyncio.create_task(_monitor_commenters(shared))
    await monitor_task
    shared.stop_requested = True
    launcher_task.cancel()
    commenter_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await launcher_task
    with contextlib.suppress(asyncio.CancelledError):
        await commenter_task

    if shared.stop_reason not in TERMINAL_STATUSES:
        await _finalize_root_after_forced_stop(shared)

    await _cancel_descendants(
        shared,
        reason=f"Cancelled because root agent stopped with status '{read_status(shared.run_root, shared.root_agent_id)}'.",
    )

    # Shutdown anything still running.
    for managed in list(shared.live.values()):
        await _terminate_process(managed)
    shared.live.clear()
    for task in list(shared.commenter_tasks.values()):
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    shared.commenter_tasks.clear()

    root_workspace = agent_workspace_paths(run_root, root_agent_id)["workspace"]
    final_report = agent_workspace_paths(run_root, root_agent_id)["publish_final"]
    final_report_exists = final_report.exists()
    stop_reason = shared.stop_reason or "unknown"
    error = shared.error
    if stop_reason == "done" and final_report_exists:
        status = "completed"
    elif stop_reason == "wall_clock_budget_exhausted":
        status = "time_out_with_deliverable" if final_report_exists else "time_out"
    else:
        status = "failed"
    if status == "failed" and error is None:
        error = "Harness run did not finish with a completed root publish/final.md."
    refresh_progress_view(run_root)
    return HarnessResponse(
        status=status,  # type: ignore[arg-type]
        stop_reason=stop_reason,
        run_root=run_root,
        root_agent_path=str(root_workspace),
        final_report_path=str(final_report) if final_report_exists else None,
        error=error,
    )


def run_harness(
    request: HarnessRequest,
    *,
    launch_agent_process: Any = None,
    registry: dict[str, SkillSpec] | None = None,
) -> HarnessResponse:
    """Run the file-based harness kernel until the root agent stops."""

    return asyncio.run(
        _supervise_async(
            request,
            launch_agent_process=launch_agent_process,
            registry_map=registry,
        )
    )
