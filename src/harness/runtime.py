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
    registry_paths,
    read_status,
    refresh_progress_view,
    save_commenter_state,
    stale_agents,
    update_agent_record,
    write_status,
    write_text_atomic,
)
from src.harness.commenter import (
    COMMENTER_DEFAULT_DELAY_SECONDS,
    build_commenter_observation_snapshot,
    commenter_gate_payload,
    complete_commenter_gate,
    compute_commenter_observation_fingerprint,
    mark_commenter_gate_running,
    refresh_commenter_for_agent,
)
from src.harness.presets import default_tool_allowlist, visible_skills_for_preset
from src.harness.prompt_builder import render_task_markdown, render_tools_markdown
from src.harness.registry import build_skill_registry, get_skills_for_packs
from src.harness.types import AgentEvent, HarnessRequest, HarnessResponse, SkillSpec


TERMINAL_STATUSES = {
    "done",
    "failed",
    "blocked",
    "stale",
    "cancelled",
    "timed_out",
    "timed_out_with_deliverable",
}
POLL_INTERVAL_SECONDS = 1.0
PROCESS_KILL_GRACE_SECONDS = 2.0
SOFT_STOP_GRACE_SECONDS = 180.0


@dataclass
class ManagedProcess:
    agent_id: str
    process: asyncio.subprocess.Process
    launched_at_epoch: float


@dataclass
class PendingCommenterRefresh:
    agent_id: str
    gate_id: str
    turn_index: int
    due_epoch: float


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
    commenter_schedules: dict[str, PendingCommenterRefresh]
    processed_commenter_gate_ids: set[str]
    stop_reason: str | None = None
    error: str | None = None
    stop_requested: bool = False
    soft_stop_requested: bool = False
    soft_stop_started_at: float | None = None
    root_last_reviewed_fingerprint: str = ""
    root_refinement_gate_id: str | None = None


async def _default_launch_agent_process(run_root: str, agent_id: str) -> asyncio.subprocess.Process:
    # Redirect subprocess stdout/stderr to a per-agent log file so that
    # print/warning output from agent workers doesn't corrupt the TUI.
    log_path = agent_workspace_paths(run_root, agent_id)["worker_log"]
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
        context_files=request.context_files,
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
    final_report = agent_workspace_paths(shared.run_root, shared.root_agent_id)["publish_final"]
    final_exists = final_report.exists()
    if shared.stop_reason == "wall_clock_budget_exhausted":
        terminal_status = "timed_out_with_deliverable" if final_exists else "timed_out"
        default_error = (
            "Harness wall-clock budget exhausted; root published a deliverable before stop."
            if final_exists
            else "Harness wall-clock budget exhausted before the root agent completed."
        )
    else:
        terminal_status = "failed"
        default_error = "Harness run stopped before the root agent completed."
    write_status(shared.run_root, shared.root_agent_id, terminal_status)
    update_agent_record(
        shared.run_root,
        agent_id=shared.root_agent_id,
        status=terminal_status,
        finished_at=datetime.now(timezone.utc).isoformat(),
        error=shared.error or default_error,
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


def _commenter_delay_for_request(request: HarnessRequest) -> float:
    override = request.commenter_interval_seconds
    if override is None:
        return COMMENTER_DEFAULT_DELAY_SECONDS
    return max(0.0, float(override))


STOP_REQUESTED_FILE = "stop_requested"


def _advance_soft_stop(shared: SupervisorState, now_epoch: float) -> bool:
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
            return True
    return False


async def _mark_stale_agents(shared: SupervisorState, now_epoch: float) -> None:
    for stale_id in stale_agents(
        shared.run_root,
        stale_after_seconds=shared.request.stale_heartbeat_seconds,
        now_epoch=now_epoch,
    ):
        if read_status(shared.run_root, stale_id) == "stale":
            continue
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


async def _reap_exited_agents(shared: SupervisorState) -> None:
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


async def _relaunch_root_for_refinement(shared: SupervisorState, now_epoch: float, current_fp: str) -> None:
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


async def _maybe_refine_or_stop_root(shared: SupervisorState, now_epoch: float) -> bool:
    root_status = read_status(shared.run_root, shared.root_agent_id)
    if root_status == "refining" and shared.root_agent_id not in shared.live:
        # Root finished a pass and is waiting for commenter feedback from that pass.
        if shared.root_refinement_gate_id is not None:
            state = load_commenter_state(shared.run_root, shared.root_agent_id) or {}
            gate = commenter_gate_payload(state, shared.root_refinement_gate_id)
            if gate is not None and str(gate.get("status") or "") in {"completed", "failed", "skipped"}:
                current_fp = _read_last_commented_fingerprint(shared.run_root, shared.root_agent_id)
                if int(gate.get("comments_written") or 0) > 0 and current_fp != shared.root_last_reviewed_fingerprint:
                    await _relaunch_root_for_refinement(shared, now_epoch, current_fp)
                shared.root_refinement_gate_id = None
    elif root_status in TERMINAL_STATUSES and shared.root_agent_id not in shared.live:
        if root_status == "done" and shared.request.continuous_refinement:
            # Root finished a pass — enter refinement wait state
            fp = _read_last_commented_fingerprint(shared.run_root, shared.root_agent_id)
            shared.root_last_reviewed_fingerprint = fp
            state = load_commenter_state(shared.run_root, shared.root_agent_id) or {}
            gate = state.get("pending_commenter_gate")
            shared.root_refinement_gate_id = str(gate.get("gate_id") or "") if isinstance(gate, dict) else None
            write_status(shared.run_root, shared.root_agent_id, "refining")
            # Do NOT stop — wait for commenter to produce new comments
        else:
            shared.stop_reason = root_status
            shared.stop_requested = True
            return True
    return False


async def _monitor_agents(shared: SupervisorState) -> None:
    while not shared.stop_requested:
        now_epoch = time.time()
        if _advance_soft_stop(shared, now_epoch):
            break

        _sync_registry_from_files(shared.run_root)
        await _mark_stale_agents(shared, now_epoch)
        await _reap_exited_agents(shared)
        if await _maybe_refine_or_stop_root(shared, now_epoch):
            break

        await asyncio.sleep(POLL_INTERVAL_SECONDS)


async def _run_commenter_refresh(
    shared: SupervisorState,
    agent_id: str,
    observation_snapshot: dict[str, Any],
    *,
    gate_id: str,
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
                details={"comments_written": written, "commenter_gate_id": gate_id},
            ),
        )
        await asyncio.to_thread(
            complete_commenter_gate,
            shared.run_root,
            agent_id,
            gate_id,
            status="completed",
            reason="commenter_refreshed",
            comments_written=written,
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
                details={"error": state["last_error"], "commenter_gate_id": gate_id},
            ),
        )
        await asyncio.to_thread(
            complete_commenter_gate,
            shared.run_root,
            agent_id,
            gate_id,
            status="failed",
            reason="commenter_failed",
            error=state["last_error"],
        )


async def _collect_finished_commenter_tasks(shared: SupervisorState) -> None:
    for agent_id, task in list(shared.commenter_tasks.items()):
        if not task.done():
            continue
        shared.commenter_tasks.pop(agent_id, None)
        with contextlib.suppress(asyncio.CancelledError):
            await task


def _schedule_commenter_gates(shared: SupervisorState, now_epoch: float, commenter_delay: float) -> None:
    events = read_jsonl(registry_paths(shared.run_root)["events_registry"])
    for row in events:
        if row.get("event_type") != "agent_turn_finished":
            continue
        details = row.get("details") if isinstance(row.get("details"), dict) else {}
        gate_id = str(details.get("commenter_gate_id") or "")
        agent_id = str(row.get("agent_id") or "")
        if not gate_id or not agent_id or gate_id in shared.processed_commenter_gate_ids:
            continue
        shared.processed_commenter_gate_ids.add(gate_id)
        state = load_commenter_state(shared.run_root, agent_id) or {}
        gate = commenter_gate_payload(state, gate_id)
        if gate is None or str(gate.get("status") or "") in {"completed", "failed", "skipped"}:
            continue
        turn_index = int(details.get("turn_index") or gate.get("turn_index") or 0)
        shared.commenter_schedules[agent_id] = PendingCommenterRefresh(
            agent_id=agent_id,
            gate_id=gate_id,
            turn_index=turn_index,
            due_epoch=now_epoch + commenter_delay,
        )


async def _skip_commenter_gate(shared: SupervisorState, agent_id: str, gate_id: str, reason: str) -> None:
    await asyncio.to_thread(
        complete_commenter_gate,
        shared.run_root,
        agent_id,
        gate_id,
        status="skipped",
        reason=reason,
    )


async def _start_due_commenter_refreshes(shared: SupervisorState, now_epoch: float) -> None:
    snapshot = latest_agent_records(shared.run_root)
    for agent_id, schedule in list(shared.commenter_schedules.items()):
        if agent_id in shared.commenter_tasks:
            continue
        if now_epoch < schedule.due_epoch:
            continue
        record = snapshot.get(agent_id)
        current_status = read_status(shared.run_root, agent_id)
        root_done_for_refinement = (
            agent_id == shared.root_agent_id
            and current_status == "done"
            and shared.request.continuous_refinement
        )
        if record is None or (current_status in TERMINAL_STATUSES and not root_done_for_refinement):
            await _skip_commenter_gate(shared, agent_id, schedule.gate_id, "agent_terminal")
            shared.commenter_schedules.pop(agent_id, None)
            continue

        fingerprint = compute_commenter_observation_fingerprint(shared.run_root, agent_id)
        state = load_commenter_state(shared.run_root, agent_id) or {}
        last_commented = str(state.get("last_commented_fingerprint") or "")
        if not fingerprint or fingerprint == last_commented:
            await _skip_commenter_gate(shared, agent_id, schedule.gate_id, "no_observation_change")
            append_event(
                shared.run_root,
                AgentEvent(
                    event_type="commenter_skipped",
                    agent_id=agent_id,
                    details={"commenter_gate_id": schedule.gate_id, "reason": "no_observation_change"},
                ),
            )
            shared.commenter_schedules.pop(agent_id, None)
            continue

        observation_snapshot = build_commenter_observation_snapshot(
            shared.run_root,
            agent_id,
            base_manifest=state.get("last_commented_manifest"),
        )
        state["last_attempted_at"] = datetime.now(timezone.utc).isoformat()
        save_commenter_state(shared.run_root, agent_id, state)
        await asyncio.to_thread(mark_commenter_gate_running, shared.run_root, agent_id, schedule.gate_id)
        shared.commenter_schedules.pop(agent_id, None)
        shared.commenter_tasks[agent_id] = asyncio.create_task(
            _run_commenter_refresh(
                shared,
                agent_id,
                observation_snapshot,
                gate_id=schedule.gate_id,
            )
        )


async def _monitor_commenters(shared: SupervisorState) -> None:
    while not shared.stop_requested:
        now_epoch = time.time()
        commenter_delay = _commenter_delay_for_request(shared.request)

        await _collect_finished_commenter_tasks(shared)
        _schedule_commenter_gates(shared, now_epoch, commenter_delay)
        await _start_due_commenter_refreshes(shared, now_epoch)

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
        commenter_schedules={},
        processed_commenter_gate_ids=set(),
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
