"""Long-lived agent worker process for the file-based harness kernel."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import os
import re
import threading
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.harness.artifacts import (
    agent_workspace_paths,
    append_event,
    append_transcript_entry,
    latest_agent_records,
    load_commenter_state,
    load_transport_state,
    load_transcript_entries,
    load_request,
    mark_commenter_comments_read,
    read_status,
    remaining_agent_seconds,
    remaining_run_seconds,
    save_skill_state,
    unread_commenter_comments,
    write_heartbeat,
    write_pid,
    write_status,
)
from src.harness.commenter import build_comment_feed_message
from src.harness.commenter import commenter_gate_finished, open_commenter_gate
from src.harness.executor import TERMINAL_STATUSES, create_or_load_session, execute_agent_command, execute_model_tool, model_tool_specs
from src.harness.presets import visible_skills_for_preset
from src.harness.prompt_builder import build_agent_prompt_bundle, build_agent_runtime_delta_prompt
from src.harness.types import AgentCommand, AgentEvent
from src.shared.llm_manager import get_llm
from src.shared.model_config import get_model
from src.harness.transport import (
    _persist_history_compaction_state,
    create_transport,
    preflight_history_compaction,
    resolve_agent_transport,
)


MAX_CONSECUTIVE_ERRORS = 4
MAX_IDLE_RETRIES = 2
TURN_MAX_PROMPT_GAP_SECONDS = 60.0
TURN_PACING_POLL_SECONDS = 1.0
COMMENTER_GATE_POLL_SECONDS = 0.5


@dataclass
class WorkerRuntime:
    request: Any
    record: Any
    session: Any
    model_name: str
    transport_name: str
    llm: Any = None
    transport: Any = None


@dataclass
class WorkerLoopState:
    consecutive_errors: int = 0
    idle_retries: int = 0
    previous_error: str | None = None
    soft_finalize_logged: bool = False
    last_turn_started_monotonic: float | None = None
    last_turn_started_epoch: float | None = None
    last_soft_time_limit_active: bool = False
    successful_turn_index: int = 0
    pending_commenter_gate_id: str | None = None
    native_initial_prompt_sent: bool = False
    soft_stop_delta_sent: bool = False
    final_status_required: bool = False


@dataclass
class TurnPrompt:
    system_prompt: str
    user_prompt: str
    native_user_prompt: str | None
    native_user_prompt_is_initial: bool
    previous_error: str | None
    comment_feed: str | None
    injected_comment_count: int


@dataclass
class NativeTurnPreparation:
    system_prompt: str
    user_prompt: str | None
    user_prompt_is_initial: bool
    tool_specs: list[dict[str, Any]]
    hard_overflow: bool
    previous_error: str | None = None


def _content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content)


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("Agent model returned empty content.")
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, flags=re.DOTALL)
    if fenced:
        stripped = fenced.group(1).strip()
    try:
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Agent response did not include a JSON object.")
    payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Agent JSON payload must be an object.")
    return payload


def _current_prompt_bundle(
    session,
    *,
    transport_name: str,
    previous_error: str | None = None,
    comment_feed: str | None = None,
    soft_stop_active: bool = False,
    show_budget_time: bool = False,
):
    response_mode = "text_json" if transport_name == "text_json" else "native_tools"
    visible_skills = visible_skills_for_preset(
        preset=session.preset,
        available_skills=session.state.available_skills,
    )
    return build_agent_prompt_bundle(
        request=session.request,
        run_root=session.run_root,
        agent_id=session.agent_id,
        preset=session.preset,
        response_mode=response_mode,
        available_tools=session.allowed_tools,
        available_skills=visible_skills,
        previous_error=previous_error,
        comment_feed=comment_feed,
        soft_stop_active=soft_stop_active,
        show_budget_time=show_budget_time,
    )


def _transcript_has_user_message(run_root: str, agent_id: str) -> bool:
    return any(entry.get("kind") == "user_message" for entry in load_transcript_entries(run_root, agent_id))


def _native_pending_user_prompt(
    runtime: WorkerRuntime,
    state: WorkerLoopState,
    *,
    full_user_prompt: str,
    previous_error: str | None,
    comment_feed: str | None,
    soft_stop_active: bool,
) -> tuple[str | None, bool]:
    if not state.native_initial_prompt_sent and not _transcript_has_user_message(
        runtime.session.run_root,
        runtime.session.agent_id,
    ):
        return full_user_prompt, True
    delta_prompt = build_agent_runtime_delta_prompt(
        request=runtime.session.request,
        run_root=runtime.session.run_root,
        agent_id=runtime.session.agent_id,
        previous_error=previous_error,
        comment_feed=comment_feed,
        soft_stop_active=soft_stop_active and not state.soft_stop_delta_sent,
    ).strip()
    return (delta_prompt or None), False


def _root_publish_files_exist(run_root: str, agent_id: str) -> bool:
    paths = agent_workspace_paths(run_root, agent_id)
    return (
        paths["publish_summary"].exists()
        and paths["publish_index"].exists()
        and paths["publish_final"].exists()
    )


def _child_publish_output_exists(run_root: str, agent_id: str) -> bool:
    publish_root = agent_workspace_paths(run_root, agent_id)["publish_root"]
    if not publish_root.exists():
        return False
    for path in publish_root.rglob("*"):
        if path.is_file() and path.stat().st_size > 0:
            return True
    return False


def _publish_outputs_satisfy_completion(run_root: str, agent_id: str) -> bool:
    record = latest_agent_records(run_root).get(agent_id)
    is_root = record is None or not record.parent_id
    if is_root:
        return _root_publish_files_exist(run_root, agent_id)
    return _child_publish_output_exists(run_root, agent_id)


def _heartbeat_loop(run_root: str, agent_id: str, stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        write_heartbeat(run_root, agent_id)
        stop_event.wait(10.0)


def _iso_to_epoch(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _child_has_new_material_since(run_root: str, record, *, since_epoch: float) -> bool:
    paths = agent_workspace_paths(run_root, record.agent_id)
    has_publish_files = False
    publish_root = paths["publish_root"]
    if publish_root.exists():
        for path in publish_root.rglob("*"):
            if not path.is_file():
                continue
            has_publish_files = True
            try:
                if path.stat().st_mtime > since_epoch:
                    return True
            except OSError:
                continue

    status = read_status(run_root, record.agent_id)
    created_epoch = _iso_to_epoch(getattr(record, "created_at", None))
    if created_epoch is not None and created_epoch > since_epoch and status == "queued" and not has_publish_files:
        return False
    try:
        return paths["status"].stat().st_mtime > since_epoch
    except OSError:
        return False


def _has_new_commenter_material_since(run_root: str, agent_id: str, *, since_epoch: float) -> bool:
    unread, _total = unread_commenter_comments(run_root, agent_id)
    for row in unread:
        generated_epoch = _iso_to_epoch(str(row.get("generated_at") or ""))
        if generated_epoch is not None and generated_epoch > since_epoch:
            return True
    return False


def _has_new_external_material_since(run_root: str, agent_id: str, *, since_epoch: float) -> bool:
    for record in latest_agent_records(run_root).values():
        if record.parent_id != agent_id:
            continue
        if _child_has_new_material_since(run_root, record, since_epoch=since_epoch):
            return True
    return _has_new_commenter_material_since(run_root, agent_id, since_epoch=since_epoch)


def _wait_for_next_turn_window(
    run_root: str,
    agent_id: str,
    *,
    request,
    previous_turn_started_monotonic: float,
    previous_turn_started_epoch: float,
    soft_time_limit_was_active: bool,
) -> tuple[bool, bool]:
    max_ready_at = previous_turn_started_monotonic + TURN_MAX_PROMPT_GAP_SECONDS
    while True:
        if read_status(run_root, agent_id) in TERMINAL_STATUSES:
            return False, soft_time_limit_was_active
        now = time.monotonic()
        remaining = min(
            remaining_agent_seconds(request, run_root, agent_id),
            remaining_run_seconds(request, run_root),
        )
        soft_time_limit_active = soft_time_limit_was_active or remaining <= 0
        if now >= max_ready_at:
            return True, soft_time_limit_active
        if soft_time_limit_active != soft_time_limit_was_active:
            return True, soft_time_limit_active
        if _has_new_external_material_since(run_root, agent_id, since_epoch=previous_turn_started_epoch):
            return True, soft_time_limit_active
        sleep_seconds = min(TURN_PACING_POLL_SECONDS, max(0.0, max_ready_at - now))
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)


def _wait_for_commenter_gate(run_root: str, agent_id: str, *, gate_id: str) -> None:
    while True:
        if read_status(run_root, agent_id) in TERMINAL_STATUSES:
            return
        if commenter_gate_finished(load_commenter_state(run_root, agent_id), gate_id):
            return
        time.sleep(COMMENTER_GATE_POLL_SECONDS)


def _record_successful_turn_finished(
    run_root: str,
    agent_id: str,
    *,
    parent_id: str,
    turn_index: int,
    tool_count: int,
    stop_reason: str | None,
) -> str:
    gate = open_commenter_gate(
        run_root,
        agent_id,
        turn_index=turn_index,
        tool_count=tool_count,
        stop_reason=stop_reason,
    )
    gate_id = str(gate["gate_id"])
    append_event(
        run_root,
        AgentEvent(
            event_type="agent_turn_finished",
            agent_id=agent_id,
            parent_id=parent_id,
            details={
                "turn_index": turn_index,
                "tool_count": tool_count,
                "stop_reason": stop_reason or "",
                "commenter_gate_id": gate_id,
                "status_after_turn": read_status(run_root, agent_id),
            },
        ),
    )
    return gate_id


def _load_worker_session(run_root: str, agent_id: str) -> tuple[Any, Any, Any]:
    request = load_request(run_root)
    record = latest_agent_records(run_root).get(agent_id)
    if record is None:
        raise ValueError(f"Unknown agent '{agent_id}'.")
    session = create_or_load_session(
        request=request,
        run_root=run_root,
        agent_id=agent_id,
        preset=record.preset,
    )
    return request, record, session


def _build_worker_runtime(
    run_root: str,
    agent_id: str,
    *,
    request: Any,
    record: Any,
    session: Any,
) -> WorkerRuntime:
    model_name = get_model("harness", "agent")
    transport_name = resolve_agent_transport(request.agent_transport, model_name)
    persisted_transport = load_transport_state(run_root, agent_id)
    if persisted_transport:
        model_name = str(persisted_transport.get("model_name") or model_name)
        transport_name = str(persisted_transport.get("transport") or transport_name)
    prompt_bundle = _current_prompt_bundle(
        session,
        transport_name=transport_name,
        soft_stop_active=False,
    )
    system_prompt = prompt_bundle.system_prompt

    llm = get_llm(model_name) if transport_name == "text_json" else None
    transport = None
    if transport_name != "text_json":
        transport = create_transport(
            transport_name=transport_name,
            run_root=run_root,
            agent_id=agent_id,
            model_name=model_name,
            system_prompt=system_prompt,
        )
        transport.ensure_initialized("")
        transport.update_system_prompt(system_prompt)

    return WorkerRuntime(
        request=request,
        record=record,
        session=session,
        model_name=model_name,
        transport_name=transport_name,
        llm=llm,
        transport=transport,
    )


def _start_worker_lifecycle(
    run_root: str,
    agent_id: str,
    *,
    parent_id: str,
) -> tuple[threading.Event, threading.Thread]:
    stop_event = threading.Event()
    heartbeat_thread = threading.Thread(
        target=_heartbeat_loop,
        args=(run_root, agent_id, stop_event),
        daemon=True,
    )

    write_pid(run_root, agent_id, os.getpid())
    write_status(run_root, agent_id, "running")
    append_event(run_root, AgentEvent(event_type="worker_started", agent_id=agent_id, parent_id=parent_id))
    heartbeat_thread.start()
    return stop_event, heartbeat_thread


def _stop_worker_lifecycle(stop_event: threading.Event, heartbeat_thread: threading.Thread) -> None:
    stop_event.set()
    heartbeat_thread.join(timeout=1.0)


def _wait_before_next_turn(run_root: str, agent_id: str, runtime: WorkerRuntime, state: WorkerLoopState) -> bool:
    if read_status(run_root, agent_id) in TERMINAL_STATUSES:
        return False
    if state.pending_commenter_gate_id is not None:
        _wait_for_commenter_gate(run_root, agent_id, gate_id=state.pending_commenter_gate_id)
        state.pending_commenter_gate_id = None
        if read_status(run_root, agent_id) in TERMINAL_STATUSES:
            return False
    if state.last_turn_started_monotonic is None or state.last_turn_started_epoch is None:
        return True

    should_continue, state.last_soft_time_limit_active = _wait_for_next_turn_window(
        run_root,
        agent_id,
        request=runtime.request,
        previous_turn_started_monotonic=state.last_turn_started_monotonic,
        previous_turn_started_epoch=state.last_turn_started_epoch,
        soft_time_limit_was_active=state.last_soft_time_limit_active,
    )
    if not should_continue:
        return False
    return read_status(run_root, agent_id) not in TERMINAL_STATUSES


def _soft_time_limit_active(run_root: str, agent_id: str, runtime: WorkerRuntime, state: WorkerLoopState) -> bool:
    remaining_agent = remaining_agent_seconds(runtime.request, run_root, agent_id)
    remaining_run = remaining_run_seconds(runtime.request, run_root)
    remaining = min(remaining_agent, remaining_run)
    if remaining <= 0 and not state.soft_finalize_logged:
        append_event(
            run_root,
            AgentEvent(
                event_type="soft_finalize_requested",
                agent_id=agent_id,
                parent_id=runtime.record.parent_id,
                details={"reason": "soft_time_limit_reached"},
            ),
        )
        state.soft_finalize_logged = True
    return remaining <= 0


def _build_turn_prompt(runtime: WorkerRuntime, state: WorkerLoopState, *, soft_stop_active: bool, show_budget_time: bool = False) -> TurnPrompt:
    comment_feed, injected_comment_count = build_comment_feed_message(runtime.session.run_root, runtime.session.agent_id)
    prompt_bundle = _current_prompt_bundle(
        runtime.session,
        transport_name=runtime.transport_name,
        previous_error=state.previous_error,
        comment_feed=comment_feed,
        soft_stop_active=soft_stop_active,
        show_budget_time=show_budget_time,
    )
    native_user_prompt, native_user_prompt_is_initial = (
        _native_pending_user_prompt(
            runtime,
            state,
            full_user_prompt=prompt_bundle.user_prompt,
            previous_error=state.previous_error,
            comment_feed=comment_feed,
            soft_stop_active=soft_stop_active,
        )
        if runtime.transport_name != "text_json"
        else (None, False)
    )
    return TurnPrompt(
        system_prompt=prompt_bundle.system_prompt,
        user_prompt=prompt_bundle.user_prompt,
        native_user_prompt=native_user_prompt,
        native_user_prompt_is_initial=native_user_prompt_is_initial,
        previous_error=state.previous_error,
        comment_feed=comment_feed,
        injected_comment_count=injected_comment_count,
    )


def _mark_successful_turn(
    run_root: str,
    agent_id: str,
    runtime: WorkerRuntime,
    state: WorkerLoopState,
    *,
    tool_count: int,
    stop_reason: str | None,
) -> None:
    state.successful_turn_index += 1
    state.pending_commenter_gate_id = _record_successful_turn_finished(
        run_root,
        agent_id,
        parent_id=runtime.record.parent_id,
        turn_index=state.successful_turn_index,
        tool_count=tool_count,
        stop_reason=stop_reason,
    )
    state.consecutive_errors = 0
    state.idle_retries = 0


def _record_turn_start(state: WorkerLoopState, *, soft_time_limit_active: bool) -> None:
    state.last_turn_started_monotonic = time.monotonic()
    state.last_turn_started_epoch = time.time()
    state.last_soft_time_limit_active = soft_time_limit_active


def _execute_text_json_turn(
    run_root: str,
    agent_id: str,
    runtime: WorkerRuntime,
    state: WorkerLoopState,
    prompt: TurnPrompt,
    *,
    soft_time_limit_active: bool,
) -> None:
    _record_turn_start(state, soft_time_limit_active=soft_time_limit_active)
    response = runtime.llm.invoke(
        [
            SystemMessage(content=prompt.system_prompt),
            HumanMessage(content=prompt.user_prompt),
        ]
    )
    raw_text = _content_to_text(response.content)
    payload = _extract_json_object(raw_text)
    command = AgentCommand.model_validate(payload)
    if prompt.injected_comment_count:
        mark_commenter_comments_read(run_root, agent_id, prompt.injected_comment_count)
    result = execute_agent_command(runtime.session, command)
    append_event(
        run_root,
        AgentEvent(
            event_type="tool_completed",
            agent_id=agent_id,
            parent_id=runtime.record.parent_id,
            details={"tool": command.tool, "result": result},
        ),
    )
    _mark_successful_turn(
        run_root,
        agent_id,
        runtime,
        state,
        tool_count=1,
        stop_reason=None,
    )


def _record_history_compaction_overflow(
    run_root: str,
    agent_id: str,
    runtime: WorkerRuntime,
    *,
    overflow_kind: str,
    estimated_before: int,
    estimated_after: int,
    compacted_user_turns: int,
    previous_error: str | None = None,
) -> None:
    event_type = f"history_compaction_{overflow_kind}_overflow"
    details = {
        "estimated_input_tokens_before": estimated_before,
        "estimated_input_tokens_after": estimated_after,
        "soft_budget_tokens": 170000,
        "hard_context_window_tokens": 200000,
        "compacted_user_turns": compacted_user_turns,
    }
    transcript_entry = {
        "kind": event_type,
        "created_at": datetime.utcnow().isoformat() + "Z",
        **details,
    }
    if previous_error is not None:
        transcript_entry["error"] = previous_error
    append_transcript_entry(
        run_root,
        agent_id,
        transcript_entry,
    )
    append_event(
        run_root,
        AgentEvent(
            event_type=event_type,
            agent_id=agent_id,
            parent_id=runtime.record.parent_id,
            details=details,
        ),
    )


def _prepare_native_turn(
    run_root: str,
    agent_id: str,
    runtime: WorkerRuntime,
    state: WorkerLoopState,
    prompt: TurnPrompt,
    *,
    soft_time_limit_active: bool,
    show_budget_time: bool = False,
) -> NativeTurnPreparation:
    tool_specs = model_tool_specs(runtime.session)
    compaction_changed_any = False
    estimated_before = 0
    estimated_after = 0
    compacted_user_turns = 0
    soft_overflow = False
    hard_overflow = False
    system_prompt = prompt.system_prompt
    user_prompt = prompt.native_user_prompt
    user_prompt_is_initial = prompt.native_user_prompt_is_initial

    while True:
        preflight = preflight_history_compaction(
            transport_name=runtime.transport_name,
            run_root=run_root,
            agent_id=agent_id,
            model_name=runtime.model_name,
            system_prompt=system_prompt,
            pending_user_prompt=user_prompt,
            tool_specs=tool_specs,
        )
        compaction_changed_any = compaction_changed_any or preflight.compaction_changed
        if estimated_before == 0:
            estimated_before = preflight.estimated_input_tokens_before
        estimated_after = preflight.estimated_input_tokens_after
        compacted_user_turns = preflight.compacted_user_turns
        soft_overflow = preflight.soft_overflow
        hard_overflow = preflight.hard_overflow
        if not preflight.compaction_changed:
            break
        prompt_bundle = _current_prompt_bundle(
            runtime.session,
            transport_name=runtime.transport_name,
            previous_error=prompt.previous_error,
            comment_feed=prompt.comment_feed,
            soft_stop_active=soft_time_limit_active,
            show_budget_time=show_budget_time,
        )
        system_prompt = prompt_bundle.system_prompt
        user_prompt, user_prompt_is_initial = _native_pending_user_prompt(
            runtime,
            state,
            full_user_prompt=prompt_bundle.user_prompt,
            previous_error=prompt.previous_error,
            comment_feed=prompt.comment_feed,
            soft_stop_active=soft_time_limit_active,
        )

    _persist_history_compaction_state(
        run_root,
        agent_id,
        compacted_user_turns=compacted_user_turns,
        estimated_input_tokens_before=estimated_before,
        estimated_input_tokens_after=estimated_after,
        compaction_applied=compaction_changed_any,
        soft_overflow=soft_overflow,
        hard_overflow=hard_overflow,
    )
    if soft_overflow:
        _record_history_compaction_overflow(
            run_root,
            agent_id,
            runtime,
            overflow_kind="soft",
            estimated_before=estimated_before,
            estimated_after=estimated_after,
            compacted_user_turns=compacted_user_turns,
        )
    if hard_overflow:
        previous_error = (
            "Next request exceeds the hard input context window even after full transcript compaction. "
            f"Estimated input tokens: {estimated_after}. Hard window: 200000."
        )
        _record_history_compaction_overflow(
            run_root,
            agent_id,
            runtime,
            overflow_kind="hard",
            previous_error=previous_error,
            estimated_before=estimated_before,
            estimated_after=estimated_after,
            compacted_user_turns=compacted_user_turns,
        )
        write_status(run_root, agent_id, "blocked")
        return NativeTurnPreparation(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            user_prompt_is_initial=user_prompt_is_initial,
            tool_specs=tool_specs,
            hard_overflow=True,
            previous_error=previous_error,
        )

    return NativeTurnPreparation(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        user_prompt_is_initial=user_prompt_is_initial,
        tool_specs=tool_specs,
        hard_overflow=False,
    )


def _record_idle_turn(run_root: str, agent_id: str, runtime: WorkerRuntime, state: WorkerLoopState, turn) -> str:
    state.idle_retries += 1
    state.previous_error = "Model returned no tool call. Use an available tool or status."
    append_event(
        run_root,
        AgentEvent(
            event_type="worker_idle",
            agent_id=agent_id,
            parent_id=runtime.record.parent_id,
            details={
                "stop_reason": turn.stop_reason or "",
                "text": "\n".join(turn.text_blocks)[:400],
            },
        ),
    )
    if state.idle_retries > MAX_IDLE_RETRIES:
        write_status(run_root, agent_id, "blocked")
        state.previous_error = "Model repeatedly failed to call a tool."
        return "stop"
    return "continue"


def _execute_model_tool_calls(
    run_root: str,
    agent_id: str,
    runtime: WorkerRuntime,
    tool_calls: list[Any],
) -> list[dict[str, Any]]:
    tool_results: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        try:
            result = execute_model_tool(runtime.session, tool_call.name, tool_call.arguments)
            append_event(
                run_root,
                AgentEvent(
                    event_type="tool_completed",
                    agent_id=agent_id,
                    parent_id=runtime.record.parent_id,
                    details={"tool": tool_call.name, "result": result},
                ),
            )
        except Exception as exc:
            result = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            append_event(
                run_root,
                AgentEvent(
                    event_type="tool_failed",
                    agent_id=agent_id,
                    parent_id=runtime.record.parent_id,
                    details={
                        "tool": tool_call.name,
                        "arguments": tool_call.arguments,
                        "error": result["error"],
                    },
                ),
            )
        tool_results.append(
            {
                "call_id": tool_call.call_id,
                "name": tool_call.name,
                "arguments": tool_call.arguments,
                "result": result,
            }
        )
    return tool_results


def _execute_native_turn(
    run_root: str,
    agent_id: str,
    runtime: WorkerRuntime,
    state: WorkerLoopState,
    prompt: TurnPrompt,
    prepared: NativeTurnPreparation,
    *,
    soft_time_limit_active: bool,
) -> str:
    runtime.transport.update_system_prompt(prepared.system_prompt)
    if prepared.user_prompt:
        runtime.transport.append_user_text(prepared.user_prompt)
        if prepared.user_prompt_is_initial:
            state.native_initial_prompt_sent = True
        if soft_time_limit_active:
            state.soft_stop_delta_sent = True
    _record_turn_start(state, soft_time_limit_active=soft_time_limit_active)
    turn = runtime.transport.execute_turn(prepared.tool_specs)
    if prompt.injected_comment_count:
        mark_commenter_comments_read(run_root, agent_id, prompt.injected_comment_count)
    if not turn.tool_calls:
        return _record_idle_turn(run_root, agent_id, runtime, state, turn)

    tool_results = _execute_model_tool_calls(run_root, agent_id, runtime, turn.tool_calls)
    if tool_results:
        runtime.transport.append_tool_results(tool_results)
    _mark_successful_turn(
        run_root,
        agent_id,
        runtime,
        state,
        tool_count=len(tool_results),
        stop_reason=turn.stop_reason,
    )
    return "completed"


def _handle_worker_turn_error(
    run_root: str,
    agent_id: str,
    runtime: WorkerRuntime,
    state: WorkerLoopState,
    exc: Exception,
) -> bool:
    state.previous_error = f"{type(exc).__name__}: {exc}"
    append_event(
        run_root,
        AgentEvent(
            event_type="worker_error",
            agent_id=agent_id,
            parent_id=runtime.record.parent_id,
            details={"error": state.previous_error},
        ),
    )
    state.consecutive_errors += 1
    if state.consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
        write_status(run_root, agent_id, "failed")
        return False
    return True


def _finalize_worker(run_root: str, agent_id: str, runtime: WorkerRuntime, state: WorkerLoopState) -> int:
    final_status = read_status(run_root, agent_id)
    record = latest_agent_records(run_root).get(agent_id) or runtime.record
    if final_status == "done" and not _publish_outputs_satisfy_completion(run_root, agent_id):
        write_status(run_root, agent_id, "failed")
        final_status = "failed"
        record = latest_agent_records(run_root).get(agent_id) or runtime.record
        if not record.parent_id:
            state.previous_error = "Root agent marked done without publishing summary, artifact index, and final output."
        else:
            state.previous_error = "Child agent marked done without publishing any non-empty file in publish/."
    if final_status in {"failed", "blocked"} and state.previous_error:
        runtime.session.state.last_error = state.previous_error
    save_skill_state(runtime.session.state)
    append_event(
        run_root,
        AgentEvent(
            event_type="worker_finished",
            agent_id=agent_id,
            parent_id=record.parent_id,
            details={"status": final_status, "error": state.previous_error or runtime.session.state.last_error or ""},
        ),
    )
    return 0 if final_status == "done" else 1


def run_agent_worker(run_root: str, agent_id: str) -> int:
    request, record, session = _load_worker_session(run_root, agent_id)
    stop_event, heartbeat_thread = _start_worker_lifecycle(
        run_root,
        agent_id,
        parent_id=record.parent_id,
    )
    runtime = _build_worker_runtime(
        run_root,
        agent_id,
        request=request,
        record=record,
        session=session,
    )
    state = WorkerLoopState()

    try:
        while True:
            if not _wait_before_next_turn(run_root, agent_id, runtime, state):
                break
            soft_time_limit_active = _soft_time_limit_active(run_root, agent_id, runtime, state)
            is_first_turn = not _transcript_has_user_message(run_root, agent_id)
            show_budget_time = is_first_turn or soft_time_limit_active

            try:
                prompt = _build_turn_prompt(runtime, state, soft_stop_active=soft_time_limit_active, show_budget_time=show_budget_time)
                state.previous_error = None
                if runtime.transport_name == "text_json":
                    _execute_text_json_turn(
                        run_root,
                        agent_id,
                        runtime,
                        state,
                        prompt,
                        soft_time_limit_active=soft_time_limit_active,
                    )
                    if read_status(run_root, agent_id) in TERMINAL_STATUSES:
                        break
                    continue

                prepared = _prepare_native_turn(
                    run_root,
                    agent_id,
                    runtime,
                    state,
                    prompt,
                    soft_time_limit_active=soft_time_limit_active,
                    show_budget_time=show_budget_time,
                )
                if prepared.hard_overflow:
                    state.previous_error = prepared.previous_error
                    break
                native_result = _execute_native_turn(
                    run_root,
                    agent_id,
                    runtime,
                    state,
                    prompt,
                    prepared,
                    soft_time_limit_active=soft_time_limit_active,
                )
                if native_result == "continue":
                    # After soft-stop, if the agent published all deliverables but
                    # did not call status("done"), prompt it to do so in a stripped-down turn.
                    if (
                        soft_time_limit_active
                        and not state.final_status_required
                        and _publish_outputs_satisfy_completion(run_root, agent_id)
                        and read_status(run_root, agent_id) != "done"
                    ):
                        state.final_status_required = True
                        runtime.transport._append_system_prompt_snapshot(reason="final-status-required")
                        runtime.transport.append_user_text(
                            "# Final Status Required\n\n"
                            "All required publish files exist. Call `status(status=\"done\")` "
                            "now to mark this task complete. Do not do any further research, "
                            "writing, or delegation. Only call status."
                        )
                    continue
                if native_result == "stop" or read_status(run_root, agent_id) in TERMINAL_STATUSES:
                    break
            except Exception as exc:
                if not _handle_worker_turn_error(run_root, agent_id, runtime, state, exc):
                    break

        return _finalize_worker(run_root, agent_id, runtime, state)
    finally:
        _stop_worker_lifecycle(stop_event, heartbeat_thread)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one harness agent worker.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--agent-id", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return run_agent_worker(args.run_root, args.agent_id)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
