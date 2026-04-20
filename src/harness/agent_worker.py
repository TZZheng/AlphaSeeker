"""Long-lived agent worker process for the file-based harness kernel."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
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
    load_transport_state,
    load_request,
    mark_commenter_comments_read,
    read_status,
    remaining_agent_seconds,
    save_skill_state,
    unread_commenter_comments,
    write_heartbeat,
    write_pid,
    write_status,
)
from src.harness.commenter import build_comment_feed_message
from src.harness.executor import TERMINAL_STATUSES, create_or_load_session, execute_agent_command, execute_model_tool, model_tool_specs
from src.harness.presets import visible_skills_for_preset
from src.harness.prompt_builder import build_agent_prompt_bundle
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
    )


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
        soft_time_limit_active = soft_time_limit_was_active or remaining_agent_seconds(request, run_root, agent_id) <= 0
        if now >= max_ready_at:
            return True, soft_time_limit_active
        if soft_time_limit_active != soft_time_limit_was_active:
            return True, soft_time_limit_active
        if _has_new_external_material_since(run_root, agent_id, since_epoch=previous_turn_started_epoch):
            return True, soft_time_limit_active
        sleep_seconds = min(TURN_PACING_POLL_SECONDS, max(0.0, max_ready_at - now))
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)


def run_agent_worker(run_root: str, agent_id: str) -> int:
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

    stop_event = threading.Event()
    heartbeat_thread = threading.Thread(
        target=_heartbeat_loop,
        args=(run_root, agent_id, stop_event),
        daemon=True,
    )

    write_pid(run_root, agent_id, os.getpid())
    write_status(run_root, agent_id, "running")
    append_event(run_root, AgentEvent(event_type="worker_started", agent_id=agent_id, parent_id=record.parent_id))
    heartbeat_thread.start()

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

    consecutive_errors = 0
    idle_retries = 0
    previous_error: str | None = None
    soft_finalize_logged = False
    last_turn_started_monotonic: float | None = None
    last_turn_started_epoch: float | None = None
    last_soft_time_limit_active = False

    try:
        while True:
            status = read_status(run_root, agent_id)
            if status in TERMINAL_STATUSES:
                break
            if last_turn_started_monotonic is not None and last_turn_started_epoch is not None:
                should_continue, last_soft_time_limit_active = _wait_for_next_turn_window(
                    run_root,
                    agent_id,
                    request=request,
                    previous_turn_started_monotonic=last_turn_started_monotonic,
                    previous_turn_started_epoch=last_turn_started_epoch,
                    soft_time_limit_was_active=last_soft_time_limit_active,
                )
                if not should_continue:
                    break
                if read_status(run_root, agent_id) in TERMINAL_STATUSES:
                    break
            remaining_agent = remaining_agent_seconds(request, run_root, agent_id)
            if remaining_agent <= 0:
                if not soft_finalize_logged:
                    append_event(
                        run_root,
                        AgentEvent(
                            event_type="soft_finalize_requested",
                            agent_id=agent_id,
                            parent_id=record.parent_id,
                            details={"reason": "soft_time_limit_reached"},
                        ),
                    )
                    soft_finalize_logged = True
            soft_time_limit_active = remaining_agent <= 0

            try:
                comment_feed, injected_comment_count = build_comment_feed_message(run_root, agent_id)
                prompt_bundle = _current_prompt_bundle(
                    session,
                    transport_name=transport_name,
                    previous_error=previous_error,
                    comment_feed=comment_feed,
                    soft_stop_active=soft_time_limit_active,
                )
                system_prompt = prompt_bundle.system_prompt
                user_prompt = prompt_bundle.user_prompt
                previous_error = None
                if transport_name == "text_json":
                    last_turn_started_monotonic = time.monotonic()
                    last_turn_started_epoch = time.time()
                    last_soft_time_limit_active = soft_time_limit_active
                    response = llm.invoke(
                        [
                            SystemMessage(content=system_prompt),
                            HumanMessage(content=user_prompt),
                        ]
                    )
                    raw_text = _content_to_text(response.content)
                    payload = _extract_json_object(raw_text)
                    command = AgentCommand.model_validate(payload)
                    if injected_comment_count:
                        mark_commenter_comments_read(run_root, agent_id, injected_comment_count)
                    result = execute_agent_command(session, command)
                    append_event(
                        run_root,
                        AgentEvent(
                            event_type="tool_completed",
                            agent_id=agent_id,
                            parent_id=record.parent_id,
                            details={"tool": command.tool, "result": result},
                        ),
                    )
                    consecutive_errors = 0
                    idle_retries = 0
                    if read_status(run_root, agent_id) in TERMINAL_STATUSES:
                        break
                    continue

                tool_specs = model_tool_specs(session)
                compaction_changed_any = False
                estimated_before = 0
                estimated_after = 0
                compacted_user_turns = 0
                soft_overflow = False
                hard_overflow = False
                while True:
                    preflight = preflight_history_compaction(
                        transport_name=transport_name,
                        run_root=run_root,
                        agent_id=agent_id,
                        model_name=model_name,
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
                        session,
                        transport_name=transport_name,
                        previous_error=previous_error,
                        comment_feed=comment_feed,
                        soft_stop_active=soft_time_limit_active,
                    )
                    system_prompt = prompt_bundle.system_prompt
                    user_prompt = prompt_bundle.user_prompt

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
                    append_transcript_entry(
                        run_root,
                        agent_id,
                        {
                            "kind": "history_compaction_soft_overflow",
                            "created_at": datetime.utcnow().isoformat() + "Z",
                            "estimated_input_tokens_before": estimated_before,
                            "estimated_input_tokens_after": estimated_after,
                            "soft_budget_tokens": 170000,
                            "hard_context_window_tokens": 200000,
                            "compacted_user_turns": compacted_user_turns,
                        },
                    )
                    append_event(
                        run_root,
                        AgentEvent(
                            event_type="history_compaction_soft_overflow",
                            agent_id=agent_id,
                            parent_id=record.parent_id,
                            details={
                                "estimated_input_tokens_before": estimated_before,
                                "estimated_input_tokens_after": estimated_after,
                                "soft_budget_tokens": 170000,
                                "hard_context_window_tokens": 200000,
                                "compacted_user_turns": compacted_user_turns,
                            },
                        ),
                    )
                if hard_overflow:
                    previous_error = (
                        "Next request exceeds the hard input context window even after full transcript compaction. "
                        f"Estimated input tokens: {estimated_after}. Hard window: 200000."
                    )
                    append_transcript_entry(
                        run_root,
                        agent_id,
                        {
                            "kind": "history_compaction_hard_overflow",
                            "created_at": datetime.utcnow().isoformat() + "Z",
                            "estimated_input_tokens_before": estimated_before,
                            "estimated_input_tokens_after": estimated_after,
                            "soft_budget_tokens": 170000,
                            "hard_context_window_tokens": 200000,
                            "compacted_user_turns": compacted_user_turns,
                            "error": previous_error,
                        },
                    )
                    append_event(
                        run_root,
                        AgentEvent(
                            event_type="history_compaction_hard_overflow",
                            agent_id=agent_id,
                            parent_id=record.parent_id,
                            details={
                                "estimated_input_tokens_before": estimated_before,
                                "estimated_input_tokens_after": estimated_after,
                                "soft_budget_tokens": 170000,
                                "hard_context_window_tokens": 200000,
                                "compacted_user_turns": compacted_user_turns,
                            },
                        ),
                    )
                    write_status(run_root, agent_id, "blocked")
                    break

                transport.update_system_prompt(system_prompt)
                transport.append_user_text(user_prompt)
                last_turn_started_monotonic = time.monotonic()
                last_turn_started_epoch = time.time()
                last_soft_time_limit_active = soft_time_limit_active
                turn = transport.execute_turn(tool_specs)
                if injected_comment_count:
                    mark_commenter_comments_read(run_root, agent_id, injected_comment_count)
                if not turn.tool_calls:
                    idle_retries += 1
                    previous_error = "Model returned no tool call. Use an available tool or set_status."
                    append_event(
                        run_root,
                        AgentEvent(
                            event_type="worker_idle",
                            agent_id=agent_id,
                            parent_id=record.parent_id,
                            details={
                                "stop_reason": turn.stop_reason or "",
                                "text": "\n".join(turn.text_blocks)[:400],
                            },
                        ),
                    )
                    if idle_retries > MAX_IDLE_RETRIES:
                        write_status(run_root, agent_id, "blocked")
                        previous_error = "Model repeatedly failed to call a tool."
                        break
                    continue

                tool_results: list[dict[str, Any]] = []
                for tool_call in turn.tool_calls:
                    try:
                        result = execute_model_tool(session, tool_call.name, tool_call.arguments)
                        append_event(
                            run_root,
                            AgentEvent(
                                event_type="tool_completed",
                                agent_id=agent_id,
                                parent_id=record.parent_id,
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
                                parent_id=record.parent_id,
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
                if transport is not None and tool_results:
                    transport.append_tool_results(tool_results)
                consecutive_errors = 0
                idle_retries = 0
                if read_status(run_root, agent_id) in TERMINAL_STATUSES:
                    break
            except Exception as exc:
                previous_error = f"{type(exc).__name__}: {exc}"
                append_event(
                    run_root,
                    AgentEvent(
                        event_type="worker_error",
                        agent_id=agent_id,
                        parent_id=record.parent_id,
                        details={"error": previous_error},
                    ),
                )
                consecutive_errors += 1
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    write_status(run_root, agent_id, "failed")
                    break

        final_status = read_status(run_root, agent_id)
        if final_status == "done" and not _publish_outputs_satisfy_completion(run_root, agent_id):
            write_status(run_root, agent_id, "failed")
            final_status = "failed"
            record = latest_agent_records(run_root).get(agent_id)
            if record is None or not record.parent_id:
                previous_error = "Root agent marked done without publishing summary, artifact index, and final output."
            else:
                previous_error = "Child agent marked done without publishing any non-empty file in publish/."
        if final_status in {"failed", "blocked"} and previous_error:
            session.state.last_error = previous_error
        save_skill_state(session.state)
        append_event(
            run_root,
            AgentEvent(
                event_type="worker_finished",
                agent_id=agent_id,
                parent_id=record.parent_id,
                details={"status": final_status, "error": previous_error or session.state.last_error or ""},
            ),
        )
        return 0 if final_status == "done" else 1
    finally:
        stop_event.set()
        heartbeat_thread.join(timeout=1.0)


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
