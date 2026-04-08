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
    latest_agent_records,
    load_transport_state,
    load_request,
    mark_commenter_comments_read,
    read_status,
    read_text,
    remaining_agent_seconds,
    save_skill_state,
    unread_commenter_comments,
    write_heartbeat,
    write_pid,
    write_status,
)
from src.harness.commenter import build_comment_feed_message
from src.harness.executor import TERMINAL_STATUSES, create_or_load_session, execute_agent_command, execute_model_tool, model_tool_specs
from src.harness.presets import preset_system_prompt_with_context, render_budget_snapshot, visible_skills_for_preset
from src.harness.types import AgentCommand, AgentEvent
from src.shared.llm_manager import get_llm
from src.shared.model_config import get_model
from src.harness.transport import create_transport, resolve_agent_transport


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


def _journal_tail(run_root: str, agent_id: str, limit: int = 8) -> str:
    journal = agent_workspace_paths(run_root, agent_id)["journal"]
    if not journal.exists():
        return "None"
    lines = journal.read_text(encoding="utf-8").splitlines()[-limit:]
    if not lines:
        return "None"
    rendered: list[str] = []
    for raw in lines:
        try:
            item = json.loads(raw)
        except json.JSONDecodeError:
            continue
        rendered.append(
            f"- tool={item.get('tool')} note={item.get('note', '')} result={item.get('result', {})}"
        )
    return "\n".join(rendered) or "None"


def _list_context_files(run_root: str, agent_id: str) -> str:
    context_root = agent_workspace_paths(run_root, agent_id)["context_root"]
    if not context_root.exists():
        return "None"
    files = [str(path) for path in sorted(context_root.rglob("*")) if path.is_file()]
    return "\n".join(f"- {name}" for name in files) or "None"


def _list_publish_files(run_root: str, agent_id: str) -> str:
    publish_root = agent_workspace_paths(run_root, agent_id)["publish_root"]
    if not publish_root.exists():
        return "None"
    rows: list[str] = []
    for path in sorted(publish_root.rglob("*")):
        if not path.is_file():
            continue
        first_line = ""
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line:
                first_line = line[:120]
                break
        rows.append(f"- {path.relative_to(publish_root).as_posix()}: {first_line}")
    return "\n".join(rows) or "None"


def _children_overview(run_root: str, agent_id: str) -> str:
    rows: list[str] = []
    for record in latest_agent_records(run_root).values():
        if record.parent_id != agent_id:
            continue
        child_paths = agent_workspace_paths(run_root, record.agent_id)
        has_summary = "yes" if child_paths["publish_summary"].exists() else "no"
        has_final = "yes" if child_paths["publish_final"].exists() else "no"
        rows.append(
            f"- {record.agent_id} [{record.status}] preset={record.preset} "
            f"desc={record.description} has_summary={has_summary} has_final={has_final} "
            f"summary={child_paths['publish_summary']} index={child_paths['publish_index']}"
        )
    return "\n".join(sorted(rows)) or "None"


def _budget_snapshot(run_root: str, agent_id: str) -> dict[str, int]:
    request = load_request(run_root)
    records = latest_agent_records(run_root)
    live_agents = sum(1 for record in records.values() if record.status in {"running", "waiting"})
    live_children = sum(
        1
        for record in records.values()
        if record.parent_id == agent_id and record.status in {"running", "waiting"}
    )
    return {
        "created_agents": len(records),
        "live_agents": live_agents,
        "remaining_agent_slots": max(0, request.max_agents_per_run - len(records)),
        "remaining_live_child_slots": max(0, request.max_live_children_per_parent - live_children),
    }


def _agent_lineage(run_root: str, agent_id: str) -> tuple[int, str]:
    records = latest_agent_records(run_root)
    depth = 0
    chain: list[str] = []
    current = records.get(agent_id)
    seen: set[str] = set()
    while current is not None and current.agent_id not in seen:
        seen.add(current.agent_id)
        chain.append(f"{current.agent_id}:{current.preset}")
        if not current.parent_id:
            break
        depth += 1
        current = records.get(current.parent_id)
    chain.reverse()
    return depth, " -> ".join(chain) if chain else "None"


def _build_initial_user_prompt(run_root: str, agent_id: str) -> str:
    paths = agent_workspace_paths(run_root, agent_id)
    request = load_request(run_root)
    records = latest_agent_records(run_root)
    current = records.get(agent_id)
    depth, lineage = _agent_lineage(run_root, agent_id)
    budget_snapshot = _budget_snapshot(run_root, agent_id)
    sections = [
        "Task",
        read_text(paths["task"]) or "None",
        "",
        "Allowed Tools",
        read_text(paths["tools"]) or "None",
        "",
        "Agent Identity",
        f"- agent_id: {agent_id}",
        f"- preset: {current.preset if current else 'unknown'}",
        f"- parent_id: {current.parent_id if current and current.parent_id else '-'}",
        f"- root_orchestrator: {'yes' if agent_id == 'agent_root' else 'no'}",
        "",
        render_budget_snapshot(
            request=request,
            snapshot=budget_snapshot,
        ),
        "",
        "Agent Lineage",
        f"- depth: {depth}",
        f"- lineage: {lineage}",
        "",
        "Current Status",
        read_status(run_root, agent_id),
        "",
        "Context Files",
        _list_context_files(run_root, agent_id),
        "- Read listed context file paths with `read_file(path=...)`.",
        "",
        "Published Files",
        _list_publish_files(run_root, agent_id),
        "",
        "Children",
        _children_overview(run_root, agent_id),
    ]
    return "\n".join(sections)


def _build_json_prompt(
    run_root: str,
    agent_id: str,
    previous_error: str | None = None,
    runtime_control: str | None = None,
    comment_feed: str | None = None,
) -> str:
    sections = [
        _build_initial_user_prompt(run_root, agent_id),
        "",
        "Recent Journal",
        _journal_tail(run_root, agent_id),
    ]
    if previous_error:
        sections.extend(["", "Previous Error To Fix", previous_error])
    if comment_feed:
        sections.extend(["", comment_feed])
    if runtime_control:
        sections.extend(["", runtime_control])
    return "\n".join(sections)


def _build_error_prompt(previous_error: str) -> str:
    return "\n".join(
        [
            "Previous Error To Fix",
            previous_error,
            "",
            "Call a valid tool next or set your status explicitly.",
        ]
    )


def _build_runtime_finalize_instruction(run_root: str, agent_id: str) -> str:
    record = latest_agent_records(run_root).get(agent_id)
    is_root = record is None or not record.parent_id
    output_line = (
        "- Write `publish/final.md`, `publish/summary.md`, and `publish/artifact_index.md`, then call `set_status(done)`."
        if is_root
        else "- Write the published output file or files your parent needs, update `publish/summary.md` if helpful, then call `set_status(done)`."
    )
    return "\n".join(
        [
            "Runtime Control",
            "Stop exploring, stop self-critiquing, and generate final output now.",
            "- Do not start new broad research, new decomposition, or new long waits.",
            "- Use the evidence, child outputs, and files you already have.",
            "- Do one more short read or short wait only if it is strictly necessary to finish a supported answer.",
            "- Do not keep rereading or polishing published outputs once they are good enough to ship.",
            "- If something remains uncertain, state that uncertainty explicitly instead of delaying completion.",
            output_line,
        ]
    )


def _current_system_prompt(session, *, transport_name: str) -> str:
    response_mode = "text_json" if transport_name == "text_json" else "native_tools"
    visible_skills = visible_skills_for_preset(
        preset=session.preset,
        available_skills=session.state.available_skills,
    )
    return preset_system_prompt_with_context(
        session.preset,
        response_mode=response_mode,
        available_tools=session.allowed_tools,
        available_skills=visible_skills,
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
    runtime_control_was_active: bool,
) -> tuple[bool, bool]:
    max_ready_at = previous_turn_started_monotonic + TURN_MAX_PROMPT_GAP_SECONDS
    while True:
        if read_status(run_root, agent_id) in TERMINAL_STATUSES:
            return False, runtime_control_was_active
        now = time.monotonic()
        runtime_control_active = runtime_control_was_active or remaining_agent_seconds(request, run_root, agent_id) <= 0
        if now >= max_ready_at:
            return True, runtime_control_active
        if runtime_control_active != runtime_control_was_active:
            return True, runtime_control_active
        if _has_new_external_material_since(run_root, agent_id, since_epoch=previous_turn_started_epoch):
            return True, runtime_control_active
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
    system_prompt = _current_system_prompt(session, transport_name=transport_name)

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
        transport.ensure_initialized(_build_initial_user_prompt(run_root, agent_id))
        transport.update_system_prompt(system_prompt)

    consecutive_errors = 0
    idle_retries = 0
    previous_error: str | None = None
    soft_finalize_requested = False
    runtime_control_persisted = False
    last_turn_started_monotonic: float | None = None
    last_turn_started_epoch: float | None = None
    last_runtime_control_active = False

    try:
        while True:
            status = read_status(run_root, agent_id)
            if status in TERMINAL_STATUSES:
                break
            if last_turn_started_monotonic is not None and last_turn_started_epoch is not None:
                should_continue, last_runtime_control_active = _wait_for_next_turn_window(
                    run_root,
                    agent_id,
                    request=request,
                    previous_turn_started_monotonic=last_turn_started_monotonic,
                    previous_turn_started_epoch=last_turn_started_epoch,
                    runtime_control_was_active=last_runtime_control_active,
                )
                if not should_continue:
                    break
                if read_status(run_root, agent_id) in TERMINAL_STATUSES:
                    break
            remaining_agent = remaining_agent_seconds(request, run_root, agent_id)
            runtime_control: str | None = None
            if remaining_agent <= 0:
                runtime_control = _build_runtime_finalize_instruction(run_root, agent_id)
                if not soft_finalize_requested:
                    append_event(
                        run_root,
                        AgentEvent(
                            event_type="soft_finalize_requested",
                            agent_id=agent_id,
                            parent_id=record.parent_id,
                            details={"reason": "soft_time_limit_reached"},
                        ),
                    )
                    soft_finalize_requested = True
            runtime_control_active = soft_finalize_requested or remaining_agent <= 0

            try:
                system_prompt = _current_system_prompt(session, transport_name=transport_name)
                comment_feed, injected_comment_count = build_comment_feed_message(run_root, agent_id)
                if transport_name == "text_json":
                    last_turn_started_monotonic = time.monotonic()
                    last_turn_started_epoch = time.time()
                    last_runtime_control_active = runtime_control_active
                    response = llm.invoke(
                        [
                            SystemMessage(content=system_prompt),
                            HumanMessage(
                                content=_build_json_prompt(
                                    run_root,
                                    agent_id,
                                    previous_error,
                                    runtime_control if soft_finalize_requested else None,
                                    comment_feed,
                                )
                            ),
                        ]
                    )
                    previous_error = None
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

                extra_user_messages: list[str] = []
                if previous_error:
                    extra_user_messages.append(_build_error_prompt(previous_error))
                    previous_error = None
                if comment_feed:
                    extra_user_messages.append(comment_feed)
                if runtime_control and (not runtime_control_persisted or extra_user_messages):
                    extra_user_messages.append(runtime_control)
                    runtime_control_persisted = True
                if transport is not None:
                    for message in extra_user_messages:
                        transport.append_user_text(message)

                transport.update_system_prompt(system_prompt)
                last_turn_started_monotonic = time.monotonic()
                last_turn_started_epoch = time.time()
                last_runtime_control_active = runtime_control_active
                turn = transport.execute_turn(model_tool_specs(session))
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
