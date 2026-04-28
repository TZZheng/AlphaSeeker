"""File-first tool backend for harness agents."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any
from uuid import uuid4

from src.harness.artifacts import (
    agent_workspace_paths,
    append_tool_call_log,
    append_event,
    build_reduction_paths,
    create_agent_workspace,
    latest_agent_records,
    load_skill_state,
    load_object_manifest,
    read_jsonl,
    read_status,
    read_text,
    refresh_progress_view,
    save_skill_state,
    snapshot_final_report_if_changed,
    write_status,
    write_text_atomic,
)
from src.harness.presets import default_tool_allowlist, visible_skills_for_preset
from src.harness.prompt_builder import render_tools_markdown
from src.harness.registry import build_skill_registry, get_skills_for_packs
from src.harness.skills.common import json_preview
from src.harness.tool_catalog import LEGAL_PRESET_LIST as _LEGAL_PRESET_LIST
from src.harness.tool_catalog import tool_specs_for_names
from src.harness.types import (
    AGENT_PRESETS,
    AgentCommand,
    AgentEvent,
    HarnessRequest,
    HarnessState,
    Observation,
    SkillResult,
    SkillSpec,
)
from src.harness.visibility import (
    VisibilityError,
    default_search_targets,
    resolve_visible_read_file,
    resolve_visible_search_target,
    resolve_visible_write_file,
    visible_workspace_entries,
)


TERMINAL_STATUSES = {"done", "failed", "blocked", "stale", "cancelled"}

BASH_ALLOWED_COMMANDS = {"cp", "mv", "mkdir", "ls", "rg", "sleep"}
BASH_DEFAULT_TIMEOUT_SECONDS = 10
BASH_DEFAULT_MAX_OUTPUT_CHARS = 12000
PATCH_BEGIN_MARKER = "*** Begin Patch"
PATCH_END_MARKER = "*** End Patch"
PATCH_UPDATE_FILE_PREFIX = "*** Update File: "
PATCH_TRUNCATION_MARKER_RE = re.compile(r"\.\.\. \[\d+ chars\]")


@dataclass
class AgentSession:
    request: HarnessRequest
    run_root: str
    agent_id: str
    preset: str
    allowed_tools: list[str]
    registry_map: dict[str, SkillSpec]
    state: HarnessState

    @property
    def workspace(self) -> Path:
        return agent_workspace_paths(self.run_root, self.agent_id)["workspace"]


@dataclass(frozen=True)
class PatchHunk:
    entries: list[tuple[str, str]]


@dataclass(frozen=True)
class ParsedPatch:
    path: str
    hunks: list[PatchHunk]


def create_or_load_session(
    *,
    request: HarnessRequest,
    run_root: str,
    agent_id: str,
    preset: str,
    registry_map: dict[str, SkillSpec] | None = None,
) -> AgentSession:
    registry = registry_map or build_skill_registry()
    enabled_packs = request.available_skill_packs or ["core"]
    allowed_skills = get_skills_for_packs(registry, enabled_packs)
    state = load_skill_state(run_root, agent_id)
    if state is None:
        workspace = agent_workspace_paths(run_root, agent_id)["workspace"]
        state = HarnessState(
            request=request,
            run_id=Path(run_root).name,
            run_root=run_root,
            agent_id=agent_id,
            workspace_path=str(workspace),
            dossier_paths=build_reduction_paths(workspace),
            enabled_packs=enabled_packs,
            available_skills=allowed_skills,
        )
        save_skill_state(state)
    else:
        state.request = request
        state.run_root = run_root
        state.agent_id = agent_id
        state.available_skills = allowed_skills
        state.enabled_packs = enabled_packs
    return AgentSession(
        request=request,
        run_root=run_root,
        agent_id=agent_id,
        preset=preset,
        allowed_tools=default_tool_allowlist(preset),
        registry_map=registry,
        state=state,
    )


def _normalize_preset_name(raw_preset: str) -> str:
    return raw_preset.strip().lower().replace("-", "_").replace(" ", "_")


def _agent_budget_snapshot(session: AgentSession) -> dict[str, int]:
    records = latest_agent_records(session.run_root)
    live_agents = [
        record
        for record in records.values()
        if record.status in {"running", "waiting"}
    ]
    live_children = [
        record
        for record in records.values()
        if record.parent_id == session.agent_id and record.status in {"running", "waiting"}
    ]
    return {
        "created_agents": len(records),
        "remaining_agent_slots": max(0, session.request.max_agents_per_run - len(records)),
        "queued_agents": sum(1 for record in records.values() if record.status == "queued"),
        "live_agents": len(live_agents),
        "remaining_live_agent_slots": max(0, session.request.max_live_agents - len(live_agents)),
        "live_children_for_parent": len(live_children),
        "remaining_live_child_slots": max(
            0,
            session.request.max_live_children_per_parent - len(live_children),
        ),
    }


def model_tool_specs(session: AgentSession) -> list[dict[str, Any]]:
    visible_skills = visible_skills_for_preset(
        preset=session.preset,
        available_skills=session.state.available_skills,
    )
    return tool_specs_for_names(
        [*session.allowed_tools, *(spec.name for spec in visible_skills)],
        available_skills=visible_skills,
    )


def execute_agent_command(session: AgentSession, command: AgentCommand) -> dict[str, Any]:
    if command.tool not in session.allowed_tools:
        raise ValueError(f"Tool '{command.tool}' is not allowed for preset '{session.preset}'.")
    handler = _HANDLERS.get(command.tool)
    if handler is None:
        raise ValueError(f"Unknown tool '{command.tool}'.")
    result = handler(session, command.arguments)
    _snapshot_final_report_after_tool(session, command.tool, result)
    _append_tool_call(
        session,
        {
            "tool": command.tool,
            "arguments": command.arguments,
            "result": result,
            "note": command.note,
            "created_at": _now_iso(),
        },
    )
    save_skill_state(session.state)
    refresh_progress_view(session.run_root)
    return result


def execute_model_tool(session: AgentSession, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if tool_name in _HANDLERS:
        if tool_name not in session.allowed_tools:
            raise ValueError(f"Tool '{tool_name}' is not allowed for preset '{session.preset}'.")
        result = _HANDLERS[tool_name](session, arguments)
    else:
        visible_skills = {spec.name: spec for spec in visible_skills_for_preset(preset=session.preset, available_skills=session.state.available_skills)}
        if tool_name not in visible_skills:
            raise ValueError(f"Tool '{tool_name}' is not available for preset '{session.preset}'.")
        result = _run_skill(session, tool_name, arguments)

    tool_call_row = {
        "tool": tool_name,
        "arguments": arguments,
        "result": result,
        "created_at": _now_iso(),
    }
    _snapshot_final_report_after_tool(session, tool_name, result)
    append_tool_call_log(session.run_root, session.agent_id, tool_call_row)
    save_skill_state(session.state)
    refresh_progress_view(session.run_root)
    return result


def _snapshot_final_report_after_tool(session: AgentSession, tool_name: str, result: dict[str, Any]) -> None:
    trigger_operation = ""
    if isinstance(result, dict):
        trigger_operation = str(result.get("operation") or "")
    try:
        snapshot_final_report_if_changed(
            session.run_root,
            session.agent_id,
            trigger={"trigger_tool": tool_name, "trigger_operation": trigger_operation},
        )
    except Exception as exc:
        with contextlib.suppress(Exception):
            append_event(
                session.run_root,
                AgentEvent(
                    event_type="final_report_snapshot_failed",
                    agent_id=session.agent_id,
                    details={
                        "trigger_tool": tool_name,
                        "trigger_operation": trigger_operation,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                ),
            )


def _append_tool_call(session: AgentSession, payload: dict[str, Any]) -> None:
    append_tool_call_log(session.run_root, session.agent_id, payload)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _record_observation(session: AgentSession, source: str, summary: str) -> None:
    session.state.observations.append(
        Observation(
            id=f"O{len(session.state.observations) + 1}",
            source=source,
            summary=summary,
        )
    )


def _assign_evidence_ids(session: AgentSession, result: SkillResult) -> None:
    next_index = len(session.state.evidence_ledger) + 1
    for offset, item in enumerate(result.evidence):
        if not item.id:
            item.id = f"E{next_index + offset}"


def _skill_output_root(session: AgentSession, skill_name: str) -> Path:
    root = agent_workspace_paths(session.run_root, session.agent_id)["skills_artifacts_root"]
    root.mkdir(parents=True, exist_ok=True)
    index = len(session.state.skill_history) + 1
    dest = root / f"{index:03d}_{skill_name}"
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def _run_skill(session: AgentSession, skill_name: str, skill_args: dict[str, Any]) -> dict[str, Any]:
    spec = session.registry_map.get(skill_name)
    if spec is None or spec.executor is None:
        raise ValueError(f"Unknown skill '{skill_name}'.")

    result = spec.executor(dict(skill_args), session.state)
    _assign_evidence_ids(session, result)
    session.state.skill_history.append(result)
    session.state.evidence_ledger.extend(result.evidence)
    session.state.last_error = result.error
    _record_observation(session, f"skill:{skill_name}", result.summary)

    output_root = _skill_output_root(session, skill_name)
    if result.output_text:
        output_path = output_root / "output.md"
        write_text_atomic(output_path, result.output_text)
    summary_path = output_root / "summary.md"
    write_text_atomic(summary_path, result.summary + "\n")
    details_path = output_root / "details.json"
    write_text_atomic(details_path, json_preview(result.details))
    if result.evidence:
        evidence_file = output_root / "evidence.json"
        write_text_atomic(
            evidence_file,
            json.dumps([item.model_dump(mode="json") for item in result.evidence], indent=2, ensure_ascii=True),
        )
    if result.artifacts:
        artifact_manifest = output_root / "artifacts.txt"
        write_text_atomic(artifact_manifest, "\n".join(result.artifacts) + "\n")

    append_event(
        session.run_root,
        AgentEvent(
            event_type="skill_executed",
            agent_id=session.agent_id,
            details={
                "skill_name": skill_name,
                "status": result.status,
                "summary": result.summary,
                "output_root": str(output_root),
            },
        ),
    )
    response: dict[str, Any] = {
        "skill_name": skill_name,
        "status": result.status,
        "summary": result.summary,
    }
    if skill_name == "search_web":
        response["results"] = result.details.get("results", [])
        results_path = result.details.get("results_path")
        if results_path:
            response["results_path"] = results_path
    elif result.output_text:
        response["content"] = result.output_text
    if result.artifacts:
        response["artifact_paths"] = list(result.artifacts)
    if result.error:
        response["error"] = result.error
    return response


def _render_child_task_markdown(
    task_name: str,
    description: str,
    instructions: str,
    expected_publish_files: list[str],
) -> str:
    output_lines = [f"- `{name}`" for name in expected_publish_files if name.strip()]
    if not output_lines:
        output_lines = ["- Use the published file or files your parent asked for."]
    return "\n".join(
        [
            "# Task Assignment",
            "",
            "## Task Name",
            task_name.strip(),
            "",
            "## One-Line Goal",
            description.strip(),
            "",
            "## Instructions",
            instructions.strip() or description.strip(),
            "",
            "## Expected Published Outputs",
            *output_lines,
            "",
            "## Success Criteria",
            "- Publish early progress to `publish/summary.md` if the task is long-running.",
            "- Publish the main child result to the file or files listed above.",
            "- `publish/summary.md` and `publish/artifact_index.md` are recommended because they help the parent synthesize or recover quickly.",
        ]
    )


def _handle_spawn_subagent(session: AgentSession, arguments: dict[str, Any]) -> dict[str, Any]:
    records = latest_agent_records(session.run_root)
    if len(records) >= session.request.max_agents_per_run:
        raise ValueError(
            f"Run-wide agent budget exhausted: {len(records)}/{session.request.max_agents_per_run} agents already exist."
        )

    task_name = str(arguments.get("task_name") or "Child Task").strip()
    description = str(arguments.get("description") or task_name).strip()
    raw_preset = str(arguments.get("preset") or "research").strip()
    preset = _normalize_preset_name(raw_preset)
    if preset not in AGENT_PRESETS:
        raise ValueError(
            f"Unknown child preset '{raw_preset}'. Legal presets: {_LEGAL_PRESET_LIST}."
        )
    instructions = str(arguments.get("instructions") or "").strip()
    context_files = [str(item) for item in arguments.get("context_files") or []]
    expected_publish_files = [str(item).strip() for item in arguments.get("expected_publish_files") or [] if str(item).strip()]
    agent_id = f"agent_{uuid4().hex[:8]}"
    child_task = str(
        arguments.get("task_markdown")
        or _render_child_task_markdown(task_name, description, instructions, expected_publish_files)
    )
    child_registry = visible_skills_for_preset(
        preset=preset,
        available_skills=get_skills_for_packs(session.registry_map, session.request.available_skill_packs or ["core"]),
    )
    tools_markdown = render_tools_markdown(
        preset=preset,
        available_tools=default_tool_allowlist(preset),
        available_skills=child_registry,
    )

    create_agent_workspace(
        session.run_root,
        agent_id=agent_id,
        parent_id=session.agent_id,
        preset=preset,
        task_name=task_name,
        description=description,
        task_markdown=child_task,
        tools_markdown=tools_markdown,
        context_files=context_files,
    )
    append_event(
        session.run_root,
        AgentEvent(
            event_type="spawn_requested",
            agent_id=agent_id,
            parent_id=session.agent_id,
            details={"preset": preset, "description": description},
        ),
    )
    return {
        "agent_id": agent_id,
        "preset": preset,
        "description": description,
        "status": "queued",
        "summary_path": str(agent_workspace_paths(session.run_root, agent_id)["publish_summary"]),
        "artifact_index_path": str(agent_workspace_paths(session.run_root, agent_id)["publish_index"]),
        "final_path": str(agent_workspace_paths(session.run_root, agent_id)["publish_final"]),
        "expected_publish_files": expected_publish_files,
        "budget": _agent_budget_snapshot(session),
    }


def _child_rows(session: AgentSession) -> list[dict[str, Any]]:
    promoted_by_agent: dict[str, list[dict[str, str]]] = {}
    for row in load_object_manifest(session.run_root):
        agent_id = str(row.get("agent_id") or "")
        if not agent_id:
            continue
        promoted_by_agent.setdefault(agent_id, []).append(
            {
                "object_id": str(row.get("object_id") or ""),
                "path": str(row.get("object_path") or ""),
                "description": str(row.get("description") or ""),
            }
        )
    rows: list[dict[str, Any]] = []
    for record in latest_agent_records(session.run_root).values():
        if record.parent_id != session.agent_id:
            continue
        paths = agent_workspace_paths(session.run_root, record.agent_id)
        publish_files = _publish_file_rows(session.run_root, record.agent_id)
        rows.append(
            {
                "agent_id": record.agent_id,
                "preset": record.preset,
                "status": record.status,
                "description": record.description,
                "summary_path": str(paths["publish_summary"]),
                "artifact_index_path": str(paths["publish_index"]),
                "final_path": str(paths["publish_final"]),
                "has_summary": paths["publish_summary"].exists(),
                "has_artifact_index": paths["publish_index"].exists(),
                "has_final": paths["publish_final"].exists(),
                "publish_files": publish_files,
                "promoted_artifacts": promoted_by_agent.get(record.agent_id, []),
                "summary_excerpt": _describe_file(paths["publish_summary"]),
                "error": record.error or "",
            }
        )
    return sorted(rows, key=lambda item: item["agent_id"])


def _handle_list_children(session: AgentSession, _arguments: dict[str, Any]) -> dict[str, Any]:
    # 1. Drain the events queue
    queue_path = agent_workspace_paths(session.run_root, session.agent_id)["events_queue"]
    queued_events = read_jsonl(queue_path)
    write_text_atomic(queue_path, "")  # clear queue

    # 2. Get all children from registry
    all_children = _child_rows(session)

    # 3. Build result: completed (from queue) + running
    completed = []
    for event in queued_events:
        child_id = event["child_id"]
        row = next((r for r in all_children if r["agent_id"] == child_id), None)
        if row:
            completed.append({**row, "just_completed": True})
        else:
            # Child already removed from registry, use event data directly
            completed.append({
                "agent_id": child_id,
                "status": event["status"],
                "error": event.get("error", ""),
                "just_completed": True,
            })

    running = [r for r in all_children if r["status"] not in TERMINAL_STATUSES]

    return {
        "children": completed + running,
        "completed_count": len(completed),
        "running_count": len(running),
        "budget": _agent_budget_snapshot(session),
    }


def _describe_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line:
            return line[:160]
    return ""


def _publish_file_rows(run_root: str, agent_id: str) -> list[dict[str, str]]:
    publish_root = agent_workspace_paths(run_root, agent_id)["publish_root"]
    rows: list[dict[str, str]] = []
    if publish_root.exists():
        for path in sorted(publish_root.rglob("*")):
            if not path.is_file():
                continue
            rows.append(
                {
                    "name": path.relative_to(publish_root).as_posix(),
                    "path": str(path),
                    "description": _describe_file(path),
                }
            )
    return rows


def _resolve_workspace_file_path(
    session: AgentSession,
    raw_path: str,
    *,
    must_exist: bool,
) -> tuple[Path, str, str]:
    try:
        return resolve_visible_write_file(
            session.run_root,
            session.agent_id,
            raw_path,
            must_exist=must_exist,
        )
    except VisibilityError as exc:
        raise ValueError(str(exc)) from exc


def _resolve_bash_cwd(session: AgentSession, raw_cwd: str | None) -> Path:
    workspace = agent_workspace_paths(session.run_root, session.agent_id)["workspace"]
    if not raw_cwd:
        return workspace
    try:
        resolved = resolve_visible_search_target(session.run_root, session.agent_id, raw_cwd)
    except VisibilityError as exc:
        raise ValueError(str(exc)) from exc
    if not resolved.is_dir():
        raise ValueError("bash cwd must be a visible directory.")
    return resolved


def _extract_non_option_args(argv: list[str], *, options_with_values: set[str] | None = None) -> list[str]:
    values: list[str] = []
    skip_next = False
    value_options = options_with_values or set()
    for item in argv:
        if skip_next:
            skip_next = False
            continue
        if item in value_options:
            skip_next = True
            continue
        if item.startswith("-"):
            continue
        values.append(item)
    return values


def _rg_path_argument_indexes(argv: list[str]) -> list[int]:
    consumes_value = {"-g", "--glob", "-e", "-m", "--max-count", "--color"}
    files_mode = any(item == "--files" for item in argv[1:])
    pattern_from_flag = any(item == "-e" for item in argv[1:])
    positional_indexes: list[int] = []
    skip_next = False
    for index, item in enumerate(argv[1:], start=1):
        if skip_next:
            skip_next = False
            continue
        if item in consumes_value:
            skip_next = True
            continue
        if item.startswith("-"):
            continue
        positional_indexes.append(index)
    if files_mode or pattern_from_flag:
        return positional_indexes
    return positional_indexes[1:]


def _validate_rg_options(argv: list[str]) -> None:
    allowed_flags = {
        "--files",
        "--json",
        "--line-number",
        "--no-heading",
        "--fixed-strings",
        "--ignore-case",
        "--color",
        "--glob",
        "--max-count",
        "-g",
        "-e",
        "-m",
        "-i",
    }
    consumes_value = {"-g", "--glob", "-e", "-m", "--max-count", "--color"}
    skip_next = False
    for item in argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if item in consumes_value:
            skip_next = True
            continue
        if item.startswith("-") and item not in allowed_flags:
            raise ValueError(f"Unsupported rg option '{item}'.")


def _truncate_shell_output(text: str, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n[truncated at {max_chars} chars]"


def _record_bash_event(session: AgentSession, argv: list[str], cwd: Path, returncode: int) -> None:
    append_event(
        session.run_root,
        AgentEvent(
            event_type="bash_executed",
            agent_id=session.agent_id,
            details={
                "argv": argv,
                "cwd": str(cwd),
                "returncode": returncode,
            },
        ),
    )


def _bash_result(
    *,
    session: AgentSession,
    argv: list[str],
    cwd: Path,
    returncode: int,
    stdout: str = "",
    stderr: str = "",
    summary: str | None = None,
) -> dict[str, Any]:
    _record_bash_event(session, argv, cwd, returncode)
    return {
        "ok": returncode == 0,
        "argv": argv,
        "cwd": str(cwd),
        "project_root": str(agent_workspace_paths(session.run_root, session.agent_id)["workspace"]),
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "content": stdout,
        "summary": summary or f"Command exited with code {returncode}.",
    }


def _handle_bash_sleep(session: AgentSession, argv: list[str], cwd: Path) -> dict[str, Any]:
    seconds = float(argv[1]) if len(argv) > 1 else 30
    time.sleep(seconds)
    return _bash_result(
        session=session,
        argv=argv,
        cwd=cwd,
        returncode=0,
        stdout="",
        stderr="",
        summary=f"Slept for {seconds} second(s).",
    ) | {"content": f"slept {seconds}s"}


def _handle_bash_ls(session: AgentSession, argv: list[str], cwd: Path) -> dict[str, Any]:
    raw_paths = _extract_non_option_args(argv[1:], options_with_values=set())
    rows: list[str] = []
    targets = raw_paths or ["."]
    for raw_path in targets:
        if raw_path in {".", "./"} and cwd == agent_workspace_paths(session.run_root, session.agent_id)["workspace"]:
            entries = visible_workspace_entries(session.run_root, session.agent_id)
            rows.extend(path.name + ("/" if path.is_dir() else "") for path in entries)
            continue
        try:
            target = resolve_visible_search_target(session.run_root, session.agent_id, raw_path, cwd=cwd)
        except VisibilityError as exc:
            raise ValueError(str(exc)) from exc
        if target.is_file():
            rows.append(str(target))
            continue
        for child in sorted(target.iterdir()):
            try:
                resolve_visible_search_target(session.run_root, session.agent_id, str(child), cwd=cwd)
            except VisibilityError:
                continue
            rows.append(str(child))
    stdout = "\n".join(rows) + ("\n" if rows else "")
    return _bash_result(session=session, argv=argv, cwd=cwd, returncode=0, stdout=stdout)


def _handle_bash_cp(session: AgentSession, argv: list[str], cwd: Path) -> dict[str, Any]:
    path_args = _extract_non_option_args(argv[1:], options_with_values=set())
    if len(path_args) != 2:
        raise ValueError("cp requires exactly one source file and one destination file.")
    try:
        source = resolve_visible_read_file(session.run_root, session.agent_id, path_args[0], cwd=cwd)
        destination, _root_name, _relative = resolve_visible_write_file(
            session.run_root,
            session.agent_id,
            path_args[1],
            cwd=cwd,
            must_exist=False,
        )
    except VisibilityError as exc:
        raise ValueError(str(exc)) from exc
    if destination.exists() and destination.is_dir():
        raise ValueError("cp destination must be a file path, not a directory.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return _bash_result(session=session, argv=argv, cwd=cwd, returncode=0)


def _handle_bash_mv(session: AgentSession, argv: list[str], cwd: Path) -> dict[str, Any]:
    path_args = _extract_non_option_args(argv[1:], options_with_values=set())
    if len(path_args) != 2:
        raise ValueError("mv requires exactly one source file and one destination file.")
    try:
        source, _source_root, _source_relative = resolve_visible_write_file(
            session.run_root,
            session.agent_id,
            path_args[0],
            cwd=cwd,
            must_exist=True,
        )
        destination, _dest_root, _dest_relative = resolve_visible_write_file(
            session.run_root,
            session.agent_id,
            path_args[1],
            cwd=cwd,
            must_exist=False,
        )
    except VisibilityError as exc:
        raise ValueError(str(exc)) from exc
    if source.is_dir() or (destination.exists() and destination.is_dir()):
        raise ValueError("mv supports file paths only.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))
    return _bash_result(session=session, argv=argv, cwd=cwd, returncode=0)


def _handle_bash_mkdir(session: AgentSession, argv: list[str], cwd: Path) -> dict[str, Any]:
    path_args = _extract_non_option_args(argv[1:], options_with_values=set())
    if not path_args:
        raise ValueError("mkdir requires at least one path.")
    try:
        targets = [
            resolve_visible_write_file(
                session.run_root,
                session.agent_id,
                raw_path,
                cwd=cwd,
                must_exist=False,
            )[0]
            for raw_path in path_args
        ]
    except VisibilityError as exc:
        raise ValueError(str(exc)) from exc
    for target in targets:
        target.mkdir(parents=True, exist_ok=True)
    return _bash_result(session=session, argv=argv, cwd=cwd, returncode=0)


def _handle_bash_rg(session: AgentSession, argv: list[str], cwd: Path, arguments: dict[str, Any]) -> dict[str, Any]:
    _validate_rg_options(argv)
    path_indexes = _rg_path_argument_indexes(argv)
    safe_argv = list(argv)
    try:
        if path_indexes:
            for index in path_indexes:
                safe_argv[index] = str(resolve_visible_search_target(session.run_root, session.agent_id, argv[index], cwd=cwd))
        else:
            safe_argv.extend(str(path) for path in default_search_targets(session.run_root, session.agent_id))
    except VisibilityError as exc:
        raise ValueError(str(exc)) from exc

    timeout_seconds = max(1, min(int(arguments.get("timeout_seconds", BASH_DEFAULT_TIMEOUT_SECONDS)), 60))
    max_output_chars = max(200, min(int(arguments.get("max_output_chars", BASH_DEFAULT_MAX_OUTPUT_CHARS)), 40000))
    try:
        completed = subprocess.run(
            safe_argv,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError:
        return _bash_result(
            session=session,
            argv=argv,
            cwd=cwd,
            returncode=127,
            stderr="Command 'rg' is not available.",
            summary="Command 'rg' is not available.",
        )
    except subprocess.TimeoutExpired:
        return _bash_result(
            session=session,
            argv=argv,
            cwd=cwd,
            returncode=124,
            stderr=f"Command timed out after {timeout_seconds} seconds.",
            summary=f"Command timed out after {timeout_seconds} seconds.",
        )
    stdout = _truncate_shell_output(completed.stdout, max_chars=max_output_chars)
    stderr = _truncate_shell_output(completed.stderr, max_chars=max_output_chars)
    return _bash_result(
        session=session,
        argv=argv,
        cwd=cwd,
        returncode=completed.returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _handle_bash(session: AgentSession, arguments: dict[str, Any]) -> dict[str, Any]:
    raw_argv = arguments.get("argv") or []
    argv = [str(item) for item in raw_argv if str(item)]
    if not argv:
        raise ValueError("bash requires a non-empty argv list.")
    command_name = Path(argv[0]).name
    if command_name not in BASH_ALLOWED_COMMANDS:
        raise ValueError(f"Illegal bash command '{command_name}'. Allowed commands: {', '.join(sorted(BASH_ALLOWED_COMMANDS))}.")

    cwd = _resolve_bash_cwd(session, str(arguments.get("cwd") or "").strip() or None)
    if command_name == "cp":
        return _handle_bash_cp(session, argv, cwd)
    elif command_name == "mv":
        return _handle_bash_mv(session, argv, cwd)
    elif command_name == "mkdir":
        return _handle_bash_mkdir(session, argv, cwd)
    elif command_name == "ls":
        return _handle_bash_ls(session, argv, cwd)
    elif command_name == "rg":
        return _handle_bash_rg(session, argv, cwd, arguments)
    elif command_name == "sleep":
        return _handle_bash_sleep(session, argv, cwd)
    raise ValueError(f"Unsupported bash command '{command_name}'.")


def _handle_write_file(session: AgentSession, arguments: dict[str, Any]) -> dict[str, Any]:
    raw_path = str(arguments.get("path") or "").strip()
    if not raw_path:
        raise ValueError("write requires path.")
    if "content" not in arguments or arguments.get("content") is None:
        raise ValueError("write requires content.")
    content = str(arguments["content"])
    path, root_name, relative = _resolve_workspace_file_path(session, raw_path, must_exist=False)
    if root_name == "publish" and not content.strip():
        raise ValueError("write requires non-empty content for publish paths.")
    write_text_atomic(path, content)
    event_type = "publish_updated" if root_name == "publish" else "scratch_updated"
    append_event(
        session.run_root,
        AgentEvent(
            event_type=event_type,
            agent_id=session.agent_id,
            details={"path": str(path), "file_name": relative},
        ),
    )
    return {"path": str(path), "description": _describe_file(path), "root": root_name}


def _find_nth_occurrence(text: str, needle: str, occurrence: int) -> int:
    if occurrence < 1:
        raise ValueError("occurrence must be >= 1.")
    index = -1
    start = 0
    for _ in range(occurrence):
        index = text.find(needle, start)
        if index == -1:
            return -1
        start = index + len(needle)
    return index


def _parse_patch_hunk(lines: list[str], *, hunk_number: int) -> PatchHunk:
    entries: list[tuple[str, str]] = []
    for raw_line in lines:
        if not raw_line:
            raise ValueError(f"Patch hunk {hunk_number} contains a line without a patch prefix.")
        prefix = raw_line[0]
        if prefix not in {" ", "+", "-"}:
            raise ValueError(f"Patch hunk {hunk_number} has an illegal line prefix '{prefix}'.")
        entries.append((prefix, raw_line[1:]))
    if not entries:
        raise ValueError(f"Patch hunk {hunk_number} is empty.")
    if all(prefix == "+" for prefix, _text in entries):
        raise ValueError(f"Patch hunk {hunk_number} must include at least one context or removed line.")
    return PatchHunk(entries=entries)


def _unwrap_patch_text(patch_text: str) -> str:
    if PATCH_TRUNCATION_MARKER_RE.search(patch_text):
        raise ValueError(
            "Patch contains a compacted replay truncation marker like '... [3485 chars]'. "
            "Read a smaller file slice with read(path=..., start_line=..., max_lines=...) "
            "or read(path=..., start_char=..., max_chars=...) and rebuild the full patch."
        )

    text = patch_text.strip()
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        if len(lines) < 2 or not lines[-1].strip().startswith("```"):
            raise ValueError("Patch code fence must close after the '*** End Patch' line.")
        text = "\n".join(lines[1:-1]).strip()
    return text


def _parse_apply_patch_text(patch_text: str) -> ParsedPatch:
    lines = _unwrap_patch_text(patch_text).splitlines()
    if not lines:
        raise ValueError("patch requires a non-empty patch.")
    if lines[0].strip() != PATCH_BEGIN_MARKER:
        raise ValueError("Patch must start with '*** Begin Patch'.")
    if lines[-1].strip() != PATCH_END_MARKER:
        raise ValueError("Patch must end with '*** End Patch'.")
    if len(lines) < 4:
        raise ValueError("Patch must include one file update and at least one hunk.")

    file_line = lines[1].strip()
    if not file_line.startswith(PATCH_UPDATE_FILE_PREFIX):
        raise ValueError("v1 patch only supports a single '*** Update File: ...' block.")
    path = file_line[len(PATCH_UPDATE_FILE_PREFIX):].strip()
    if not path:
        raise ValueError("Patch update file path cannot be empty.")

    hunks: list[PatchHunk] = []
    current_hunk_lines: list[str] = []
    hunk_number = 0
    for line in lines[2:-1]:
        if line.startswith("@@"):
            if current_hunk_lines:
                hunks.append(_parse_patch_hunk(current_hunk_lines, hunk_number=hunk_number))
                current_hunk_lines = []
            hunk_number += 1
            continue
        if line.startswith("*** "):
            raise ValueError("v1 patch only supports one updated file and no nested patch directives.")
        if hunk_number == 0:
            raise ValueError("Patch must include '@@' before hunk lines.")
        current_hunk_lines.append(line)

    if current_hunk_lines:
        hunks.append(_parse_patch_hunk(current_hunk_lines, hunk_number=hunk_number))
    if not hunks:
        raise ValueError("Patch must include at least one hunk.")
    return ParsedPatch(path=path, hunks=hunks)


def _find_unique_line_match(lines: list[str], expected_lines: list[str], *, hunk_number: int) -> int:
    if not expected_lines:
        raise ValueError(f"Patch hunk {hunk_number} has no anchor lines to match.")
    match_indexes = [
        index
        for index in range(len(lines) - len(expected_lines) + 1)
        if lines[index:index + len(expected_lines)] == expected_lines
    ]
    if not match_indexes:
        raise ValueError(f"Patch hunk {hunk_number} context was not found in the target file.")
    if len(match_indexes) > 1:
        raise ValueError(f"Patch hunk {hunk_number} context matched multiple locations.")
    return match_indexes[0]


def _normalize_separator_space_hunk(hunk: PatchHunk) -> PatchHunk:
    return PatchHunk(
        entries=[
            (prefix, text[1:] if text.startswith(" ") else text)
            for prefix, text in hunk.entries
        ]
    )


def _patch_uses_separator_space(parsed_patch: ParsedPatch) -> bool:
    return any(
        text.startswith(" ")
        for hunk in parsed_patch.hunks
        for _prefix, text in hunk.entries
    )


def _normalized_separator_space_patch(parsed_patch: ParsedPatch) -> ParsedPatch:
    return ParsedPatch(
        path=parsed_patch.path,
        hunks=[_normalize_separator_space_hunk(hunk) for hunk in parsed_patch.hunks],
    )


def _apply_line_patch_once(
    original: str,
    parsed_patch: ParsedPatch,
    *,
    patch_mode: str,
) -> tuple[str, dict[str, Any]]:
    current_lines = original.splitlines()
    trailing_newline = original.endswith("\n")

    for hunk_number, hunk in enumerate(parsed_patch.hunks, start=1):
        expected_lines = [text for prefix, text in hunk.entries if prefix != "+"]
        replacement_lines = [text for prefix, text in hunk.entries if prefix != "-"]
        start_index = _find_unique_line_match(current_lines, expected_lines, hunk_number=hunk_number)
        current_lines[start_index:start_index + len(expected_lines)] = replacement_lines

    updated = "\n".join(current_lines)
    if current_lines and trailing_newline:
        updated += "\n"
    return (
        updated,
        {
            "operation": "patch",
            "hunks_applied": len(parsed_patch.hunks),
            "patch_mode": patch_mode,
        },
    )


def _apply_line_patch(original: str, parsed_patch: ParsedPatch) -> tuple[str, dict[str, Any]]:
    try:
        return _apply_line_patch_once(original, parsed_patch, patch_mode="strict")
    except ValueError as strict_exc:
        if not _patch_uses_separator_space(parsed_patch):
            raise

        try:
            return _apply_line_patch_once(
                original,
                _normalized_separator_space_patch(parsed_patch),
                patch_mode="normalized_separator_space",
            )
        except ValueError as normalized_exc:
            if "matched multiple locations" in str(normalized_exc):
                raise normalized_exc from strict_exc
            raise strict_exc


def _apply_text_edit(original: str, arguments: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    operation = str(arguments.get("operation") or "").strip()
    content = str(arguments.get("content") or "")
    target_text = str(arguments.get("target_text") or "")
    occurrence = int(arguments.get("occurrence", 1))
    replace_all = bool(arguments.get("replace_all", False))

    if operation not in {"replace", "insert_before", "insert_after", "append", "prepend"}:
        raise ValueError("Illegal edit operation.")

    if operation == "append":
        return original + content, {"operation": operation, "matched": True, "match_count": 1}
    if operation == "prepend":
        return content + original, {"operation": operation, "matched": True, "match_count": 1}
    if not target_text:
        raise ValueError(f"{operation} requires target_text.")

    if operation == "replace":
        if replace_all:
            match_count = original.count(target_text)
            if match_count == 0:
                raise ValueError("target_text not found for replace.")
            return (
                original.replace(target_text, content),
                {"operation": operation, "matched": True, "match_count": match_count, "replace_all": True},
            )
        index = _find_nth_occurrence(original, target_text, occurrence)
        if index == -1:
            raise ValueError("target_text not found for replace.")
        updated = original[:index] + content + original[index + len(target_text) :]
        return (
            updated,
            {"operation": operation, "matched": True, "match_count": 1, "occurrence": occurrence, "replace_all": False},
        )

    index = _find_nth_occurrence(original, target_text, occurrence)
    if index == -1:
        raise ValueError(f"target_text not found for {operation}.")
    insert_at = index if operation == "insert_before" else index + len(target_text)
    updated = original[:insert_at] + content + original[insert_at:]
    return (
        updated,
        {"operation": operation, "matched": True, "match_count": 1, "occurrence": occurrence},
    )


def _emit_workspace_update(
    session: AgentSession,
    *,
    root_name: str,
    path: Path,
    relative: str,
    details: dict[str, Any],
) -> None:
    append_event(
        session.run_root,
        AgentEvent(
            event_type=f"{root_name}_updated",
            agent_id=session.agent_id,
            details={
                "path": str(path),
                "file_name": relative,
                **details,
            },
        ),
    )


def _handle_edit_file(session: AgentSession, arguments: dict[str, Any]) -> dict[str, Any]:
    raw_path = str(arguments.get("path") or "").strip()
    if not raw_path:
        raise ValueError("edit requires path.")
    path, root_name, relative = _resolve_workspace_file_path(session, raw_path, must_exist=True)
    if not path.is_file():
        raise ValueError(f"File '{relative}' does not exist inside {root_name}/.")
    original = read_text(path)
    updated, details = _apply_text_edit(original, arguments)
    write_text_atomic(path, updated)
    _emit_workspace_update(session, root_name=root_name, path=path, relative=relative, details=details)
    return {
        "path": str(path),
        "description": _describe_file(path),
        "root": root_name,
        "operation": details["operation"],
        "match_count": int(details.get("match_count", 0)),
        "before_chars": len(original),
        "after_chars": len(updated),
    }


def _rewrite_apply_patch_error(message: str, *, display_path: str, has_separator_space: bool = False) -> str:
    read_hint = f"Read the current file with read(path='{display_path}') before retrying."
    if "context was not found" in message:
        separator_hint = (
            " Patch prefixes are exact: '-### Heading' matches '### Heading', while "
            "'- ### Heading' matches a line that begins with a space."
            if has_separator_space
            else ""
        )
        return (
            f"{message} The file content does not match the patch context anymore. "
            f"{read_hint} Rebuild the patch from the exact current lines.{separator_hint}"
        )
    if "matched multiple locations" in message:
        return (
            f"{message} The patch context is ambiguous. "
            f"{read_hint} Use more specific surrounding lines in the next hunk."
        )
    return message


def _handle_apply_patch(session: AgentSession, arguments: dict[str, Any]) -> dict[str, Any]:
    raw_patch = arguments.get("patch")
    if raw_patch is None:
        raise ValueError("patch requires patch.")
    parsed_patch = _parse_apply_patch_text(str(raw_patch))
    path, root_name, relative = _resolve_workspace_file_path(session, parsed_patch.path, must_exist=True)
    if not path.is_file():
        raise ValueError(f"File '{relative}' does not exist inside {root_name}/.")

    original = read_text(path)
    try:
        updated, details = _apply_line_patch(original, parsed_patch)
    except ValueError as exc:
        raise ValueError(
            _rewrite_apply_patch_error(
                str(exc),
                display_path=f"{root_name}/{relative}",
                has_separator_space=_patch_uses_separator_space(parsed_patch),
            )
        ) from exc
    if root_name == "publish" and not updated.strip():
        raise ValueError("patch cannot leave publish paths empty.")

    write_text_atomic(path, updated)
    _emit_workspace_update(session, root_name=root_name, path=path, relative=relative, details=details)
    return {
        "path": str(path),
        "description": _describe_file(path),
        "root": root_name,
        "operation": details["operation"],
        "hunks_applied": int(details["hunks_applied"]),
        "patch_mode": str(details.get("patch_mode") or "strict"),
        "before_chars": len(original),
        "after_chars": len(updated),
    }


def _handle_set_status(session: AgentSession, arguments: dict[str, Any]) -> dict[str, Any]:
    status = str(arguments.get("status") or "").strip()
    if status not in {"queued", "running", "waiting", "done", "failed", "blocked"}:
        raise ValueError(f"Illegal status '{status}'.")
    write_status(session.run_root, session.agent_id, status)
    error = str(arguments.get("error") or "").strip()
    append_event(
        session.run_root,
        AgentEvent(
            event_type="status_set",
            agent_id=session.agent_id,
            details={"status": status, "error": error},
        ),
    )
    if error:
        session.state.last_error = error
    return {"status": status, "error": error}


_HANDLERS = {
    "delegate": _handle_spawn_subagent,
    "agents": _handle_list_children,
    "bash": _handle_bash,
    "write": _handle_write_file,
    "edit": _handle_edit_file,
    "patch": _handle_apply_patch,
    "status": _handle_set_status,
}
