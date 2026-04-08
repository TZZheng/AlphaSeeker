"""File-first tool backend for harness agents."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any
from uuid import uuid4

from src.harness.artifacts import (
    agent_workspace_paths,
    append_tool_history,
    append_event,
    build_reduction_paths,
    create_agent_workspace,
    latest_agent_records,
    load_skill_state,
    load_object_manifest,
    promote_object,
    read_jsonl,
    read_status,
    read_text,
    refresh_progress_view,
    save_skill_state,
    write_status,
    write_text_atomic,
)
from src.harness.presets import default_tool_allowlist, render_tools_markdown, visible_skills_for_preset
from src.harness.registry import build_skill_registry, get_skills_for_packs
from src.harness.skills.common import json_preview
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


TOOL_NAMES = [
    "spawn_subagent",
    "list_children",
    "list_publish_files",
    "promote_artifact",
    "bash",
    "write_file",
    "edit_file",
    "set_status",
]

TERMINAL_STATUSES = {"done", "failed", "blocked", "stale", "cancelled"}

_TYPE_MAP = {
    "string": {"type": "string"},
    "integer": {"type": "integer"},
    "number": {"type": "number"},
    "boolean": {"type": "boolean"},
}
_LEGAL_PRESET_LIST = ", ".join(f"'{preset}'" for preset in AGENT_PRESETS)
BASH_ALLOWED_COMMANDS = {"cp", "mv", "mkdir", "ls", "rg", "sleep"}
BASH_DEFAULT_TIMEOUT_SECONDS = 10
BASH_DEFAULT_MAX_OUTPUT_CHARS = 12000


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


def _tool_schema_properties(input_schema: dict[str, Any]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    for name, raw_type in input_schema.items():
        if isinstance(raw_type, dict) and "type" in raw_type:
            properties[name] = raw_type
            continue
        type_name = str(raw_type).strip()
        if type_name.endswith("[]"):
            item_type = type_name[:-2]
            properties[name] = {
                "type": "array",
                "items": _TYPE_MAP.get(item_type, {"type": "string"}),
            }
            continue
        properties[name] = _TYPE_MAP.get(type_name, {"type": "string"})
    return properties


def _tool_definitions() -> dict[str, dict[str, Any]]:
    return {
        "spawn_subagent": {
            "description": f"Launch one child agent for a narrower task. Legal preset values: {_LEGAL_PRESET_LIST}.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "task_name": {"type": "string"},
                    "description": {"type": "string"},
                    "preset": {
                        "type": "string",
                        "enum": list(AGENT_PRESETS),
                        "description": f"One of: {_LEGAL_PRESET_LIST}.",
                    },
                    "instructions": {"type": "string"},
                    "context_files": {"type": "array", "items": {"type": "string"}},
                    "expected_publish_files": {"type": "array", "items": {"type": "string"}},
                    "task_markdown": {"type": "string"},
                },
            },
        },
        "list_children": {
            "description": "List all child agents with status. Drains the events queue so callers know which children just finished. Use in a loop: drain, do useful work, repeat until running_count=0.",
            "input_schema": {"type": "object", "properties": {}},
        },
        "list_publish_files": {
            "description": "List published files for an agent.",
            "input_schema": {
                "type": "object",
                "properties": {"agent_id": {"type": "string"}},
            },
        },
        "promote_artifact": {
            "description": "Promote a local artifact into the shared run object store.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "source_path": {"type": "string"},
                    "description": {"type": "string"},
                },
            },
        },
        "bash": {
            "description": "Run one repo-scoped bash command from the allowlist (cp, mv, mkdir, ls, rg, sleep).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "argv": {"type": "array", "items": {"type": "string"}},
                    "cwd": {"type": "string"},
                    "timeout_seconds": {"type": "integer"},
                    "max_output_chars": {"type": "integer"},
                },
            },
        },
        "write_file": {
            "description": "Write one file under this agent's publish/ or scratch/ tree.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
        "edit_file": {
            "description": "Apply one anchored text edit to a file under this agent's publish/ or scratch/ tree.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "operation": {
                        "type": "string",
                        "enum": ["replace", "insert_before", "insert_after", "append", "prepend"],
                    },
                    "target_text": {"type": "string"},
                    "content": {"type": "string"},
                    "occurrence": {"type": "integer"},
                    "replace_all": {"type": "boolean"},
                },
            },
        },
        "set_status": {
            "description": "Set this agent's status when it is ready to stop or block.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "error": {"type": "string"},
                },
            },
        },
}


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
    tools: list[dict[str, Any]] = []
    base = _tool_definitions()
    for name in session.allowed_tools:
        if name not in base:
            continue
        spec = base[name]
        tools.append(
            {
                "name": name,
                "description": spec["description"],
                "input_schema": spec["input_schema"],
            }
        )

    for spec in visible_skills_for_preset(preset=session.preset, available_skills=session.state.available_skills):
        tools.append(
            {
                "name": spec.name,
                "description": spec.description,
                "input_schema": {
                    "type": "object",
                    "properties": _tool_schema_properties(spec.input_schema),
                },
            }
        )
    return tools


def execute_agent_command(session: AgentSession, command: AgentCommand) -> dict[str, Any]:
    if command.tool not in session.allowed_tools:
        raise ValueError(f"Tool '{command.tool}' is not allowed for preset '{session.preset}'.")
    handler = _HANDLERS.get(command.tool)
    if handler is None:
        raise ValueError(f"Unknown tool '{command.tool}'.")
    result = handler(session, command.arguments)
    _append_journal(
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

    journal_row = {
        "tool": tool_name,
        "arguments": arguments,
        "result": result,
        "created_at": _now_iso(),
    }
    _append_journal(session, journal_row)
    append_tool_history(session.run_root, session.agent_id, journal_row)
    save_skill_state(session.state)
    refresh_progress_view(session.run_root)
    return result


def _append_journal(session: AgentSession, payload: dict[str, Any]) -> None:
    journal_path = agent_workspace_paths(session.run_root, session.agent_id)["journal"]
    with journal_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=True) + "\n")


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
    root = agent_workspace_paths(session.run_root, session.agent_id)["scratch_root"] / "skills"
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
    output_files: list[str] = []
    if result.output_text:
        output_path = output_root / "output.md"
        write_text_atomic(output_path, result.output_text)
        output_files.append(str(output_path))
    summary_path = output_root / "summary.md"
    write_text_atomic(summary_path, result.summary + "\n")
    output_files.append(str(summary_path))
    details_path = output_root / "details.json"
    write_text_atomic(details_path, json_preview(result.details))
    output_files.append(str(details_path))
    evidence_path: str | None = None
    if result.evidence:
        evidence_file = output_root / "evidence.json"
        write_text_atomic(
            evidence_file,
            json.dumps([item.model_dump(mode="json") for item in result.evidence], indent=2, ensure_ascii=True),
        )
        evidence_path = str(evidence_file)
        output_files.append(evidence_path)
    artifact_manifest_path: str | None = None
    if result.artifacts:
        artifact_manifest = output_root / "artifacts.txt"
        write_text_atomic(artifact_manifest, "\n".join(result.artifacts) + "\n")
        artifact_manifest_path = str(artifact_manifest)
        output_files.append(artifact_manifest_path)

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
    return {
        "skill_name": skill_name,
        "status": result.status,
        "summary": result.summary,
        "output_root": str(output_root),
        "artifact_paths": list(result.artifacts),
        "output_files": output_files,
        "primary_artifact_path": result.artifacts[0] if result.artifacts else "",
        "summary_path": str(summary_path),
        "details_path": str(details_path),
        "evidence_path": evidence_path or "",
        "artifacts_manifest_path": artifact_manifest_path or "",
        "artifact_count": len(result.artifacts),
        "evidence_count": len(result.evidence),
        "error": result.error,
        "content": (
            result.output_text
            if skill_name in {"read_file", "read_web_pages", "condense_context"}
            else ""
        ),
        "details": result.details,
    }


def _handle_call_skill(session: AgentSession, arguments: dict[str, Any]) -> dict[str, Any]:
    skill_name = str(arguments.get("skill_name") or arguments.get("name") or "").strip()
    skill_args = arguments.get("arguments") or {}
    if not skill_name:
        raise ValueError("call_skill requires skill_name.")
    return _run_skill(session, skill_name, skill_args)


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
            f"# {task_name}",
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


def _handle_list_publish_files(session: AgentSession, arguments: dict[str, Any]) -> dict[str, Any]:
    target_agent = str(arguments.get("agent_id") or session.agent_id).strip()
    return {"agent_id": target_agent, "files": _publish_file_rows(session.run_root, target_agent)}


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


def _safe_relative_path(file_name: str) -> Path:
    relative = Path(file_name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Illegal file path.")
    return relative


def _resolve_workspace_file_path(
    session: AgentSession,
    raw_path: str,
    *,
    must_exist: bool,
) -> tuple[Path, str, str]:
    candidate = Path(raw_path).expanduser()
    workspace_paths = agent_workspace_paths(session.run_root, session.agent_id)
    allowed_roots = {
        "publish": workspace_paths["publish_root"].resolve(strict=False),
        "scratch": workspace_paths["scratch_root"].resolve(strict=False),
    }

    if candidate.is_absolute():
        resolved = candidate.resolve(strict=False)
        for root_name, root in allowed_roots.items():
            if resolved == root or root in resolved.parents:
                if must_exist and not resolved.exists():
                    raise ValueError(f"File '{raw_path}' does not exist.")
                relative = resolved.relative_to(root).as_posix()
                if not relative or relative == ".":
                    raise ValueError("Path must point to a file inside publish/ or scratch/.")
                return resolved, root_name, relative
        raise ValueError("Path must stay inside this agent's publish/ or scratch/ tree.")

    relative = _safe_relative_path(raw_path)
    if not relative.parts:
        raise ValueError("Illegal file path.")
    root_name = relative.parts[0]
    if root_name not in allowed_roots:
        raise ValueError("Relative paths must start with 'publish/' or 'scratch/'.")
    sub_relative = Path(*relative.parts[1:]) if len(relative.parts) > 1 else Path()
    if not sub_relative.parts:
        raise ValueError("Path must point to a file inside publish/ or scratch/.")
    resolved = allowed_roots[root_name] / sub_relative
    if must_exist and not resolved.exists():
        raise ValueError(f"File '{raw_path}' does not exist.")
    return resolved, root_name, sub_relative.as_posix()


def _project_root_for_session(session: AgentSession) -> Path:
    cwd = Path.cwd().resolve()
    if (cwd / "pyproject.toml").exists() or (cwd / ".git").exists():
        return cwd
    run_root = Path(session.run_root).resolve(strict=False)
    for candidate in (run_root, *run_root.parents):
        if (candidate / "pyproject.toml").exists() or (candidate / ".git").exists():
            return candidate
    return cwd


def _is_within(root: Path, candidate: Path) -> bool:
    resolved_root = root.resolve(strict=False)
    resolved_candidate = candidate.resolve(strict=False)
    return resolved_candidate == resolved_root or resolved_root in resolved_candidate.parents


def _resolve_repo_path(project_root: Path, cwd: Path, raw_path: str, *, allow_missing: bool) -> Path:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute():
        candidate = cwd / candidate
    resolved = candidate.resolve(strict=False)
    if not _is_within(project_root, resolved):
        raise ValueError(f"Path '{raw_path}' escapes the project root.")
    if not allow_missing and not resolved.exists():
        raise ValueError(f"Path '{raw_path}' does not exist.")
    return resolved


def _resolve_bash_cwd(session: AgentSession, raw_cwd: str | None) -> tuple[Path, Path]:
    project_root = _project_root_for_session(session)
    if not raw_cwd:
        return project_root, project_root
    resolved = _resolve_repo_path(project_root, project_root, raw_cwd, allow_missing=False)
    if not resolved.is_dir():
        raise ValueError("bash cwd must be a directory inside the project root.")
    return project_root, resolved


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


def _validate_bash_cp_mv(argv: list[str], *, project_root: Path, cwd: Path, command_name: str) -> None:
    options_with_values: set[str] = set()
    path_args = _extract_non_option_args(argv[1:], options_with_values=options_with_values)
    if len(path_args) < 2:
        raise ValueError(f"{command_name} requires at least one source path and one destination path.")
    for raw_path in path_args[:-1]:
        _resolve_repo_path(project_root, cwd, raw_path, allow_missing=False)
    _resolve_repo_path(project_root, cwd, path_args[-1], allow_missing=True)


def _validate_bash_mkdir(argv: list[str], *, project_root: Path, cwd: Path) -> None:
    path_args = _extract_non_option_args(argv[1:], options_with_values=set())
    if not path_args:
        raise ValueError("mkdir requires at least one path.")
    for raw_path in path_args:
        _resolve_repo_path(project_root, cwd, raw_path, allow_missing=True)


def _validate_bash_ls(argv: list[str], *, project_root: Path, cwd: Path) -> None:
    path_args = _extract_non_option_args(argv[1:], options_with_values=set())
    for raw_path in path_args:
        _resolve_repo_path(project_root, cwd, raw_path, allow_missing=False)


def _validate_bash_rg(argv: list[str], *, project_root: Path, cwd: Path) -> None:
    consumes_value = {"-g", "--glob", "-e", "-m", "--max-count", "--color"}
    files_mode = any(item == "--files" for item in argv[1:])
    pattern_from_flag = any(item == "-e" for item in argv[1:])
    positional: list[str] = []
    skip_next = False
    for item in argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if item in consumes_value:
            skip_next = True
            continue
        if item.startswith("-"):
            continue
        positional.append(item)
    path_args = positional if files_mode or pattern_from_flag else positional[1:]
    for raw_path in path_args:
        _resolve_repo_path(project_root, cwd, raw_path, allow_missing=False)


def _truncate_shell_output(text: str, *, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n\n[truncated at {max_chars} chars]"


def _handle_bash(session: AgentSession, arguments: dict[str, Any]) -> dict[str, Any]:
    raw_argv = arguments.get("argv") or []
    argv = [str(item) for item in raw_argv if str(item)]
    if not argv:
        raise ValueError("bash requires a non-empty argv list.")
    command_name = Path(argv[0]).name
    if command_name not in BASH_ALLOWED_COMMANDS:
        raise ValueError(f"Illegal bash command '{command_name}'. Allowed commands: {', '.join(sorted(BASH_ALLOWED_COMMANDS))}.")

    project_root, cwd = _resolve_bash_cwd(session, str(arguments.get("cwd") or "").strip() or None)
    if command_name == "cp":
        _validate_bash_cp_mv(argv, project_root=project_root, cwd=cwd, command_name="cp")
    elif command_name == "mv":
        _validate_bash_cp_mv(argv, project_root=project_root, cwd=cwd, command_name="mv")
    elif command_name == "mkdir":
        _validate_bash_mkdir(argv, project_root=project_root, cwd=cwd)
    elif command_name == "ls":
        _validate_bash_ls(argv, project_root=project_root, cwd=cwd)
    elif command_name == "rg":
        _validate_bash_rg(argv, project_root=project_root, cwd=cwd)
    elif command_name == "sleep":
        # Handled inline without subprocess
        seconds = float(argv[1]) if len(argv) > 1 else 30
        time.sleep(seconds)
        return {
            "ok": True,
            "argv": argv,
            "cwd": str(cwd),
            "project_root": str(project_root),
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "content": f"slept {seconds}s",
            "summary": f"Slept for {seconds} second(s).",
        }

    timeout_seconds = max(1, min(int(arguments.get("timeout_seconds", BASH_DEFAULT_TIMEOUT_SECONDS)), 60))
    max_output_chars = max(200, min(int(arguments.get("max_output_chars", BASH_DEFAULT_MAX_OUTPUT_CHARS)), 40000))
    try:
        completed = subprocess.run(
            argv,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            env={**os.environ, "PWD": str(cwd)},
        )
    except FileNotFoundError:
        return {
            "ok": False,
            "argv": argv,
            "cwd": str(cwd),
            "project_root": str(project_root),
            "returncode": 127,
            "stdout": "",
            "stderr": f"Command '{command_name}' is not available.",
            "content": "",
            "summary": f"Command '{command_name}' is not available.",
        }
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "argv": argv,
            "cwd": str(cwd),
            "project_root": str(project_root),
            "returncode": 124,
            "stdout": "",
            "stderr": f"Command timed out after {timeout_seconds} seconds.",
            "content": "",
            "summary": f"Command timed out after {timeout_seconds} seconds.",
        }
    stdout = _truncate_shell_output(completed.stdout, max_chars=max_output_chars)
    stderr = _truncate_shell_output(completed.stderr, max_chars=max_output_chars)
    append_event(
        session.run_root,
        AgentEvent(
            event_type="bash_executed",
            agent_id=session.agent_id,
            details={
                "argv": argv,
                "cwd": str(cwd),
                "returncode": completed.returncode,
            },
        ),
    )
    return {
        "ok": completed.returncode == 0,
        "argv": argv,
        "cwd": str(cwd),
        "project_root": str(project_root),
        "returncode": completed.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "content": stdout,
        "summary": f"Command exited with code {completed.returncode}.",
    }


def _handle_promote_artifact(session: AgentSession, arguments: dict[str, Any]) -> dict[str, Any]:
    source_path = str(arguments.get("source_path") or "").strip()
    if not source_path:
        raise ValueError("promote_artifact requires source_path.")
    description = str(arguments.get("description") or Path(source_path).name).strip()
    promoted = promote_object(
        session.run_root,
        source_path=source_path,
        description=description,
        agent_id=session.agent_id,
    )
    promoted["source_path"] = source_path
    return promoted


def _handle_write_file(session: AgentSession, arguments: dict[str, Any]) -> dict[str, Any]:
    raw_path = str(arguments.get("path") or "").strip()
    if not raw_path:
        raise ValueError("write_file requires path.")
    if "content" not in arguments or arguments.get("content") is None:
        raise ValueError("write_file requires content.")
    content = str(arguments["content"])
    path, root_name, relative = _resolve_workspace_file_path(session, raw_path, must_exist=False)
    if root_name == "publish" and not content.strip():
        raise ValueError("write_file requires non-empty content for publish paths.")
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


def _handle_edit_file(session: AgentSession, arguments: dict[str, Any]) -> dict[str, Any]:
    raw_path = str(arguments.get("path") or "").strip()
    if not raw_path:
        raise ValueError("edit_file requires path.")
    path, root_name, relative = _resolve_workspace_file_path(session, raw_path, must_exist=True)
    if not path.is_file():
        raise ValueError(f"File '{relative}' does not exist inside {root_name}/.")
    original = read_text(path)
    updated, details = _apply_text_edit(original, arguments)
    write_text_atomic(path, updated)
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
    return {
        "path": str(path),
        "description": _describe_file(path),
        "root": root_name,
        "operation": details["operation"],
        "match_count": int(details.get("match_count", 0)),
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
    "call_skill": _handle_call_skill,
    "spawn_subagent": _handle_spawn_subagent,
    "spawn_agent": _handle_spawn_subagent,
    "list_children": _handle_list_children,
    "list_publish_files": _handle_list_publish_files,
    "promote_artifact": _handle_promote_artifact,
    "bash": _handle_bash,
    "write_file": _handle_write_file,
    "edit_file": _handle_edit_file,
    "set_status": _handle_set_status,
}
