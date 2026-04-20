"""Internal paired commenter sidecars for harness agents."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, TypedDict

import anthropic
from openai import OpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.harness.artifacts import (
    agent_workspace_paths,
    append_commenter_comments,
    latest_agent_records,
    load_commenter_state,
    load_skill_state,
    read_status,
    save_commenter_state,
    unread_commenter_comments,
    write_text_atomic,
)
from src.harness.prompt_builder import build_commenter_prompt_bundle
from src.harness.skills.core import read_file_skill, search_in_files_skill
from src.harness.transport import (
    minimax_anthropic_base_url,
    minimax_openai_base_url,
    normalize_minimax_model_name,
    resolve_agent_transport,
)
from src.harness.types import HarnessRequest, HarnessState, SkillResult
from src.shared.llm_manager import get_llm
from src.shared.model_config import get_model


COMMENTER_REFRESH_INTERVAL_SECONDS = 60.0
COMMENTER_MAX_FEED_COMMENTS = 3
COMMENTER_MAX_TOOL_STEPS = 100
COMMENTER_DEFAULT_READ_MAX_CHARS = 12_000
COMMENTER_MAX_RENDERED_CHANGED_ENTRIES = 20
COMMENTER_TERMINAL_STATUSES = {"done", "failed", "blocked", "stale", "cancelled"}
COMMENTER_OPERATING_LOG_FILES = {
    "journal.jsonl",
    "tool_history.jsonl",
    "transcript.jsonl",
    "worker.log",
}


class CommenterChangedEntry(TypedDict):
    change_type: str
    path: str
    display_path: str
    category: str
    size_bytes: int
    mtime_ns: int


class CommenterObservationSnapshot(TypedDict):
    fired_at: str
    base_fingerprint: str
    target_fingerprint: str
    base_manifest: list[dict[str, Any]]
    target_manifest: list[dict[str, Any]]
    changed_entries: list[CommenterChangedEntry]
    inspectable_scope: list[str]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _llm_text_content(content: object) -> str:
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


def _strip_provider_thinking(text: str) -> str:
    if not text:
        return ""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<thinking>.*?</thinking>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    return cleaned.strip()


def _invoke_text_response(
    *,
    model_name: str,
    transport_name: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    if transport_name in {"minimax_anthropic", "anthropic"}:
        if transport_name == "anthropic":
            client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
            model = model_name
        else:
            client = anthropic.Anthropic(
                base_url=minimax_anthropic_base_url(),
                api_key=os.environ["MINIMAX_API_KEY"],
            )
            model = normalize_minimax_model_name(model_name)
        for max_tokens in (768, 1536):
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": [{"type": "text", "text": user_prompt}]}],
            )
            text = "\n".join(
                str(getattr(block, "text", ""))
                for block in response.content
                if getattr(block, "type", "") == "text"
            )
            text = _strip_provider_thinking(text)
            if text:
                return text
        return ""
    if transport_name in {"minimax_openai", "openai"}:
        if transport_name == "openai":
            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            model = model_name
        else:
            client = OpenAI(
                base_url=minimax_openai_base_url(),
                api_key=os.environ["MINIMAX_API_KEY"],
            )
            model = normalize_minimax_model_name(model_name)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=2048,
        )
        choice = response.choices[0]
        text = getattr(choice.message, "content", None) or ""
        return _strip_provider_thinking(str(text))
    llm = get_llm(model_name)
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    return _strip_provider_thinking(_llm_text_content(response.content))


def _iter_workspace_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file())


def _file_entry(path: Path, *, category: str, display_path: str) -> dict[str, Any]:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return {}
    return {
        "path": str(path),
        "display_path": display_path,
        "category": category,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _commenter_scratch_category(path: Path, scratch_root: Path) -> str:
    relative = path.relative_to(scratch_root).as_posix()
    if relative in COMMENTER_OPERATING_LOG_FILES:
        return "operating_log"
    if relative.startswith("llm_turns/"):
        return "llm_trace"
    if relative.startswith("skills/"):
        return "tool_artifact"
    return "scratch"


def build_commenter_observation_manifest(run_root: str, agent_id: str) -> list[dict[str, Any]]:
    paths = agent_workspace_paths(run_root, agent_id)
    manifest: list[dict[str, Any]] = []
    for key, category in (("task", "task"), ("tools", "tools")):
        path = paths[key]
        if path.exists():
            entry = _file_entry(
                path,
                category=category,
                display_path=str(path.relative_to(paths["workspace"])),
            )
            if entry:
                manifest.append(entry)
    for root_key, category in (
        ("context_root", "context"),
        ("scratch_root", "scratch"),
        ("publish_root", "publish"),
    ):
        root = paths[root_key]
        for path in _iter_workspace_files(root):
            if paths["commenter_root"] in path.parents or path == paths["commenter_latest"]:
                continue
            effective_category = category
            if root_key == "scratch_root":
                effective_category = _commenter_scratch_category(path, root)
            entry = _file_entry(
                path,
                category=effective_category,
                display_path=str(path.relative_to(paths["workspace"])),
            )
            if entry:
                manifest.append(entry)
    for record in latest_agent_records(run_root).values():
        if record.parent_id != agent_id:
            continue
        child_paths = agent_workspace_paths(run_root, record.agent_id)
        for path in _iter_workspace_files(child_paths["publish_root"]):
            entry = _file_entry(
                path,
                category="child_publish",
                display_path=f"{record.agent_id}/publish/{path.relative_to(child_paths['publish_root']).as_posix()}",
            )
            if entry:
                manifest.append(entry)
    manifest.sort(key=lambda item: item["path"])
    return manifest


def _normalize_manifest_entries(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        return []
    manifest: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "").strip()
        display_path = str(item.get("display_path") or "").strip()
        category = str(item.get("category") or "").strip()
        if not path or not display_path:
            continue
        manifest.append(
            {
                "path": path,
                "display_path": display_path,
                "category": category,
                "size_bytes": int(item.get("size_bytes") or 0),
                "mtime_ns": int(item.get("mtime_ns") or 0),
            }
        )
    manifest.sort(key=lambda item: item["path"])
    return manifest


def _fingerprint_manifest(manifest: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for item in manifest:
        digest.update(
            f"{item['path']}|{item['size_bytes']}|{item['mtime_ns']}".encode("utf-8")
        )
    return digest.hexdigest()


def compute_commenter_observation_fingerprint(run_root: str, agent_id: str) -> str:
    return _fingerprint_manifest(build_commenter_observation_manifest(run_root, agent_id))


def _sanitize_transcript_content(text: str) -> str:
    rows: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            rows.append(raw_line)
            continue
        if payload.get("kind") != "user_message":
            rows.append(raw_line)
            continue
        message = payload.get("message")
        if not isinstance(message, dict):
            rows.append(raw_line)
            continue
        content = message.get("content")
        text_parts: list[str] = []
        if isinstance(content, str):
            text_parts = [content]
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    text_parts.append(item["text"])
        joined = "\n".join(text_parts)
        if joined.startswith("Comment Feed"):
            continue
        rows.append(raw_line)
    return "\n".join(rows)


def _read_commenter_file_content(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if path.name == "transcript.jsonl":
        text = _sanitize_transcript_content(text)
    return text


def _serialize_payload(payload: Any) -> Any:
    if hasattr(payload, "model_dump"):
        return payload.model_dump(mode="json")
    if isinstance(payload, dict):
        return {key: _serialize_payload(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_serialize_payload(item) for item in payload]
    if hasattr(payload, "__dict__"):
        return {
            key: _serialize_payload(value)
            for key, value in vars(payload).items()
            if not key.startswith("_")
        }
    return payload


def _manifest_alias_map(manifest: list[dict[str, Any]]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for item in manifest:
        exact_path = str(item.get("path") or "").strip()
        display_path = str(item.get("display_path") or "").strip()
        if exact_path:
            aliases[exact_path] = exact_path
        if display_path:
            aliases[display_path] = exact_path
    return aliases


def _commenter_allowed_roots(run_root: str, agent_id: str) -> list[Path]:
    paths = agent_workspace_paths(run_root, agent_id)
    roots = [paths["context_root"], paths["scratch_root"], paths["publish_root"]]
    for record in latest_agent_records(run_root).values():
        if record.parent_id != agent_id:
            continue
        roots.append(agent_workspace_paths(run_root, record.agent_id)["publish_root"])
    return roots


def _commenter_inspectable_scope(run_root: str, agent_id: str) -> list[str]:
    scope = ["task.md", "tools.md", "context/", "scratch/", "publish/"]
    child_publish_roots = sorted(
        f"{record.agent_id}/publish/"
        for record in latest_agent_records(run_root).values()
        if record.parent_id == agent_id
    )
    scope.extend(child_publish_roots)
    return scope


def _build_changed_entries(
    *,
    base_manifest: list[dict[str, Any]],
    target_manifest: list[dict[str, Any]],
) -> list[CommenterChangedEntry]:
    base_by_path = {str(item["path"]): item for item in base_manifest}
    target_by_path = {str(item["path"]): item for item in target_manifest}
    changed_entries: list[CommenterChangedEntry] = []
    for path in sorted(set(base_by_path) | set(target_by_path)):
        base_item = base_by_path.get(path)
        target_item = target_by_path.get(path)
        if base_item is None and target_item is not None:
            source_item = target_item
            change_type = "added"
        elif target_item is None and base_item is not None:
            source_item = base_item
            change_type = "deleted"
        elif base_item is not None and target_item is not None:
            if (
                str(base_item.get("display_path") or "") == str(target_item.get("display_path") or "")
                and str(base_item.get("category") or "") == str(target_item.get("category") or "")
                and int(base_item.get("size_bytes") or 0) == int(target_item.get("size_bytes") or 0)
                and int(base_item.get("mtime_ns") or 0) == int(target_item.get("mtime_ns") or 0)
            ):
                continue
            source_item = target_item
            change_type = "modified"
        else:
            continue
        changed_entries.append(
            {
                "change_type": change_type,
                "path": str(source_item.get("path") or ""),
                "display_path": str(source_item.get("display_path") or source_item.get("path") or ""),
                "category": str(source_item.get("category") or ""),
                "size_bytes": int(source_item.get("size_bytes") or 0),
                "mtime_ns": int(source_item.get("mtime_ns") or 0),
            }
        )
    return changed_entries


def build_commenter_observation_snapshot(
    run_root: str,
    agent_id: str,
    *,
    base_manifest: list[dict[str, Any]] | None = None,
    fired_at: str | None = None,
) -> CommenterObservationSnapshot:
    normalized_base_manifest = _normalize_manifest_entries(base_manifest or [])
    target_manifest = build_commenter_observation_manifest(run_root, agent_id)
    return {
        "fired_at": fired_at or _utc_now_iso(),
        "base_fingerprint": _fingerprint_manifest(normalized_base_manifest),
        "target_fingerprint": _fingerprint_manifest(target_manifest),
        "base_manifest": normalized_base_manifest,
        "target_manifest": target_manifest,
        "changed_entries": _build_changed_entries(
            base_manifest=normalized_base_manifest,
            target_manifest=target_manifest,
        ),
        "inspectable_scope": _commenter_inspectable_scope(run_root, agent_id),
    }


def _resolve_commenter_target_path(
    *,
    raw_path: str,
    manifest: list[dict[str, Any]],
    workspace_root: Path,
) -> Path:
    normalized = raw_path.strip()
    aliases = _manifest_alias_map(manifest)
    if normalized in aliases:
        return Path(aliases[normalized])
    candidate = Path(normalized).expanduser()
    if not candidate.is_absolute():
        candidate = workspace_root / normalized
    return candidate


def _is_commenter_path_allowed(candidate: Path, *, run_root: str, agent_id: str) -> bool:
    paths = agent_workspace_paths(run_root, agent_id)
    try:
        resolved = candidate.resolve(strict=False)
    except OSError:
        resolved = candidate
    commenter_root = paths["commenter_root"].resolve(strict=False)
    if resolved == commenter_root or commenter_root in resolved.parents:
        return False
    exact_files = {paths["task"].resolve(strict=False), paths["tools"].resolve(strict=False)}
    if resolved in exact_files:
        return True
    for root in _commenter_allowed_roots(run_root, agent_id):
        resolved_root = root.resolve(strict=False)
        if resolved == resolved_root or resolved_root in resolved.parents:
            return True
    return False


def _commenter_skill_state(run_root: str, agent_id: str, request: HarnessRequest) -> HarnessState:
    existing = load_skill_state(run_root, agent_id)
    if existing is not None:
        existing.request = request
        existing.run_root = run_root
        existing.agent_id = agent_id
        existing.workspace_path = str(agent_workspace_paths(run_root, agent_id)["workspace"])
        return existing
    return HarnessState(
        request=request,
        run_id=Path(run_root).name,
        run_root=run_root,
        agent_id=agent_id,
        workspace_path=str(agent_workspace_paths(run_root, agent_id)["workspace"]),
        enabled_packs=request.available_skill_packs or ["core"],
    )


def _compact_skill_result(result: SkillResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "summary": result.summary,
        "details": result.details,
        "output_text": result.output_text,
        "error": result.error,
    }


def _commenter_read_file(
    *,
    run_root: str,
    agent_id: str,
    request: HarnessRequest,
    manifest: list[dict[str, Any]],
    arguments: dict[str, Any],
) -> dict[str, Any]:
    workspace_root = agent_workspace_paths(run_root, agent_id)["workspace"]
    raw_path = str(arguments.get("path") or "").strip()
    if not raw_path:
        return {
            "status": "failed",
            "summary": "read_file requires a path.",
            "details": {"path": raw_path},
            "output_text": None,
            "error": "Missing path.",
        }
    candidate = _resolve_commenter_target_path(
        raw_path=raw_path,
        manifest=manifest,
        workspace_root=workspace_root,
    )
    if not _is_commenter_path_allowed(candidate, run_root=run_root, agent_id=agent_id):
        return {
            "status": "failed",
            "summary": f"Path '{raw_path}' is outside the commenter observation surface.",
            "details": {"path": raw_path},
            "output_text": None,
            "error": "Path not allowed.",
        }
    max_chars = int(arguments.get("max_chars", COMMENTER_DEFAULT_READ_MAX_CHARS))
    start_char = int(arguments.get("start_char", 0))
    skill_result = read_file_skill(
        {
            "path": str(candidate),
            "max_chars": max_chars,
            "start_char": start_char,
        },
        _commenter_skill_state(run_root, agent_id, request),
    )
    if skill_result.status in ("ok", "truncated") and candidate.name == "transcript.jsonl":
        full_text = _sanitize_transcript_content(_read_commenter_file_content(candidate))
        start_char = max(0, start_char)
        end_char = len(full_text) if max_chars <= 0 else min(len(full_text), start_char + max_chars)
        skill_result.output_text = full_text[start_char:end_char]
        skill_result.details = {
            "path": str(candidate),
            "start_char": start_char,
            "returned_chars": len(skill_result.output_text or ""),
            "total_chars": len(full_text),
        }
        skill_result.summary = (
            f"Read {len(skill_result.output_text or '')} character(s) from {candidate} starting at offset {start_char}."
            + (" Content was truncated." if end_char < len(full_text) else "")
        )
    return _compact_skill_result(skill_result)


def _commenter_search_in_files(
    *,
    run_root: str,
    agent_id: str,
    request: HarnessRequest,
    manifest: list[dict[str, Any]],
    arguments: dict[str, Any],
) -> dict[str, Any]:
    workspace_root = agent_workspace_paths(run_root, agent_id)["workspace"]
    raw_paths = arguments.get("paths")
    if isinstance(raw_paths, list):
        requested_paths = [str(item) for item in raw_paths]
    elif raw_paths is None:
        requested_paths = []
    else:
        requested_paths = [str(raw_paths)]
    resolved_paths: list[str] = []
    if requested_paths:
        for item in requested_paths:
            candidate = _resolve_commenter_target_path(
                raw_path=item,
                manifest=manifest,
                workspace_root=workspace_root,
            )
            if _is_commenter_path_allowed(candidate, run_root=run_root, agent_id=agent_id):
                resolved_paths.append(str(candidate))
    else:
        resolved_paths = [str(root) for root in _commenter_allowed_roots(run_root, agent_id)]
        if agent_workspace_paths(run_root, agent_id)["task"].exists():
            resolved_paths.append(str(agent_workspace_paths(run_root, agent_id)["task"]))
        if agent_workspace_paths(run_root, agent_id)["tools"].exists():
            resolved_paths.append(str(agent_workspace_paths(run_root, agent_id)["tools"]))
    skill_result = search_in_files_skill(
        {
            "pattern": arguments.get("pattern") or arguments.get("query") or "",
            "paths": resolved_paths,
            "max_results": int(arguments.get("max_results", 20)),
            "fixed_strings": bool(arguments.get("fixed_strings", True)),
            "ignore_case": bool(arguments.get("ignore_case", True)),
        },
        _commenter_skill_state(run_root, agent_id, request),
    )
    return _compact_skill_result(skill_result)


def _commenter_tool_specs() -> list[dict[str, Any]]:
    return [
        {
            "name": "read_file",
            "description": "Read one inspectable file by display path or exact path. Use this when you need actual file content before writing the spark.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_chars": {"type": "integer"},
                    "start_char": {"type": "integer"},
                },
            },
        },
        {
            "name": "search_in_files",
            "description": "Search inspectable files for a text pattern before deciding what to read. Paths may use display paths or exact paths.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "paths": {"type": "array", "items": {"type": "string"}},
                    "max_results": {"type": "integer"},
                    "fixed_strings": {"type": "boolean"},
                    "ignore_case": {"type": "boolean"},
                },
            },
        },
    ]


def _execute_commenter_tool_call(
    *,
    run_root: str,
    agent_id: str,
    request: HarnessRequest,
    manifest: list[dict[str, Any]],
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    if tool_name == "read_file":
        return _commenter_read_file(
            run_root=run_root,
            agent_id=agent_id,
            request=request,
            manifest=manifest,
            arguments=arguments,
        )
    if tool_name == "search_in_files":
        return _commenter_search_in_files(
            run_root=run_root,
            agent_id=agent_id,
            request=request,
            manifest=manifest,
            arguments=arguments,
        )
    return {
        "status": "failed",
        "summary": f"Unknown commenter tool '{tool_name}'.",
        "details": {"tool_name": tool_name, "arguments": arguments},
        "output_text": None,
        "error": "Unknown tool.",
    }


def _summarize_tool_call(tool_name: str, arguments: dict[str, Any], result: dict[str, Any]) -> str:
    """Lightweight summary of a commenter tool call for message history (not the trace)."""
    if tool_name == "read_file":
        path = arguments.get("path", "?")
        output_text = result.get("output_text") or ""
        char_count = len(output_text)
        return f"read {path}: {char_count} chars"
    if tool_name == "search_in_files":
        pattern = arguments.get("pattern", "?")
        paths = arguments.get("paths", [])
        output_text = result.get("output_text") or ""
        # Count lines in output as a rough match count
        match_count = output_text.count("\n") + 1 if output_text.strip() else 0
        return f"searched '{pattern}' in {len(paths)} files: ~{match_count} matches"
    return f"{tool_name}: done"


def _build_review_prompt(
    *,
    run_root: str,
    agent_id: str,
    observation_snapshot: CommenterObservationSnapshot,
) -> str:
    task_text = agent_workspace_paths(run_root, agent_id)["task"].read_text(encoding="utf-8")
    changed_entries = observation_snapshot["changed_entries"]
    lines: list[str] = []
    for item in changed_entries[:COMMENTER_MAX_RENDERED_CHANGED_ENTRIES]:
        line = f"- {item['change_type']}: {item['display_path']}"
        category = str(item.get("category") or "").strip()
        if category:
            line += f" [{category}]"
        lines.append(line)
    overflow = len(changed_entries) - len(lines)
    if overflow > 0:
        lines.append(f"- ...and {overflow} more changed file(s) within inspectable scope.")
    changed_lines = "\n".join(lines) or "None"
    scope_lines = "\n".join(f"- {entry}" for entry in observation_snapshot["inspectable_scope"]) or "None"
    return "\n".join(
        [
            "Current Task",
            task_text.strip(),
            "",
            "Changed Since Last Review",
            changed_lines,
            "",
            "Inspectable Scope",
            scope_lines,
        ]
    )


def _run_commenter_dialog(
    *,
    run_root: str,
    agent_id: str,
    request: HarnessRequest,
    manifest: list[dict[str, Any]],
    system_prompt: str,
    user_prompt: str,
    model_name: str,
    transport_name: str,
) -> tuple[str, list[dict[str, Any]]]:
    if transport_name not in {"minimax_anthropic", "minimax_openai"}:
        return (
            _invoke_text_response(
                model_name=model_name,
                transport_name=transport_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            ),
            [],
        )
    trace: list[dict[str, Any]] = []
    tool_specs = _commenter_tool_specs()
    if transport_name == "minimax_anthropic":
        client = anthropic.Anthropic(
            base_url=minimax_anthropic_base_url(),
            api_key=os.environ["MINIMAX_API_KEY"],
        )
        messages: list[dict[str, Any]] = [{"role": "user", "content": [{"type": "text", "text": user_prompt}]}]
        final_text = ""
        for step in range(1, COMMENTER_MAX_TOOL_STEPS + 1):
            response = client.messages.create(
                model=normalize_minimax_model_name(model_name),
                max_tokens=1536,
                system=system_prompt,
                messages=messages,
                tools=tool_specs,
            )
            tool_calls: list[dict[str, Any]] = []
            text_blocks: list[str] = []
            thinking_blocks = 0
            for block in response.content:
                block_type = getattr(block, "type", "")
                if block_type == "tool_use":
                    tool_calls.append(
                        {
                            "call_id": str(getattr(block, "id", "")),
                            "name": str(getattr(block, "name", "")),
                            "arguments": dict(getattr(block, "input", {}) or {}),
                        }
                    )
                elif block_type == "text":
                    text_blocks.append(str(getattr(block, "text", "")))
                elif block_type == "thinking":
                    thinking_blocks += 1
            final_text = _strip_provider_thinking("\n".join(text_blocks).strip())
            trace.append(
                {
                    "step": step,
                    "transport": transport_name,
                    "request_messages": _serialize_payload(messages),
                    "raw_response": _serialize_payload(response),
                    "tool_calls": tool_calls,
                    "text_blocks": text_blocks,
                    "provider_thinking_blocks": thinking_blocks,
                    "stop_reason": getattr(response, "stop_reason", None),
                }
            )
            if not tool_calls:
                return final_text, trace
            messages.append({"role": "assistant", "content": [_serialize_payload(block) for block in response.content]})
            # Full results for the current step so the model can read content.
            # Previous step's full results are replaced with summaries to save context.
            full_results = [
                {
                    "type": "tool_result",
                    "tool_use_id": call["call_id"],
                    "content": json.dumps(
                        _execute_commenter_tool_call(
                            run_root=run_root,
                            agent_id=agent_id,
                            request=request,
                            manifest=manifest,
                            tool_name=call["name"],
                            arguments=call["arguments"],
                        ),
                        ensure_ascii=True,
                    ),
                }
                for call in tool_calls
            ]
            trace[-1]["tool_results"] = full_results
            # Replace previous step's full results with summaries
            if len(trace) >= 2:
                prev_tool_calls = trace[-2].get("tool_calls", [])
                prev_full_results = trace[-2].get("tool_results", [])
                if prev_full_results and prev_tool_calls:
                    summary_results = [
                        {
                            "type": "tool_result",
                            "tool_use_id": prev_call["call_id"],
                            "content": _summarize_tool_call(
                                prev_call["name"],
                                prev_call["arguments"],
                                json.loads(prev_full_results[i]["content"]),
                            ),
                        }
                        for i, prev_call in enumerate(prev_tool_calls)
                        if i < len(prev_full_results)
                    ]
                    # The previous tool results are the last user message
                    messages[-2] = {"role": "user", "content": summary_results}
            messages.append({"role": "user", "content": full_results})
        return final_text, trace
    client = OpenAI(
        base_url=minimax_openai_base_url(),
        api_key=os.environ["MINIMAX_API_KEY"],
    )
    messages = [{"role": "user", "content": user_prompt}]
    final_text = ""
    for step in range(1, COMMENTER_MAX_TOOL_STEPS + 1):
        request_messages = [{"role": "system", "content": system_prompt}, *messages]
        response = client.chat.completions.create(
            model=normalize_minimax_model_name(model_name),
            messages=request_messages,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": spec["name"],
                        "description": spec["description"],
                        "parameters": spec["input_schema"],
                    },
                }
                for spec in tool_specs
            ],
            tool_choice="auto",
            max_tokens=1536,
            extra_body={"reasoning_split": True},
        )
        choice = response.choices[0]
        raw_tool_calls = getattr(choice.message, "tool_calls", None) or []
        tool_calls: list[dict[str, Any]] = []
        for tool_call in raw_tool_calls:
            raw_arguments = getattr(tool_call.function, "arguments", "{}")
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                arguments = {}
            tool_calls.append(
                {
                    "call_id": str(getattr(tool_call, "id", "")),
                    "name": str(getattr(tool_call.function, "name", "")),
                    "arguments": arguments,
                }
            )
        content = getattr(choice.message, "content", None) or ""
        final_text = _strip_provider_thinking(str(content))
        trace.append(
            {
                "step": step,
                "transport": transport_name,
                "request_messages": request_messages,
                "raw_response": _serialize_payload(response),
                "tool_calls": tool_calls,
                "text_blocks": [final_text] if final_text else [],
                "provider_thinking_blocks": 1 if getattr(choice.message, "reasoning_content", None) else 0,
                "stop_reason": getattr(choice, "finish_reason", None),
            }
        )
        if not tool_calls:
            return final_text, trace
        messages.append({"role": "assistant", **_serialize_payload(choice.message)})
        # Full results for the current step so the model can read content.
        # Previous step's full results are replaced with summaries to save context.
        full_results: list[dict[str, Any]] = []
        for call in tool_calls:
            result = _execute_commenter_tool_call(
                run_root=run_root,
                agent_id=agent_id,
                request=request,
                manifest=manifest,
                tool_name=call["name"],
                arguments=call["arguments"],
            )
            full_results.append(
                {
                    "role": "tool",
                    "tool_call_id": call["call_id"],
                    "content": json.dumps(result, ensure_ascii=True),
                }
            )
        trace[-1]["tool_results"] = full_results
        # Replace previous step's full results with summaries
        if len(trace) >= 2:
            prev_tool_calls = trace[-2].get("tool_calls", [])
            prev_full_results = trace[-2].get("tool_results", [])
            if prev_full_results and prev_tool_calls:
                # Find where previous tool results sit in messages and replace them
                prev_summary_results = [
                    {
                        "role": "tool",
                        "tool_call_id": prev_call["call_id"],
                        "content": _summarize_tool_call(
                            prev_call["name"],
                            prev_call["arguments"],
                            json.loads(prev_full_results[i]["content"]),
                        ),
                    }
                    for i, prev_call in enumerate(prev_tool_calls)
                    if i < len(prev_full_results)
                ]
                # Previous tool results are the N messages before the last assistant message
                # (last assistant = messages[-1], previous tools = messages[-1-N:-1])
                n_prev = len(prev_full_results)
                replace_start = len(messages) - 1 - n_prev
                if replace_start >= 0:
                    messages[replace_start:len(messages) - 1] = prev_summary_results
        messages.extend(full_results)
    return final_text, trace


def _record_commenter_turn(
    *,
    run_root: str,
    agent_id: str,
    state: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    response_text: str,
    trace: list[dict[str, Any]] | None = None,
) -> None:
    turns_root = agent_workspace_paths(run_root, agent_id)["commenter_turns_root"]
    turns_root.mkdir(parents=True, exist_ok=True)
    turn_index = int(state.get("turn_index", 0)) + 1
    write_text_atomic(turns_root / f"{turn_index:04d}_system_prompt.md", system_prompt)
    write_text_atomic(turns_root / f"{turn_index:04d}_input.md", user_prompt)
    write_text_atomic(turns_root / f"{turn_index:04d}_response.md", response_text)
    if trace is not None:
        write_text_atomic(
            turns_root / f"{turn_index:04d}_trace.json",
            json.dumps(trace, indent=2, ensure_ascii=True),
        )
    state["turn_index"] = turn_index


def refresh_commenter_for_agent(
    run_root: str,
    agent_id: str,
    request: HarnessRequest,
    *,
    model_name: str | None = None,
    transport_name: str | None = None,
    observation_snapshot: CommenterObservationSnapshot | None = None,
) -> int:
    if read_status(run_root, agent_id) in COMMENTER_TERMINAL_STATUSES:
        return 0
    state = load_commenter_state(run_root, agent_id) or {}
    resolved_model = model_name or get_model("harness", "agent")
    resolved_transport = transport_name or resolve_agent_transport(request.agent_transport, resolved_model)
    snapshot = observation_snapshot or build_commenter_observation_snapshot(
        run_root,
        agent_id,
        base_manifest=_normalize_manifest_entries(state.get("last_commented_manifest")),
    )
    manifest = _normalize_manifest_entries(snapshot["target_manifest"])
    review_input = _build_review_prompt(
        run_root=run_root,
        agent_id=agent_id,
        observation_snapshot=snapshot,
    )
    prompt_bundle = build_commenter_prompt_bundle(
        tools_available=resolved_transport in {"minimax_anthropic", "minimax_openai"},
        user_prompt=review_input,
    )
    response_text, trace = _run_commenter_dialog(
        run_root=run_root,
        agent_id=agent_id,
        request=request,
        manifest=manifest,
        system_prompt=prompt_bundle.system_prompt,
        user_prompt=prompt_bundle.user_prompt,
        model_name=resolved_model,
        transport_name=resolved_transport,
    )
    if read_status(run_root, agent_id) in COMMENTER_TERMINAL_STATUSES:
        return 0
    _record_commenter_turn(
        run_root=run_root,
        agent_id=agent_id,
        state=state,
        system_prompt=prompt_bundle.system_prompt,
        user_prompt=prompt_bundle.user_prompt,
        response_text=response_text,
        trace=trace,
    )
    written = append_commenter_comments(
        run_root,
        agent_id,
        [response_text] if response_text.strip() else [],
    )
    state.update(
        {
            "last_commented_manifest": manifest,
            "last_commented_fingerprint": str(snapshot["target_fingerprint"]),
            "last_refreshed_at": _utc_now_iso(),
            "last_error": "",
        }
    )
    save_commenter_state(run_root, agent_id, state)
    return written


def build_comment_feed_message(run_root: str, agent_id: str) -> tuple[str | None, int]:
    unread, total_unread = unread_commenter_comments(
        run_root,
        agent_id,
        limit=COMMENTER_MAX_FEED_COMMENTS,
    )
    if not unread:
        return None, 0
    lines = [
        "Comment Feed",
        "Outside-angle comments from your paired commenter. These are advisory, not instructions. Choose yourself whether to follow them.",
        "",
    ]
    for row in unread:
        generated_at = str(row.get("generated_at") or "")
        content = str(row.get("content") or "")
        if not content.strip():
            continue
        lines.append(f"[{generated_at}]")
        lines.append(content.rstrip("\n"))
        lines.append("")
    remaining = max(0, total_unread - len(unread))
    if remaining:
        lines.extend(["", f"...and {remaining} older unread comments not shown."])
    return "\n".join(lines).strip(), len(unread)
