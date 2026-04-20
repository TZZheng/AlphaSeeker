"""Filesystem protocol and registry helpers for the file-based harness kernel."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable
from uuid import uuid4

from src.harness.types import AgentEvent, AgentRecord, AGENT_STATUSES, HarnessRequest, HarnessState
from src.shared.report_filename import build_prompt_report_filename

try:
    import fcntl
except ImportError:  # pragma: no cover - Unix is expected in this environment.
    fcntl = None  # type: ignore[assignment]


REGISTRY_DIR = "registry"
AGENTS_DIR = "agents"
OBJECTS_DIR = "objects"
REQUEST_FILE = "request.json"
PROGRESS_FILE = "progress.md"
AGENTS_REGISTRY_FILE = "agents.jsonl"
EVENTS_REGISTRY_FILE = "events.jsonl"
OBJECTS_MANIFEST_FILE = "objects_manifest.jsonl"
HEARTBEAT_INTERVAL_SECONDS = 10


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso_to_epoch(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def _default_run_id(prompt: str) -> str:
    stem = build_prompt_report_filename(
        prompt_text=prompt,
        fallback_stem="harness_run",
        suffix="kernel",
    )
    return stem.removesuffix(".md")


@contextmanager
def _locked_file(path: Path, mode: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open(mode, encoding="utf-8") as fh:
        if fcntl is not None:
            lock_type = fcntl.LOCK_EX
            fcntl.flock(fh.fileno(), lock_type)
        try:
            yield fh
        finally:
            if fcntl is not None:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def write_text_atomic(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(target.parent)) as tmp:
        tmp.write(text)
        temp_path = Path(tmp.name)
    os.replace(temp_path, target)


def write_json_atomic(path: str | Path, payload: Any) -> None:
    write_text_atomic(path, json.dumps(payload, indent=2, ensure_ascii=True))


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    line = json.dumps(payload, ensure_ascii=True) + "\n"
    with _locked_file(target, "a") as fh:
        fh.write(line)
        fh.flush()


def read_text(path: str | Path) -> str:
    target = Path(path)
    if not target.exists():
        return ""
    return target.read_text(encoding="utf-8")


def read_json(path: str | Path) -> Any:
    text = read_text(path)
    if not text.strip():
        return None
    return json.loads(text)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    rows: list[dict[str, Any]] = []
    for raw_line in target.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def build_run_root(request: HarnessRequest) -> Path:
    if request.resume_from_run_root:
        return Path(request.resume_from_run_root)
    run_id = request.run_id or _default_run_id(request.user_prompt)
    return Path.cwd() / "data" / "harness_runs" / run_id


def registry_paths(run_root: str | Path) -> dict[str, Path]:
    root = Path(run_root)
    registry_root = root / REGISTRY_DIR
    return {
        "run_root": root,
        "registry_root": registry_root,
        "agents_registry": registry_root / AGENTS_REGISTRY_FILE,
        "events_registry": registry_root / EVENTS_REGISTRY_FILE,
        "objects_root": registry_root / OBJECTS_DIR,
        "objects_manifest": registry_root / OBJECTS_MANIFEST_FILE,
        "request": root / REQUEST_FILE,
        "progress": root / PROGRESS_FILE,
    }


def agent_workspace_paths(run_root: str | Path, agent_id: str) -> dict[str, Path]:
    workspace = Path(run_root) / AGENTS_DIR / agent_id
    publish = workspace / "publish"
    scratch = workspace / "scratch"
    commenter = scratch / "commenter"
    commenter_notes = commenter / "notes"
    commenter_turns = commenter / "turns"
    state = workspace / "state"
    context_root = workspace / "context"
    return {
        "workspace": workspace,
        "task": workspace / "task.md",
        "tools": workspace / "tools.md",
        "publish_root": publish,
        "publish_summary": publish / "summary.md",
        "publish_index": publish / "artifact_index.md",
        "publish_final": publish / "final.md",
        "scratch_root": scratch,
        "commenter_root": commenter,
        "commenter_notes_root": commenter_notes,
        "commenter_turns_root": commenter_turns,
        "commenter_comments": commenter / "comments.jsonl",
        "commenter_latest": commenter / "latest.md",
        "journal": scratch / "journal.jsonl",
        "transcript": scratch / "transcript.jsonl",
        "tool_history": scratch / "tool_history.jsonl",
        "skill_state": state / "skill_state.json",
        "transport_state": state / "transport_state.json",
        "commenter_state": state / "commenter_state.json",
        "prompt_memory": state / "prompt_memory.md",
        "history_summary": state / "history_summary.md",
        "state_root": state,
        "status": state / "status.txt",
        "heartbeat": state / "heartbeat.txt",
        "pid": state / "pid.txt",
        "parent": state / "parent.txt",
        "events_queue": state / "events_queue.jsonl",
        "preset": state / "preset.txt",
        "context_root": context_root,
    }


def build_reduction_paths(workspace: str | Path) -> dict[str, str]:
    scratch_root = agent_workspace_paths(Path(workspace).parents[1], Path(workspace).name)["scratch_root"]
    reduction_root = scratch_root / "reduction"
    reduction_root.mkdir(parents=True, exist_ok=True)
    return {
        "discovered_sources": str(reduction_root / "discovered_sources.json"),
        "read_queue": str(reduction_root / "read_queue.json"),
        "read_results": str(reduction_root / "read_results.json"),
        "source_cards": str(reduction_root / "source_cards.jsonl"),
        "fact_index": str(reduction_root / "fact_index.json"),
        "section_briefs": str(reduction_root / "section_briefs.json"),
        "coverage_matrix": str(reduction_root / "coverage_matrix.json"),
    }


def initialize_run_root(request: HarnessRequest) -> tuple[Path, str]:
    run_root = build_run_root(request)
    paths = registry_paths(run_root)
    paths["registry_root"].mkdir(parents=True, exist_ok=True)
    paths["objects_root"].mkdir(parents=True, exist_ok=True)
    write_json_atomic(paths["request"], request.model_dump(mode="json"))
    for path in (paths["agents_registry"], paths["events_registry"], paths["objects_manifest"]):
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            write_text_atomic(path, "")
    refresh_progress_view(run_root)
    root_agent_id = "agent_root"
    return run_root, root_agent_id


def load_request(run_root: str | Path) -> HarnessRequest:
    payload = read_json(registry_paths(run_root)["request"])
    if not isinstance(payload, dict):
        raise ValueError("Harness run request file is missing or invalid.")
    return HarnessRequest.model_validate(payload)


def create_agent_workspace(
    run_root: str | Path,
    *,
    agent_id: str,
    parent_id: str,
    preset: str,
    task_name: str,
    description: str,
    task_markdown: str,
    tools_markdown: str,
    context_files: Iterable[str] | None = None,
) -> Path:
    paths = agent_workspace_paths(run_root, agent_id)
    for key in (
        "workspace",
        "publish_root",
        "scratch_root",
        "commenter_root",
        "commenter_notes_root",
        "commenter_turns_root",
        "state_root",
        "context_root",
    ):
        paths[key].mkdir(parents=True, exist_ok=True)

    write_text_atomic(paths["task"], task_markdown.strip() + "\n")
    write_text_atomic(paths["tools"], tools_markdown.strip() + "\n")
    write_text_atomic(paths["status"], "queued\n")
    write_text_atomic(paths["heartbeat"], "")
    write_text_atomic(paths["pid"], "")
    write_text_atomic(paths["parent"], (parent_id or "") + "\n")
    write_text_atomic(paths["preset"], preset + "\n")
    write_text_atomic(paths["journal"], "")
    write_text_atomic(paths["transcript"], "")
    write_text_atomic(paths["tool_history"], "")
    write_text_atomic(paths["commenter_comments"], "")
    write_text_atomic(paths["commenter_latest"], "")
    write_text_atomic(paths["prompt_memory"], "")
    write_text_atomic(paths["history_summary"], "")

    if context_files:
        for source_path in context_files:
            source = Path(source_path)
            if not source.exists() or not source.is_file():
                continue
            dest = _unique_destination(paths["context_root"], source.name)
            _copy_or_link(source, dest)

    append_agent_record(
        run_root,
        AgentRecord(
            agent_id=agent_id,
            parent_id=parent_id,
            preset=preset,  # type: ignore[arg-type]
            workspace_path=str(paths["workspace"]),
            task_name=task_name,
            description=description,
            status="queued",
        ),
    )
    append_event(
        run_root,
        AgentEvent(
            event_type="workspace_created",
            agent_id=agent_id,
            parent_id=parent_id,
            details={"task_name": task_name, "description": description, "preset": preset},
        ),
    )
    refresh_progress_view(run_root)
    return paths["workspace"]


def _unique_destination(directory: Path, name: str) -> Path:
    stem = Path(name).stem
    suffix = Path(name).suffix
    candidate = directory / name
    index = 1
    while candidate.exists():
        candidate = directory / f"{stem}_{index}{suffix}"
        index += 1
    return candidate


def _copy_or_link(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, dest)
    except OSError:
        shutil.copy2(source, dest)


def append_agent_record(run_root: str | Path, record: AgentRecord) -> None:
    append_jsonl(registry_paths(run_root)["agents_registry"], record.model_dump(mode="json"))


def append_event(run_root: str | Path, event: AgentEvent) -> None:
    append_jsonl(registry_paths(run_root)["events_registry"], event.model_dump(mode="json"))


def latest_agent_records(run_root: str | Path) -> dict[str, AgentRecord]:
    latest: dict[str, AgentRecord] = {}
    for row in read_jsonl(registry_paths(run_root)["agents_registry"]):
        record = AgentRecord.model_validate(row)
        latest[record.agent_id] = record
    return latest


def root_agent_record(run_root: str | Path) -> AgentRecord | None:
    records = latest_agent_records(run_root)
    if "agent_root" in records:
        return records["agent_root"]
    for record in records.values():
        if not record.parent_id:
            return record
    return None


def agent_started_epoch(run_root: str | Path, agent_id: str) -> float | None:
    record = latest_agent_records(run_root).get(agent_id)
    if record is None:
        return None
    return _iso_to_epoch(record.started_at) or _iso_to_epoch(record.created_at)


def run_started_epoch(run_root: str | Path) -> float | None:
    root = root_agent_record(run_root)
    if root is None:
        return None
    return _iso_to_epoch(root.started_at) or _iso_to_epoch(root.created_at)


def root_time_limit_seconds(request: HarnessRequest) -> int:
    return request.root_wall_clock_seconds or request.wall_clock_budget_seconds


def effective_agent_time_limit_seconds(
    request: HarnessRequest,
    run_root: str | Path,
    agent_id: str,
) -> int:
    record = latest_agent_records(run_root).get(agent_id)
    if record is None:
        return request.per_agent_wall_clock_seconds
    if not record.parent_id:
        return root_time_limit_seconds(request)
    return request.per_agent_wall_clock_seconds


def remaining_run_seconds(
    request: HarnessRequest,
    run_root: str | Path,
    *,
    now_epoch: float | None = None,
) -> int:
    started = run_started_epoch(run_root)
    if started is None:
        return request.wall_clock_budget_seconds
    now = now_epoch or datetime.now(timezone.utc).timestamp()
    elapsed = max(0.0, now - started)
    return max(0, int(request.wall_clock_budget_seconds - elapsed))


def remaining_root_seconds(
    request: HarnessRequest,
    run_root: str | Path,
    *,
    now_epoch: float | None = None,
) -> int:
    started = run_started_epoch(run_root)
    root_limit = root_time_limit_seconds(request)
    if started is None:
        return min(root_limit, request.wall_clock_budget_seconds)
    now = now_epoch or datetime.now(timezone.utc).timestamp()
    elapsed = max(0.0, now - started)
    remaining_root = max(0, int(root_limit - elapsed))
    return min(remaining_root, remaining_run_seconds(request, run_root, now_epoch=now))


def remaining_agent_seconds(
    request: HarnessRequest,
    run_root: str | Path,
    agent_id: str,
    *,
    now_epoch: float | None = None,
) -> int:
    started = agent_started_epoch(run_root, agent_id)
    agent_limit = effective_agent_time_limit_seconds(request, run_root, agent_id)
    now = now_epoch or datetime.now(timezone.utc).timestamp()
    if started is None:
        remaining_agent = agent_limit
    else:
        elapsed = max(0.0, now - started)
        remaining_agent = max(0, int(agent_limit - elapsed))
    return min(remaining_agent, remaining_run_seconds(request, run_root, now_epoch=now))


def update_agent_record(
    run_root: str | Path,
    *,
    agent_id: str,
    status: str | None = None,
    pid: int | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
    error: str | None = None,
) -> AgentRecord:
    records = latest_agent_records(run_root)
    existing = records.get(agent_id)
    if existing is None:
        raise ValueError(f"Unknown agent id '{agent_id}'.")
    updated = existing.model_copy(
        update={
            "status": status or existing.status,
            "pid": pid if pid is not None else existing.pid,
            "started_at": started_at if started_at is not None else existing.started_at,
            "finished_at": finished_at if finished_at is not None else existing.finished_at,
            "error": error if error is not None else existing.error,
            "updated_at": _utc_now_iso(),
        }
    )
    append_agent_record(run_root, updated)
    refresh_progress_view(run_root)
    return updated


def write_status(run_root: str | Path, agent_id: str, status: str) -> None:
    if status not in AGENT_STATUSES:
        raise ValueError(f"Illegal agent status '{status}'.")
    paths = agent_workspace_paths(run_root, agent_id)
    write_text_atomic(paths["status"], status + "\n")


def read_status(run_root: str | Path, agent_id: str) -> str:
    status = read_text(agent_workspace_paths(run_root, agent_id)["status"]).strip()
    return status or "queued"


def write_heartbeat(run_root: str | Path, agent_id: str, timestamp: str | None = None) -> None:
    paths = agent_workspace_paths(run_root, agent_id)
    write_text_atomic(paths["heartbeat"], (timestamp or _utc_now_iso()) + "\n")


def read_heartbeat(run_root: str | Path, agent_id: str) -> str:
    return read_text(agent_workspace_paths(run_root, agent_id)["heartbeat"]).strip()


def write_pid(run_root: str | Path, agent_id: str, pid: int) -> None:
    write_text_atomic(agent_workspace_paths(run_root, agent_id)["pid"], f"{pid}\n")


def load_skill_state(run_root: str | Path, agent_id: str) -> HarnessState | None:
    path = agent_workspace_paths(run_root, agent_id)["skill_state"]
    payload = read_json(path)
    if not isinstance(payload, dict):
        return None
    return HarnessState.model_validate(payload)


def save_skill_state(state: HarnessState) -> None:
    if not state.run_root or not state.agent_id:
        raise ValueError("HarnessState is missing run_root or agent_id.")
    path = agent_workspace_paths(state.run_root, state.agent_id)["skill_state"]
    write_json_atomic(path, state.model_dump(mode="json"))


def load_transport_state(run_root: str | Path, agent_id: str) -> dict[str, Any] | None:
    payload = read_json(agent_workspace_paths(run_root, agent_id)["transport_state"])
    return payload if isinstance(payload, dict) else None


def save_transport_state(run_root: str | Path, agent_id: str, payload: dict[str, Any]) -> None:
    write_json_atomic(agent_workspace_paths(run_root, agent_id)["transport_state"], payload)


def load_commenter_state(run_root: str | Path, agent_id: str) -> dict[str, Any] | None:
    payload = read_json(agent_workspace_paths(run_root, agent_id)["commenter_state"])
    return payload if isinstance(payload, dict) else None


def save_commenter_state(run_root: str | Path, agent_id: str, payload: dict[str, Any]) -> None:
    write_json_atomic(agent_workspace_paths(run_root, agent_id)["commenter_state"], payload)


def load_commenter_comments(run_root: str | Path, agent_id: str) -> list[dict[str, Any]]:
    return read_jsonl(agent_workspace_paths(run_root, agent_id)["commenter_comments"])


def _render_commenter_note(*, generated_at: str, read: bool, content: str) -> str:
    status = "read" if read else "unread"
    body = content.rstrip("\n")
    return "\n".join(
        [
            f"generated_at: {generated_at}",
            f"status: {status}",
            "",
            body,
            "",
        ]
    )


def _write_commenter_note(
    run_root: str | Path,
    agent_id: str,
    *,
    generated_at: str,
    read: bool,
    content: str,
    note_path: str | Path | None = None,
) -> str:
    paths = agent_workspace_paths(run_root, agent_id)
    notes_root = paths["commenter_notes_root"]
    notes_root.mkdir(parents=True, exist_ok=True)
    target = Path(note_path) if note_path else notes_root / f"{uuid4().hex[:12]}.md"
    write_text_atomic(
        target,
        _render_commenter_note(generated_at=generated_at, read=read, content=content),
    )
    return str(target)


def append_commenter_comments(
    run_root: str | Path,
    agent_id: str,
    comments: Iterable[str],
    *,
    generated_at: str | None = None,
) -> int:
    path = agent_workspace_paths(run_root, agent_id)["commenter_comments"]
    written = 0
    for raw_comment in comments:
        content = str(raw_comment)
        if not content.strip():
            continue
        comment_generated_at = generated_at or _utc_now_iso()
        note_path = _write_commenter_note(
            run_root,
            agent_id,
            generated_at=comment_generated_at,
            read=False,
            content=content,
        )
        append_jsonl(
            path,
            {
                "generated_at": comment_generated_at,
                "read": False,
                "content": content,
                "note_path": note_path,
            },
        )
        written += 1
    if written:
        refresh_commenter_latest(run_root, agent_id)
    return written


def unread_commenter_comments(
    run_root: str | Path,
    agent_id: str,
    *,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], int]:
    unread = [row for row in load_commenter_comments(run_root, agent_id) if not bool(row.get("read"))]
    total = len(unread)
    if limit is not None:
        unread = unread[:limit]
    return unread, total


def mark_commenter_comments_read(run_root: str | Path, agent_id: str, count: int) -> int:
    if count <= 0:
        return 0
    path = agent_workspace_paths(run_root, agent_id)["commenter_comments"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    marked = 0
    with _locked_file(path, "r+") as fh:
        rows: list[dict[str, Any]] = []
        for raw_line in fh.read().splitlines():
            line = raw_line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        remaining = count
        for row in rows:
            if remaining <= 0:
                break
            if bool(row.get("read")):
                continue
            row["read"] = True
            note_path = str(row.get("note_path") or "").strip()
            if note_path:
                _write_commenter_note(
                    run_root,
                    agent_id,
                    generated_at=str(row.get("generated_at") or _utc_now_iso()),
                    read=True,
                    content=str(row.get("content") or ""),
                    note_path=note_path,
                )
            marked += 1
            remaining -= 1
        fh.seek(0)
        fh.truncate()
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")
        fh.flush()
    if marked:
        refresh_commenter_latest(run_root, agent_id)
    return marked


def refresh_commenter_latest(run_root: str | Path, agent_id: str, *, limit: int = 20) -> None:
    rows = load_commenter_comments(run_root, agent_id)[-limit:]
    if not rows:
        write_text_atomic(agent_workspace_paths(run_root, agent_id)["commenter_latest"], "")
        return
    lines = ["# Latest Comments", ""]
    for row in rows:
        timestamp = str(row.get("generated_at") or "")
        content = str(row.get("content") or "")
        status = "read" if bool(row.get("read")) else "unread"
        if not content.strip():
            continue
        lines.append(f"## [{timestamp}] [{status}]")
        lines.append("")
        lines.append(content.rstrip("\n"))
        lines.append("")
    write_text_atomic(
        agent_workspace_paths(run_root, agent_id)["commenter_latest"],
        "\n".join(lines).strip() + "\n",
    )


def load_transcript_entries(run_root: str | Path, agent_id: str) -> list[dict[str, Any]]:
    return read_jsonl(agent_workspace_paths(run_root, agent_id)["transcript"])


def append_transcript_entry(run_root: str | Path, agent_id: str, payload: dict[str, Any]) -> None:
    append_jsonl(agent_workspace_paths(run_root, agent_id)["transcript"], payload)


def append_tool_history(run_root: str | Path, agent_id: str, payload: dict[str, Any]) -> None:
    append_jsonl(agent_workspace_paths(run_root, agent_id)["tool_history"], payload)


def promote_object(
    run_root: str | Path,
    *,
    source_path: str,
    description: str,
    agent_id: str,
) -> dict[str, str]:
    source = Path(source_path)
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(source_path)
    object_id = f"obj_{uuid4().hex[:12]}"
    registry = registry_paths(run_root)
    target = registry["objects_root"] / f"{object_id}_{source.name}"
    _copy_or_link(source, target)
    manifest_row = {
        "object_id": object_id,
        "agent_id": agent_id,
        "description": description,
        "source_path": str(source),
        "object_path": str(target),
        "created_at": _utc_now_iso(),
    }
    append_jsonl(registry["objects_manifest"], manifest_row)
    append_event(
        run_root,
        AgentEvent(
            event_type="artifact_promoted",
            agent_id=agent_id,
            details={"object_id": object_id, "object_path": str(target), "description": description},
        ),
    )
    return {"object_id": object_id, "object_path": str(target), "description": description}


def load_object_manifest(run_root: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(registry_paths(run_root)["objects_manifest"])


def refresh_progress_view(run_root: str | Path) -> None:
    records = latest_agent_records(run_root)
    by_parent: dict[str, list[AgentRecord]] = {}
    for record in records.values():
        by_parent.setdefault(record.parent_id, []).append(record)

    lines = ["# Harness Progress", ""]
    if not records:
        lines.append("- No agents registered yet.")
    else:
        for record in sorted(records.values(), key=lambda item: item.agent_id):
            lines.append(
                f"- {record.agent_id} [{record.status}] preset={record.preset} "
                f"parent={record.parent_id or '-'} task={record.task_name} :: {record.description}"
            )
            workspace = Path(record.workspace_path)
            summary_path = workspace / "publish" / "summary.md"
            summary_line = _first_nonempty_line(summary_path)
            if summary_line:
                lines.append(f"  summary: {summary_line}")
            child_count = len(by_parent.get(record.agent_id, []))
            if child_count:
                lines.append(f"  children: {child_count}")
    write_text_atomic(registry_paths(run_root)["progress"], "\n".join(lines).strip() + "\n")


def _first_nonempty_line(path: Path) -> str:
    if not path.exists():
        return ""
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line:
            return line
    return ""


def sync_reduction_artifacts(state: HarnessState) -> None:
    """Persist retrieval reduction artifacts inside the agent scratch workspace."""

    if not state.workspace_path:
        return
    paths = state.dossier_paths or build_reduction_paths(state.workspace_path)
    state.dossier_paths = paths
    write_json_atomic(
        paths["discovered_sources"],
        {
            "query_buckets": [bucket.model_dump(mode="json") for bucket in state.query_buckets],
            "sources": [source.model_dump(mode="json") for source in state.discovered_sources],
        },
    )
    write_json_atomic(
        paths["read_queue"],
        {"queue": [entry.model_dump(mode="json") for entry in state.read_queue]},
    )
    write_json_atomic(
        paths["read_results"],
        {"results": [entry.model_dump(mode="json") for entry in state.read_results]},
    )
    source_card_lines = "\n".join(
        json.dumps(card.model_dump(mode="json"), ensure_ascii=True) for card in state.source_cards
    )
    write_text_atomic(paths["source_cards"], source_card_lines + ("\n" if source_card_lines else ""))
    write_json_atomic(
        paths["fact_index"],
        {"facts": [record.model_dump(mode="json") for record in state.fact_index]},
    )
    write_json_atomic(
        paths["section_briefs"],
        {"sections": [brief.model_dump(mode="json") for brief in state.section_briefs]},
    )
    write_json_atomic(
        paths["coverage_matrix"],
        state.coverage_matrix.model_dump(mode="json") if state.coverage_matrix else {},
    )


def stale_agents(run_root: str | Path, *, stale_after_seconds: int, now_epoch: float | None = None) -> list[str]:
    now = now_epoch or datetime.now(timezone.utc).timestamp()
    stale: list[str] = []
    for agent_id, record in latest_agent_records(run_root).items():
        if record.status not in {"running", "waiting"}:
            continue
        heartbeat_text = read_heartbeat(run_root, agent_id)
        if not heartbeat_text:
            continue
        try:
            heartbeat = datetime.fromisoformat(heartbeat_text)
        except ValueError:
            continue
        age = now - heartbeat.timestamp()
        if age > stale_after_seconds:
            stale.append(agent_id)
    return stale
