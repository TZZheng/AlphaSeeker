"""Filesystem protocol and registry helpers for the file-based harness kernel."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
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
REPORT_VERSIONS_DIR = "report_versions"
REQUEST_FILE = "request.json"
PROGRESS_FILE = "progress.md"
AGENTS_REGISTRY_FILE = "agents.jsonl"
EVENTS_REGISTRY_FILE = "events.jsonl"
OBJECTS_MANIFEST_FILE = "objects_manifest.jsonl"
FINAL_REPORT_VERSIONS_FILE = "final_report_versions.jsonl"
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
        "report_versions_root": registry_root / REPORT_VERSIONS_DIR,
        "final_report_versions": registry_root / FINAL_REPORT_VERSIONS_FILE,
        "request": root / REQUEST_FILE,
        "progress": root / PROGRESS_FILE,
    }


def final_report_snapshot_paths(run_root: str | Path, agent_id: str) -> dict[str, Path]:
    paths = registry_paths(run_root)
    versions_root = paths["report_versions_root"] / agent_id
    return {
        "source": agent_workspace_paths(run_root, agent_id)["publish_final"],
        "versions_root": versions_root,
        "manifest": paths["final_report_versions"],
    }


def agent_workspace_paths(run_root: str | Path, agent_id: str) -> dict[str, Path]:
    workspace = Path(run_root) / AGENTS_DIR / agent_id
    publish = workspace / "publish"
    scratch = workspace / "scratch"
    artifacts = workspace / "artifacts"
    skills_artifacts = artifacts / "skills"
    search_artifacts = artifacts / "search"
    harness = workspace / "_harness"
    harness_state = harness / "state"
    harness_logs = harness / "logs"
    llm_turns = harness / "llm_turns"
    commenter = harness / "commenter"
    commenter_notes = commenter / "notes"
    commenter_turns = commenter / "turns"
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
        "artifacts_root": artifacts,
        "skills_artifacts_root": skills_artifacts,
        "search_artifacts_root": search_artifacts,
        "harness_root": harness,
        "harness_state_root": harness_state,
        "harness_logs_root": harness_logs,
        "llm_turns_root": llm_turns,
        "commenter_root": commenter,
        "commenter_notes_root": commenter_notes,
        "commenter_turns_root": commenter_turns,
        "commenter_comments": commenter / "comments.jsonl",
        "commenter_latest": commenter / "latest.md",
        "transcript": harness_logs / "transcript.jsonl",
        "tool_calls_log": harness_logs / "tool_calls.jsonl",
        "worker_log": harness_logs / "worker.log",
        "events_queue": harness_logs / "events_queue.jsonl",
        "skill_state": harness_state / "skill_state.json",
        "transport_state": harness_state / "transport_state.json",
        "conversation": harness_state / "conversation.jsonl",
        "commenter_state": harness_state / "commenter_state.json",
        "prompt_memory": harness_state / "prompt_memory.md",
        "history_summary": harness_state / "history_summary.md",
        "status": harness_state / "status.txt",
        "heartbeat": harness_state / "heartbeat.txt",
        "pid": harness_state / "pid.txt",
        "parent": harness_state / "parent.txt",
        "preset": harness_state / "preset.txt",
        "context_root": context_root,
    }

def initialize_run_root(request: HarnessRequest) -> tuple[Path, str]:
    run_root = build_run_root(request)
    paths = registry_paths(run_root)
    paths["registry_root"].mkdir(parents=True, exist_ok=True)
    paths["objects_root"].mkdir(parents=True, exist_ok=True)
    paths["report_versions_root"].mkdir(parents=True, exist_ok=True)
    write_json_atomic(paths["request"], request.model_dump(mode="json"))
    for path in (
        paths["agents_registry"],
        paths["events_registry"],
        paths["objects_manifest"],
        paths["final_report_versions"],
    ):
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
        "artifacts_root",
        "skills_artifacts_root",
        "search_artifacts_root",
        "harness_root",
        "harness_state_root",
        "harness_logs_root",
        "llm_turns_root",
        "commenter_root",
        "commenter_notes_root",
        "commenter_turns_root",
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
    write_text_atomic(paths["transcript"], "")
    write_text_atomic(paths["conversation"], "")
    write_text_atomic(paths["tool_calls_log"], "")
    write_text_atomic(paths["events_queue"], "")
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


def snapshot_final_report_if_changed(
    run_root: str | Path,
    agent_id: str,
    *,
    trigger: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    records = latest_agent_records(run_root)
    record = records.get(agent_id)
    if record is None or record.parent_id:
        return None

    paths = final_report_snapshot_paths(run_root, agent_id)
    source = paths["source"]
    if not source.exists() or not source.is_file():
        return None
    content = source.read_text(encoding="utf-8")
    if not content:
        return None

    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    manifest_path = paths["manifest"]
    agent_rows = [row for row in read_jsonl(manifest_path) if row.get("agent_id") == agent_id]
    latest_row = agent_rows[-1] if agent_rows else None
    if latest_row and latest_row.get("sha256") == digest:
        return None

    existing_versions = [
        int(row.get("version", 0))
        for row in agent_rows
        if isinstance(row.get("version"), int) or str(row.get("version", "")).isdigit()
    ]
    version = (max(existing_versions) if existing_versions else 0) + 1
    versions_root = paths["versions_root"]
    versions_root.mkdir(parents=True, exist_ok=True)
    snapshot_path = versions_root / f"v{version:04d}.md"
    while snapshot_path.exists():
        version += 1
        snapshot_path = versions_root / f"v{version:04d}.md"

    write_text_atomic(snapshot_path, content)
    trigger_payload = trigger or {}
    manifest_row = {
        "version": version,
        "agent_id": agent_id,
        "source_path": str(source),
        "snapshot_path": str(snapshot_path),
        "sha256": digest,
        "chars": len(content),
        "created_at": _utc_now_iso(),
        "trigger_tool": str(
            trigger_payload.get("trigger_tool") or trigger_payload.get("tool") or ""
        ),
        "trigger_operation": str(
            trigger_payload.get("trigger_operation") or trigger_payload.get("operation") or ""
        ),
    }
    append_jsonl(manifest_path, manifest_row)
    append_event(
        run_root,
        AgentEvent(
            event_type="final_report_snapshot_created",
            agent_id=agent_id,
            details=manifest_row,
        ),
    )
    return manifest_row


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


DEPRECATED_SKILL_STATE_KEYS = {
    "dossier_paths",
    "query_buckets",
    "discovered_sources",
    "read_queue",
    "read_results",
    "source_cards",
    "fact_index",
    "section_briefs",
    "coverage_matrix",
    "retrieval_wave_count",
}


def _strip_deprecated_skill_state_keys(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key not in DEPRECATED_SKILL_STATE_KEYS}


def load_skill_state(run_root: str | Path, agent_id: str) -> HarnessState | None:
    path = agent_workspace_paths(run_root, agent_id)["skill_state"]
    payload = read_json(path)
    if not isinstance(payload, dict):
        return None
    payload = _strip_deprecated_skill_state_keys(payload)
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


def _missing_conversation_state_error(path: Path, run_root: str | Path, agent_id: str) -> FileNotFoundError:
    return FileNotFoundError(
        "Canonical conversation state is missing for native transport "
        f"(agent_id={agent_id!r}, run_root={str(run_root)!r}, path={str(path)!r}). "
        "This run predates the canonical conversation cutover and cannot be resumed with native transports."
    )


def load_conversation_entries(run_root: str | Path, agent_id: str) -> list[dict[str, Any]]:
    path = agent_workspace_paths(run_root, agent_id)["conversation"]
    if not path.exists():
        raise _missing_conversation_state_error(path, run_root, agent_id)
    return read_jsonl(path)


def append_conversation_entry(run_root: str | Path, agent_id: str, payload: dict[str, Any]) -> None:
    path = agent_workspace_paths(run_root, agent_id)["conversation"]
    if not path.exists():
        raise _missing_conversation_state_error(path, run_root, agent_id)
    append_jsonl(path, payload)


def append_tool_call_log(run_root: str | Path, agent_id: str, payload: dict[str, Any]) -> None:
    append_jsonl(agent_workspace_paths(run_root, agent_id)["tool_calls_log"], payload)


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
