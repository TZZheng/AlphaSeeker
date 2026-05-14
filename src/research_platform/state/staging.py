"""Staged direct-edit workflow for v3.3 research state updates."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import shutil
from typing import Any

from src.research_platform.state.storage import (
    StatePaths,
    append_jsonl,
    parse_markdown_sections,
    snapshot_state,
    state_file_lock,
)

STATE_FILE_NAMES = [
    "research_state.md",
    "state_index.json",
    "evidence_index.json",
    "open_questions.json",
    "conflicts.json",
    "valuation_snapshot.json",
]

EDITABLE_STAGE_FILE_NAMES = [
    "research_state.md",
    "open_questions.json",
    "conflicts.json",
    "valuation_snapshot.json",
]


@dataclass(frozen=True)
class StagePaths:
    """Paths for one staged research-state edit run."""

    live: StatePaths
    baseline: StatePaths
    stage: StatePaths

    @property
    def editable_files(self) -> list[Path]:
        return [self.stage.root / name for name in EDITABLE_STAGE_FILE_NAMES]

    @property
    def context_files(self) -> list[Path]:
        return [self.stage.root / name for name in STATE_FILE_NAMES]


@dataclass
class StageDiff:
    """Run-level audit summary for a staged state commit."""

    sections_added: list[str] = field(default_factory=list)
    sections_removed: list[str] = field(default_factory=list)
    sections_modified: list[str] = field(default_factory=list)
    sidecars_modified: list[str] = field(default_factory=list)
    bytes_before: int = 0
    bytes_after: int = 0
    bytes_delta: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StageCommitResult:
    """Commit result for a validated staged edit."""

    committed: bool
    diff: StageDiff

    def to_dict(self) -> dict[str, Any]:
        return {"committed": self.committed, "diff": self.diff.to_dict()}


def _copy_state_files(source: StatePaths, destination: StatePaths) -> None:
    destination.root.mkdir(parents=True, exist_ok=True)
    for name in STATE_FILE_NAMES:
        source_path = source.root / name
        if source_path.exists():
            shutil.copy2(source_path, destination.root / name)


def prepare_state_stage(*, live_paths: StatePaths, baseline_root: str | Path, stage_root: str | Path) -> StagePaths:
    """Snapshot live state and create a writable staged copy for an LLM StateOwner."""

    baseline = StatePaths(Path(baseline_root))
    stage = StatePaths(Path(stage_root))
    snapshot_state(live_paths, baseline.root)
    if stage.root.exists():
        shutil.rmtree(stage.root)
    _copy_state_files(baseline, stage)
    return StagePaths(live=live_paths, baseline=baseline, stage=stage)


def _file_bytes(paths: StatePaths) -> int:
    total = 0
    for name in STATE_FILE_NAMES:
        path = paths.root / name
        if path.exists():
            total += len(path.read_bytes())
    return total


def _json_entry_count(path: Path, key: str) -> int:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    value = payload.get(key) if isinstance(payload, dict) else None
    return len(value) if isinstance(value, list) else 0


def compute_stage_diff(stage_paths: StagePaths) -> StageDiff:
    """Compute a compact run-level diff between baseline and staged state."""

    before_md = stage_paths.baseline.research_state.read_text(encoding="utf-8") if stage_paths.baseline.research_state.exists() else ""
    after_md = stage_paths.stage.research_state.read_text(encoding="utf-8") if stage_paths.stage.research_state.exists() else ""
    before_sections = {section.section_key: section.body for section in parse_markdown_sections(before_md)}
    after_sections = {section.section_key: section.body for section in parse_markdown_sections(after_md)}
    before_keys = set(before_sections)
    after_keys = set(after_sections)

    sidecars_modified: list[str] = []
    for name in ["open_questions.json", "conflicts.json", "valuation_snapshot.json"]:
        before = stage_paths.baseline.root / name
        after = stage_paths.stage.root / name
        if before.exists() and after.exists() and before.read_text(encoding="utf-8") != after.read_text(encoding="utf-8"):
            sidecars_modified.append(name)
        elif before.exists() != after.exists():
            sidecars_modified.append(name)

    before_bytes = _file_bytes(stage_paths.baseline)
    after_bytes = _file_bytes(stage_paths.stage)
    return StageDiff(
        sections_added=sorted(after_keys - before_keys),
        sections_removed=sorted(before_keys - after_keys),
        sections_modified=sorted(key for key in before_keys & after_keys if before_sections[key].strip() != after_sections[key].strip()),
        sidecars_modified=sidecars_modified,
        bytes_before=before_bytes,
        bytes_after=after_bytes,
        bytes_delta=after_bytes - before_bytes,
    )


def _atomic_copy_text(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    tmp.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    tmp.replace(destination)


def commit_stage(*, stage_paths: StagePaths, vault_root: str | Path, ticker: str, run_id: str, diff: StageDiff | None = None) -> StageCommitResult:
    """Atomically commit a validated staged state folder into the live state folder."""

    diff = diff or compute_stage_diff(stage_paths)
    with state_file_lock(vault_root, ticker) as locked_paths:
        if locked_paths.root.resolve() != stage_paths.live.root.resolve():
            raise RuntimeError("state lock path mismatch")
        for name in STATE_FILE_NAMES:
            source = stage_paths.stage.root / name
            if source.exists():
                _atomic_copy_text(source, locked_paths.root / name)
        append_jsonl(
            locked_paths.diff_log,
            {
                "run_id": run_id,
                "event": "v33_stage_commit",
                "diff": diff.to_dict(),
            },
        )
    return StageCommitResult(committed=True, diff=diff)
