"""Artifact discovery/copying for research-platform investment memos."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.harness.artifacts import agent_workspace_paths, root_agent_record

REQUIRED_ROOT_PUBLISH = {"final": "final.md", "source_use_table": "source_use_table.md"}
OPTIONAL_ROOT_PUBLISH = {
    "execution_plan": "execution_plan.md",
    "work_products_manifest": "work_products_manifest.md",
    "integration_notes": "integration_notes.md",
    "revision_report": "revision_report.md",
}
ROOT_AGENT_ID = "agent_root"
FINAL_BYTE_FLOOR = 500


class ArtifactReason(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str
    severity: Literal["info", "warn", "error"] = "warn"
    message: str


class MemoArtifactPaths(BaseModel):
    """Paths discovered/copied for the memo product and evaluator binder."""

    model_config = ConfigDict(extra="forbid")

    run_root: str | None = None
    root_agent_id: str | None = None
    root_publish_dir: str | None = None
    root_final_path: str | None = None
    root_source_use_table_path: str | None = None
    harness_tool_calls_path: str | None = None
    product_final_path: str | None = None
    product_source_use_table_path: str | None = None
    optional_root_publish_paths: dict[str, str] = Field(default_factory=dict)


class MemoArtifactSet(BaseModel):
    """Result of collecting root-published memo artifacts."""

    model_config = ConfigDict(extra="forbid")

    paths: MemoArtifactPaths
    has_usable_final: bool
    has_required_artifacts: bool
    missing_required: list[str] = Field(default_factory=list)
    optional_present: list[str] = Field(default_factory=list)
    reasons: list[ArtifactReason] = Field(default_factory=list)


def resolve_root_publish_dir(*, run_root: str | Path, root_agent_path: str | Path | None = None) -> tuple[Path, str]:
    """Resolve the root agent publish directory.

    Prefer the harness registry/root record when present, falling back to the
    canonical ``agent_root`` workspace and then to ``root_agent_path`` supplied
    on ``HarnessResponse`` for older/fake tests.
    """

    root = Path(run_root)
    record = root_agent_record(root) if root.exists() else None
    if record is not None:
        return Path(record.workspace_path) / "publish", record.agent_id
    canonical = agent_workspace_paths(root, ROOT_AGENT_ID)
    if canonical["publish_root"].exists():
        return canonical["publish_root"], ROOT_AGENT_ID
    if root_agent_path:
        path = Path(root_agent_path)
        if path.name == "publish":
            return path, path.parent.name or ROOT_AGENT_ID
        return path / "publish", path.name or ROOT_AGENT_ID
    return canonical["publish_root"], ROOT_AGENT_ID


def _is_nonempty_file(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def collect_memo_artifacts(*, run_root: str | Path, root_agent_path: str | Path | None = None) -> MemoArtifactSet:
    run_root_path = Path(run_root)
    publish_dir, root_agent_id = resolve_root_publish_dir(run_root=run_root_path, root_agent_path=root_agent_path)
    paths = MemoArtifactPaths(
        run_root=str(run_root_path),
        root_agent_id=root_agent_id,
        root_publish_dir=str(publish_dir),
    )
    reasons: list[ArtifactReason] = []
    missing: list[str] = []

    final_path = publish_dir / REQUIRED_ROOT_PUBLISH["final"]
    if final_path.exists() and final_path.is_file():
        size = final_path.stat().st_size
        if size >= FINAL_BYTE_FLOOR:
            paths.root_final_path = str(final_path)
        else:
            missing.append("final.md")
            reasons.append(
                ArtifactReason(
                    code="final_below_byte_floor",
                    severity="error",
                    message=f"Root publish/final.md is {size} bytes, <{FINAL_BYTE_FLOOR} byte floor.",
                )
            )
    else:
        missing.append("final.md")
        reasons.append(ArtifactReason(code="missing_final", severity="error", message="Root publish/final.md is missing."))

    source_use_table = publish_dir / REQUIRED_ROOT_PUBLISH["source_use_table"]
    if _is_nonempty_file(source_use_table):
        paths.root_source_use_table_path = str(source_use_table)
    else:
        missing.append("source_use_table.md")
        reasons.append(
            ArtifactReason(
                code="missing_source_use_table",
                severity="error",
                message="Root publish/source_use_table.md is missing or empty.",
            )
        )

    tool_calls = agent_workspace_paths(run_root_path, root_agent_id)["tool_calls_log"]
    if _is_nonempty_file(tool_calls):
        paths.harness_tool_calls_path = str(tool_calls)
    else:
        reasons.append(
            ArtifactReason(
                code="missing_tool_calls_log",
                severity="warn",
                message="Root harness tool_calls.jsonl is missing or empty; evaluator will record a trace gap but this alone does not downgrade pass.",
            )
        )

    optional_present: list[str] = []
    for key, filename in OPTIONAL_ROOT_PUBLISH.items():
        candidate = publish_dir / filename
        if _is_nonempty_file(candidate):
            paths.optional_root_publish_paths[filename] = str(candidate)
            optional_present.append(filename)

    has_usable_final = paths.root_final_path is not None
    has_required = has_usable_final and paths.root_source_use_table_path is not None
    return MemoArtifactSet(
        paths=paths,
        has_usable_final=has_usable_final,
        has_required_artifacts=has_required,
        missing_required=missing,
        optional_present=optional_present,
        reasons=reasons,
    )


def copy_product_artifacts(*, artifacts: MemoArtifactSet, memo_dir: str | Path) -> MemoArtifactSet:
    """Copy root-published product artifacts into the research memo directory."""

    target_dir = Path(memo_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    paths = artifacts.paths.model_copy(deep=True)
    if paths.root_final_path:
        destination = target_dir / "final.md"
        shutil.copy2(Path(paths.root_final_path), destination)
        paths.product_final_path = str(destination)
    if paths.root_source_use_table_path:
        destination = target_dir / "source_use_table.md"
        shutil.copy2(Path(paths.root_source_use_table_path), destination)
        paths.product_source_use_table_path = str(destination)
    return artifacts.model_copy(update={"paths": paths})
