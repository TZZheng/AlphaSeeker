"""Agent-visible filesystem policy for harness workspaces."""

from __future__ import annotations

from pathlib import Path

from src.harness.artifacts import agent_workspace_paths, latest_agent_records


class VisibilityError(ValueError):
    """Raised when a tool path is outside the agent-visible surface."""


def _is_within(root: Path, candidate: Path) -> bool:
    resolved_root = root.resolve(strict=False)
    resolved_candidate = candidate.resolve(strict=False)
    return resolved_candidate == resolved_root or resolved_root in resolved_candidate.parents


def _resolve_candidate(run_root: str | Path, agent_id: str, raw_path: str, *, cwd: str | Path | None = None) -> Path:
    paths = agent_workspace_paths(run_root, agent_id)
    raw = str(raw_path or "").strip()
    if not raw:
        raise VisibilityError("Path is required.")
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        return candidate.resolve(strict=False)

    parts = candidate.parts
    if len(parts) >= 3 and parts[0].startswith("agent_") and parts[1] == "publish":
        return (Path(run_root) / "agents" / candidate).resolve(strict=False)

    base = Path(cwd).expanduser() if cwd is not None else paths["workspace"]
    if not base.is_absolute():
        base = paths["workspace"] / base
    return (base / candidate).resolve(strict=False)


def _direct_child_publish_roots(run_root: str | Path, agent_id: str) -> list[Path]:
    roots: list[Path] = []
    for record in latest_agent_records(run_root).values():
        if record.parent_id == agent_id:
            roots.append(agent_workspace_paths(run_root, record.agent_id)["publish_root"])
    return roots


def visible_read_roots(run_root: str | Path, agent_id: str) -> list[Path]:
    paths = agent_workspace_paths(run_root, agent_id)
    return [
        paths["context_root"],
        paths["publish_root"],
        paths["scratch_root"],
        *_direct_child_publish_roots(run_root, agent_id),
    ]


def default_search_targets(run_root: str | Path, agent_id: str) -> list[Path]:
    paths = agent_workspace_paths(run_root, agent_id)
    targets: list[Path] = []
    for key in ("task", "tools"):
        path = paths[key]
        if path.exists():
            targets.append(path)
    targets.extend(root for root in visible_read_roots(run_root, agent_id) if root.exists())
    return targets


def visible_workspace_entries(run_root: str | Path, agent_id: str) -> list[Path]:
    paths = agent_workspace_paths(run_root, agent_id)
    entries = [
        paths["task"],
        paths["tools"],
        paths["context_root"],
        paths["publish_root"],
        paths["scratch_root"],
    ]
    return [path for path in entries if path.exists()]


def is_private_path(run_root: str | Path, agent_id: str, candidate: str | Path) -> bool:
    return _is_within(agent_workspace_paths(run_root, agent_id)["harness_root"], Path(candidate))


def is_artifact_path(run_root: str | Path, agent_id: str, candidate: str | Path) -> bool:
    return _is_within(agent_workspace_paths(run_root, agent_id)["artifacts_root"], Path(candidate))


def is_artifact_file(run_root: str | Path, agent_id: str, candidate: str | Path) -> bool:
    path = Path(candidate).resolve(strict=False)
    return is_artifact_path(run_root, agent_id, path) and path.exists() and path.is_file()


def _is_visible_read_target(run_root: str | Path, agent_id: str, candidate: Path) -> bool:
    paths = agent_workspace_paths(run_root, agent_id)
    if candidate in {paths["task"].resolve(strict=False), paths["tools"].resolve(strict=False)}:
        return True
    return any(_is_within(root, candidate) for root in visible_read_roots(run_root, agent_id))


def resolve_visible_read_file(
    run_root: str | Path,
    agent_id: str,
    raw_path: str,
    *,
    cwd: str | Path | None = None,
) -> Path:
    candidate = _resolve_candidate(run_root, agent_id, raw_path, cwd=cwd)
    if is_private_path(run_root, agent_id, candidate):
        raise VisibilityError("Path is inside the harness-private area.")
    if not candidate.exists() or not candidate.is_file():
        raise VisibilityError(f"File '{raw_path}' is missing or unreadable.")
    if _is_visible_read_target(run_root, agent_id, candidate) or is_artifact_file(run_root, agent_id, candidate):
        return candidate
    raise VisibilityError("Path is outside the agent-visible file surface.")


def resolve_visible_search_target(
    run_root: str | Path,
    agent_id: str,
    raw_path: str,
    *,
    cwd: str | Path | None = None,
) -> Path:
    candidate = _resolve_candidate(run_root, agent_id, raw_path, cwd=cwd)
    if is_private_path(run_root, agent_id, candidate):
        raise VisibilityError("Path is inside the harness-private area.")
    if not candidate.exists():
        raise VisibilityError(f"Search path '{raw_path}' does not exist.")
    if is_artifact_path(run_root, agent_id, candidate):
        if candidate.is_file():
            return candidate
        raise VisibilityError("Artifact directories are not searchable; use an exact artifact file path.")
    if _is_visible_read_target(run_root, agent_id, candidate):
        return candidate
    raise VisibilityError("Path is outside the agent-visible search surface.")


def resolve_visible_write_file(
    run_root: str | Path,
    agent_id: str,
    raw_path: str,
    *,
    cwd: str | Path | None = None,
    must_exist: bool = False,
) -> tuple[Path, str, str]:
    candidate = _resolve_candidate(run_root, agent_id, raw_path, cwd=cwd)
    paths = agent_workspace_paths(run_root, agent_id)
    allowed_roots = {
        "publish": paths["publish_root"].resolve(strict=False),
        "scratch": paths["scratch_root"].resolve(strict=False),
    }
    if is_private_path(run_root, agent_id, candidate):
        raise VisibilityError("Path is inside the harness-private area.")
    for root_name, root in allowed_roots.items():
        if _is_within(root, candidate):
            if candidate == root:
                raise VisibilityError("Path must point to a file inside publish/ or scratch/.")
            if must_exist and not candidate.exists():
                raise VisibilityError(f"File '{raw_path}' does not exist.")
            relative = candidate.relative_to(root).as_posix()
            return candidate, root_name, relative
    raise VisibilityError("Write paths must stay inside this agent's publish/ or scratch/ tree.")

