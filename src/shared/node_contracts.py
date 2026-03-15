"""
Strict node result contracts for LangGraph state patches.

LangGraph nodes still have to return dict state patches, but this module makes
every patch carry a typed `NodeResult` so success, partial completion, and
failure are explicit and machine-readable.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping, MutableMapping
from typing import Any, Callable, Literal, TypeVar, cast

from pydantic import BaseModel, Field


class NodeResult(BaseModel):
    """
    Result envelope attached to each node patch.

    `data` is intentionally lightweight. It should summarize what changed
    instead of copying full state payloads into every node result.
    """

    status: Literal["ok", "partial", "failed"] = Field(
        ...,
        description="Whether the node completed fully, partially, or failed.",
    )
    data: dict[str, Any] | None = Field(
        default=None,
        description="Compact structured metadata about the node's output.",
    )
    error: str | None = Field(
        default=None,
        description="Human-readable error for partial/failed outcomes.",
    )


PatchT = TypeVar("PatchT", bound=MutableMapping[str, Any])

_NODE_STATUS_KEY = "__node_status__"
_NODE_ERROR_KEY = "__node_error__"
_NODE_DATA_KEY = "__node_data__"


def partial_patch(
    updates: Mapping[str, Any] | None = None,
    *,
    error: str,
    data: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Annotate a patch as partial without tripping the graph-wide `error` field."""
    patch = dict(updates or {})
    patch[_NODE_STATUS_KEY] = "partial"
    patch[_NODE_ERROR_KEY] = error
    if data is not None:
        patch[_NODE_DATA_KEY] = dict(data)
    return patch


def _build_summary(patch: Mapping[str, Any]) -> dict[str, Any]:
    filtered = {
        key: value
        for key, value in patch.items()
        if key not in {"error", "last_node_result", "node_results"}
        and not key.startswith("__node_")
    }
    artifact_paths = {
        key: value
        for key, value in filtered.items()
        if isinstance(value, str) and value and key.endswith("_path")
    }
    none_keys = sorted(key for key, value in filtered.items() if value is None)
    return {
        "updated_keys": sorted(filtered.keys()),
        "artifact_paths": artifact_paths,
        "none_keys": none_keys,
    }


def _attach_node_result(
    node_name: str,
    patch: MutableMapping[str, Any],
    *,
    status: Literal["ok", "partial", "failed"],
    error: str | None,
    data: Mapping[str, Any] | None = None,
) -> MutableMapping[str, Any]:
    result = NodeResult(
        status=status,
        error=error,
        data=dict(data) if data is not None else _build_summary(patch),
    )
    patch["last_node_result"] = result
    patch["node_results"] = {node_name: result}
    return patch


def contract_node(node_name: str) -> Callable[[Callable[..., PatchT]], Callable[..., PatchT]]:
    """
    Wrap a LangGraph node so every returned state patch carries a `NodeResult`.

    Notes:
    - Upstream graph errors become an explicit `partial` result for skipped nodes.
    - Returning the input `state` object is normalized into an empty patch so the
      wrapper does not re-emit the full state.
    - Nodes can override the inferred status by returning `partial_patch(...)`.
    """

    def decorator(func: Callable[..., PatchT]) -> Callable[..., PatchT]:
        @functools.wraps(func)
        def wrapper(state: Mapping[str, Any], *args: Any, **kwargs: Any) -> PatchT:
            upstream_error = state.get("error")
            if upstream_error:
                skipped = _attach_node_result(
                    node_name,
                    {},
                    status="partial",
                    error=f"Skipped because upstream error is already set: {upstream_error}",
                    data={"updated_keys": []},
                )
                return cast(PatchT, skipped)

            try:
                raw_patch = func(state, *args, **kwargs)
            except Exception as exc:
                failed = _attach_node_result(
                    node_name,
                    {"error": str(exc)},
                    status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
                return cast(PatchT, failed)

            if not isinstance(raw_patch, MutableMapping):
                failed = _attach_node_result(
                    node_name,
                    {"error": f"{node_name} returned invalid patch type"},
                    status="failed",
                    error=f"{node_name} returned {type(raw_patch).__name__}, expected dict-like patch",
                )
                return cast(PatchT, failed)

            patch = {} if raw_patch is state else dict(raw_patch)
            override_status = patch.pop(_NODE_STATUS_KEY, None)
            override_error = patch.pop(_NODE_ERROR_KEY, None)
            override_data = patch.pop(_NODE_DATA_KEY, None)
            explicit_error = override_error or patch.get("error")

            if override_status in {"ok", "partial", "failed"}:
                status = cast(Literal["ok", "partial", "failed"], override_status)
            else:
                status = "failed" if explicit_error else "ok"

            if status == "ok" and explicit_error:
                status = "failed"

            attached = _attach_node_result(
                node_name,
                patch,
                status=status,
                error=str(explicit_error) if explicit_error else None,
                data=cast(Mapping[str, Any] | None, override_data),
            )
            return cast(PatchT, attached)

        return wrapper

    return decorator
