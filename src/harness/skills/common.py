"""Shared helpers for harness skill adapters."""

from __future__ import annotations

import json
import os
from typing import Any

from src.harness.types import EvidenceItem, SkillResult


def ensure_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else []


def json_preview(data: Any) -> str:
    try:
        return json.dumps(data, ensure_ascii=True, indent=2, default=str)
    except Exception:
        return str(data)


def artifact_evidence(
    skill_name: str,
    summary: str,
    path: str,
    *,
    content: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> EvidenceItem:
    return EvidenceItem(
        skill_name=skill_name,
        source_type="artifact",
        summary=summary,
        content=content,
        artifact_paths=[path] if path else [],
        metadata=metadata or {},
    )


def url_evidence(
    skill_name: str,
    summary: str,
    url: str,
    *,
    content: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> EvidenceItem:
    return EvidenceItem(
        skill_name=skill_name,
        source_type="url",
        summary=summary,
        content=content,
        sources=[url] if url else [],
        metadata=metadata or {},
    )


def note_evidence(
    skill_name: str,
    summary: str,
    *,
    content: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> EvidenceItem:
    return EvidenceItem(
        skill_name=skill_name,
        source_type="note",
        summary=summary,
        content=content,
        metadata=metadata or {},
    )


def make_result(
    skill_name: str,
    arguments: dict[str, Any],
    *,
    status: str,
    summary: str,
    structured_data: dict[str, Any] | None = None,
    output_text: str | None = None,
    artifacts: list[str] | None = None,
    evidence: list[EvidenceItem] | None = None,
    error: str | None = None,
) -> SkillResult:
    return SkillResult(
        skill_name=skill_name,
        arguments=arguments,
        status=status,
        summary=summary,
        structured_data=structured_data or {},
        output_text=output_text,
        artifacts=artifacts or [],
        evidence=evidence or [],
        error=error,
    )


def safe_read(path: str, max_chars: int = 5000) -> str:
    if not path or not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    if len(text) > max_chars:
        return text[:max_chars] + f"\n... [truncated at {max_chars} chars]"
    return text
