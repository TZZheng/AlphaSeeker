"""
Utilities for generating readable, filesystem-safe report filenames.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Iterable


def extract_prompt_text(messages: Iterable[Any] | None) -> str | None:
    """
    Extract the first non-empty text content from a LangChain-style message list.
    """
    if not messages:
        return None

    for msg in messages:
        content = getattr(msg, "content", None)
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict):
                    text_val = item.get("text")
                    if isinstance(text_val, str):
                        text_parts.append(text_val)
            joined = " ".join(part.strip() for part in text_parts if part.strip())
            if joined:
                return joined
    return None


def build_prompt_report_filename(
    prompt_text: str | None,
    fallback_stem: str = "report",
    max_prompt_chars: int = 96,
    suffix: str | None = None,
) -> str:
    """
    Build a timestamped markdown filename from prompt text.

    Example:
      "analyze_apple_vs_msft_20260315_101500.md"
    """
    stem_source = prompt_text.strip() if prompt_text and prompt_text.strip() else fallback_stem
    stem = re.sub(r"[^A-Za-z0-9]+", "_", stem_source).strip("_")
    if not stem:
        stem = fallback_stem
    stem = stem[:max_prompt_chars].rstrip("_") or fallback_stem
    if suffix:
        safe_suffix = re.sub(r"[^A-Za-z0-9]+", "_", suffix).strip("_")
        if safe_suffix:
            stem = f"{stem}_{safe_suffix}"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stem}_{timestamp}.md"
