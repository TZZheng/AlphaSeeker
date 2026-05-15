from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from src.research_platform.team.ids import now_iso
from src.research_platform.team.mailbox import _atomic_write_text

SummaryFn = Callable[[str], str]


@dataclass(frozen=True)
class PadMeta:
    created_at: str
    summarizations: int
    bytes: int
    last_appended_at: str | None


class Pad:
    def __init__(
        self,
        path: Path,
        *,
        max_chars: int = 32_768,
        summary_target_chars: int = 8_192,
        summary_fn: SummaryFn | None = None,
        clock=None,
    ) -> None:
        self.path = Path(path)
        self.max_chars = max_chars
        self.summary_target_chars = summary_target_chars
        self.summary_fn = summary_fn
        self.clock = clock
        self.archive_path = self.path.with_name("pad.archive.md")
        self.meta_path = self.path.with_name("pad.meta.json")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.meta_path.exists():
            self._write_meta(PadMeta(created_at=now_iso(clock), summarizations=0, bytes=self.size(), last_appended_at=None))

    def read(self) -> str:
        if not self.path.exists():
            return ""
        return self.path.read_text(encoding="utf-8")

    def size(self) -> int:
        return len(self.read())

    def needs_summary(self) -> bool:
        return self.size() > self.max_chars

    def append(self, title: str, body: str) -> None:
        ts = now_iso(self.clock)
        current = self.read()
        addition = f"\n## {title} — {ts}\n\n{body.strip()}\n"
        _atomic_write_text(self.path, current + addition)
        meta = self.meta()
        self._write_meta(PadMeta(meta.created_at, meta.summarizations, self.size(), ts))

    def overwrite(self, content: str) -> None:
        _atomic_write_text(self.path, content)
        meta = self.meta()
        self._write_meta(PadMeta(meta.created_at, meta.summarizations, self.size(), meta.last_appended_at))

    def summarize_if_oversize(self, summary_fn: SummaryFn | None = None) -> bool:
        current = self.read()
        if len(current) <= self.max_chars:
            return False
        ts = now_iso(self.clock)
        self.archive_path.parent.mkdir(parents=True, exist_ok=True)
        with self.archive_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n\n# Archived pad before summary — {ts}\n\n")
            handle.write(current)
        fn = summary_fn or self.summary_fn
        if fn is not None:
            summary = fn(current)
        else:
            head_len = max(1, self.summary_target_chars // 4)
            tail_len = max(1, self.summary_target_chars // 2)
            summary = current[:head_len] + "\n\n[... pad truncated by deterministic fallback ...]\n\n" + current[-tail_len:]
        new_content = f"## Summary of prior pad — {ts}\n\n{summary.strip()}\n"
        _atomic_write_text(self.path, new_content)
        meta = self.meta()
        self._write_meta(PadMeta(meta.created_at, meta.summarizations + 1, self.size(), meta.last_appended_at))
        return True

    def meta(self) -> PadMeta:
        if not self.meta_path.exists():
            return PadMeta(created_at=now_iso(self.clock), summarizations=0, bytes=self.size(), last_appended_at=None)
        data = json.loads(self.meta_path.read_text(encoding="utf-8"))
        return PadMeta(
            created_at=data["created_at"],
            summarizations=int(data.get("summarizations", 0)),
            bytes=int(data.get("bytes", self.size())),
            last_appended_at=data.get("last_appended_at"),
        )

    def _write_meta(self, meta: PadMeta) -> None:
        _atomic_write_text(
            self.meta_path,
            json.dumps(
                {
                    "created_at": meta.created_at,
                    "summarizations": meta.summarizations,
                    "bytes": meta.bytes,
                    "last_appended_at": meta.last_appended_at,
                },
                indent=2,
                sort_keys=True,
            ),
        )
