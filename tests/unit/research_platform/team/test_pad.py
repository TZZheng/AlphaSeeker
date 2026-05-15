from __future__ import annotations

from datetime import datetime, timezone

from src.research_platform.team.pad import Pad


def _clock():
    return datetime(2026, 5, 15, 15, 0, 0, tzinfo=timezone.utc)


def test_append_creates_file_with_header_and_preserves_prior_content(tmp_path):
    pad = Pad(tmp_path / "pad.md", clock=_clock)
    pad.append("first", "hello")
    pad.append("second", "world")
    text = pad.read()
    assert "## first — 2026-05-15T15:00:00Z" in text
    assert "hello" in text and "world" in text
    assert pad.meta().last_appended_at == "2026-05-15T15:00:00Z"


def test_summary_noop_under_cap(tmp_path):
    pad = Pad(tmp_path / "pad.md", max_chars=100, clock=_clock)
    pad.overwrite("short")
    assert pad.needs_summary() is False
    assert pad.summarize_if_oversize() is False


def test_summarize_calls_fn_archives_and_updates_meta(tmp_path):
    pad = Pad(tmp_path / "pad.md", max_chars=5, clock=_clock)
    pad.overwrite("abcdefghij")
    assert pad.summarize_if_oversize(lambda text: "summary:" + text[:3]) is True
    assert "summary:abc" in pad.read()
    assert "abcdefghij" in (tmp_path / "pad.archive.md").read_text()
    assert pad.meta().summarizations == 1


def test_summarize_fallback_when_no_fn(tmp_path):
    pad = Pad(tmp_path / "pad.md", max_chars=5, summary_target_chars=8, clock=_clock)
    pad.overwrite("abcdefghijklmnopqrstuvwxyz")
    assert pad.summarize_if_oversize() is True
    assert "pad truncated" in pad.read()
