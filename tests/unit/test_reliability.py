from __future__ import annotations

from pathlib import Path

import pytest

from src.shared import reliability

pytestmark = pytest.mark.unit


def test_cached_retry_call_reuses_cached_value(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reliability, "_CACHE_ROOT", tmp_path)

    calls = {"count": 0}

    def _load() -> dict[str, int]:
        calls["count"] += 1
        return {"value": 7}

    first = reliability.cached_retry_call(
        "sample",
        {"key": "alpha"},
        _load,
        ttl_seconds=60,
        attempts=2,
    )
    second = reliability.cached_retry_call(
        "sample",
        {"key": "alpha"},
        _load,
        ttl_seconds=60,
        attempts=2,
    )

    assert first == {"value": 7}
    assert second == {"value": 7}
    assert calls["count"] == 1


def test_cached_retry_call_retries_transient_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reliability, "_CACHE_ROOT", tmp_path)

    attempts = {"count": 0}

    def _flaky() -> str:
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise ValueError("temporary")
        return "ok"

    out = reliability.cached_retry_call(
        "sample",
        {"key": "beta"},
        _flaky,
        ttl_seconds=60,
        attempts=3,
        retry_exceptions=(ValueError,),
        min_wait_seconds=1,
        max_wait_seconds=1,
    )

    assert out == "ok"
    assert attempts["count"] == 3
