from __future__ import annotations

from pathlib import Path

import pytest

from src.agents.commodity.tools import eia

pytestmark = pytest.mark.unit


def test_fetch_eia_series_redacts_api_key_in_error_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    api_key = "unit-test-eia-key"
    leaked_url = (
        "404 Client Error: Not Found for url: "
        f"https://api.eia.gov/series/?api_key={api_key}&series_id=PET.WCESTUS1.W"
    )

    monkeypatch.setenv("EIA_API_KEY", api_key)
    monkeypatch.setattr(eia.os, "getcwd", lambda: str(tmp_path))

    def _raise_request_error(*_args, **_kwargs):
        raise RuntimeError(leaked_url)

    monkeypatch.setattr(eia, "request_json", _raise_request_error)

    path, metadata = eia.fetch_eia_series(["PET.WCESTUS1.W"])

    content = Path(path).read_text(encoding="utf-8")
    assert metadata == {}
    assert api_key not in content
    assert "api_key=[REDACTED]" in content
