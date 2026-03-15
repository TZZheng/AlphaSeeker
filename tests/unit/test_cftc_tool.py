from __future__ import annotations

import csv
import datetime as dt
import io
import zipfile
from pathlib import Path

import pytest

from src.agents.commodity.tools import cftc

pytestmark = pytest.mark.unit


class _DummyResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None


def _build_zip_payload_from_csv(csv_path: Path) -> bytes:
    csv_text = csv_path.read_text(encoding="utf-8")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("deacot2026.txt", csv_text)
    return buffer.getvalue()


def _load_fixture_rows(csv_path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(csv_path.read_text(encoding="utf-8").splitlines()))


def test_download_year_rows_parses_zip_fixture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fixture_csv = Path("tests/fixtures/cftc/cot_sample.csv")
    payload = _build_zip_payload_from_csv(fixture_csv)

    monkeypatch.setattr(
        cftc.requests,
        "get",
        lambda _url, **_kwargs: _DummyResponse(payload),
    )

    rows = cftc._download_year_rows(2026)

    assert len(rows) == 4
    assert rows[0]["CFTC Contract Market Code"] == "067651"


def test_load_recent_positions_sorts_and_deduplicates_dates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_csv = Path("tests/fixtures/cftc/cot_sample.csv")
    fixture_rows = _load_fixture_rows(fixture_csv)

    monkeypatch.setattr(cftc, "_download_year_rows", lambda _year: fixture_rows)

    out = cftc._load_recent_positions(contract_code="067651", weeks=12)

    assert len(out) == 2
    assert out[0]["date"] == dt.date(2026, 3, 3)
    assert out[1]["date"] == dt.date(2026, 2, 24)
    assert out[0]["spec_net"] == 15000


def test_fetch_cot_report_writes_markdown_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rows = [
        {
            "date": dt.date(2026, 3, 3),
            "market_name": "CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE",
            "open_interest": 100000,
            "noncomm_long": 60000,
            "noncomm_short": 45000,
            "comm_long": 21000,
            "comm_short": 25000,
            "spec_net": 15000,
            "commercial_net": -4000,
        },
        {
            "date": dt.date(2026, 2, 24),
            "market_name": "CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE",
            "open_interest": 95000,
            "noncomm_long": 55000,
            "noncomm_short": 46000,
            "comm_long": 22000,
            "comm_short": 24000,
            "spec_net": 9000,
            "commercial_net": -2000,
        },
    ]

    monkeypatch.setattr(cftc, "_load_recent_positions", lambda *_args, **_kwargs: rows)
    monkeypatch.setattr(cftc.os, "getcwd", lambda: str(tmp_path))

    path, meta = cftc.fetch_cot_report("crude oil", num_weeks=2)

    assert path is not None
    assert Path(path).exists()
    content = Path(path).read_text(encoding="utf-8")
    assert "# COT Positioning" in content
    assert "Speculative Net" in content
    assert meta["contract_code"] == "067651"
    assert meta["spec_net_long"] == 15000


def test_to_int_parsing() -> None:
    assert cftc._to_int("12,345") == 12345
    assert cftc._to_int(42) == 42
    assert cftc._to_int(None) == 0
