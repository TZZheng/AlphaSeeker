from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from src.agents.commodity.tools import futures

pytestmark = pytest.mark.unit


def test_iter_candidate_contracts_generates_expected_symbols(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FixedDate(dt.date):
        @classmethod
        def today(cls) -> "_FixedDate":
            return cls(2026, 1, 15)

    monkeypatch.setattr(futures.datetime, "date", _FixedDate)

    config = futures.FUTURES_TICKERS["gold"]
    candidates = futures._iter_candidate_contracts(config, months_ahead=12)

    assert candidates
    assert candidates[0]["symbol"] == "GCG26.CMX"
    assert all(c["symbol"].endswith(".CMX") for c in candidates)


def test_curve_structure_classification_from_fixture_payload() -> None:
    fixture = Path("tests/fixtures/futures/curve_samples.json")
    payload = json.loads(fixture.read_text(encoding="utf-8"))

    for scenario in payload["scenarios"]:
        structure = futures.classify_curve_structure(
            spot_price=scenario["spot"],
            forward_price=scenario["forward"],
        )
        assert structure == scenario["expected_structure"]


def test_roll_yield_signs() -> None:
    # backwardation -> positive roll yield for long holders
    positive = futures.calculate_roll_yield(spot_price=75.0, forward_price=70.0, days_between=365)
    # contango -> negative roll yield for long holders
    negative = futures.calculate_roll_yield(spot_price=70.0, forward_price=75.0, days_between=365)

    assert positive > 0
    assert negative < 0


def test_fetch_futures_curve_writes_markdown_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _FixedDate(dt.date):
        @classmethod
        def today(cls) -> "_FixedDate":
            return cls(2026, 1, 15)

    monkeypatch.setattr(futures.datetime, "date", _FixedDate)
    monkeypatch.setattr(futures.os, "getcwd", lambda: str(tmp_path))

    # Deterministic candidate set and prices
    monkeypatch.setattr(
        futures,
        "_iter_candidate_contracts",
        lambda *_args, **_kwargs: [
            {
                "symbol": "CLG26.NYM",
                "contract_month": "2026-02",
                "contract_date": _FixedDate(2026, 2, 1),
            },
            {
                "symbol": "CLJ26.NYM",
                "contract_month": "2026-04",
                "contract_date": _FixedDate(2026, 4, 1),
            },
            {
                "symbol": "CLM26.NYM",
                "contract_month": "2026-06",
                "contract_date": _FixedDate(2026, 6, 1),
            },
        ],
    )

    prices = {
        "CLG26.NYM": 70.0,
        "CLJ26.NYM": 72.0,
        "CLM26.NYM": 74.0,
    }
    monkeypatch.setattr(futures, "_fetch_last_close", lambda symbol: prices.get(symbol))

    path, meta = futures.fetch_futures_curve("crude oil", num_contracts=3)

    assert path is not None
    assert Path(path).exists()
    content = Path(path).read_text(encoding="utf-8")
    assert "# Futures Curve" in content
    assert "**Market Structure:** Contango" in content
    assert meta["structure"] == "contango"
    assert meta["curve_points"] == 3
