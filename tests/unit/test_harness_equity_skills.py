from __future__ import annotations

import pytest

from src.harness.skills import equity as equity_skills
from src.harness.types import EvidenceItem, HarnessRequest, HarnessState


def test_fetch_company_profile_requires_ticker() -> None:
    state = HarnessState(request=HarnessRequest(user_prompt="Analyze AAPL"))

    result = equity_skills.fetch_company_profile_skill({}, state)

    assert result.status == "failed"
    assert "requires a ticker" in result.summary


def test_search_sec_filings_uses_profile_evidence_company_name(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _fake_search_and_read_filings(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(equity_skills, "search_and_read_filings", _fake_search_and_read_filings)

    state = HarnessState(
        request=HarnessRequest(user_prompt="Analyze AAPL"),
        evidence_ledger=[
            EvidenceItem(
                skill_name="fetch_company_profile",
                source_type="note",
                summary="Apple profile",
                metadata={"ticker": "AAPL", "company_name": "Apple Inc."},
            )
        ],
    )

    result = equity_skills.search_sec_filings_skill({"ticker": "AAPL"}, state)

    assert result.status == "ok"
    assert captured["company_name"] == "Apple Inc."
    assert captured["ticker"] == "AAPL"
