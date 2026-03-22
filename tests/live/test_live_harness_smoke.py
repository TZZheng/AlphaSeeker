from __future__ import annotations

from typing import Any

import pytest

from src.harness.runtime import run_harness
from src.harness.types import ControllerDecision, HarnessRequest
from src.shared.model_config import get_missing_provider_env_vars
from src.shared import web_search
from src.harness.skills import core as core_skills

pytestmark = [pytest.mark.live, pytest.mark.network]


def _assert_live_model_env_ready() -> None:
    missing = get_missing_provider_env_vars()
    if missing:
        formatted = ", ".join(f"{provider}: {env_req}" for provider, env_req in missing.items())
        pytest.fail(
            "Live harness test cannot run because required model-provider keys are missing: "
            f"{formatted}"
        )


def _configure_live_smoke_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    original_deep_search = web_search.deep_search

    def _capped_deep_search(
        queries: list[str],
        urls_per_query: int = 3,
        max_workers: int = 15,
        max_chars_per_url: int = 8000,
        search_delay: float = 0.3,
        use_news: bool = False,
        download_timeout_seconds: int = 12,
        extraction_timeout_seconds: int = 12,
    ) -> list[dict[str, str]]:
        return original_deep_search(
            queries=queries[:1],
            urls_per_query=min(urls_per_query, 1),
            max_workers=min(max_workers, 2),
            max_chars_per_url=min(max_chars_per_url, 2500),
            search_delay=max(search_delay, 0.2),
            use_news=use_news,
            download_timeout_seconds=download_timeout_seconds,
            extraction_timeout_seconds=extraction_timeout_seconds,
        )

    monkeypatch.setattr(web_search, "deep_search", _capped_deep_search)
    monkeypatch.setattr(core_skills, "deep_search", _capped_deep_search)


def test_live_harness_smoke_equity_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    _assert_live_model_env_ready()
    _configure_live_smoke_limits(monkeypatch)

    decisions = iter(
        [
            ControllerDecision(
                action="call_skill",
                rationale="Gather evidence",
                skill_call={"name": "search_and_read", "arguments": {"queries": ["AAPL latest valuation risk"], "urls_per_query": 1}},
            ),
            ControllerDecision(action="draft", rationale="Write draft"),
        ]
    )

    response = run_harness(
        HarnessRequest(
            user_prompt="Analyze AAPL valuation and risk using current evidence.",
            selected_packs=["core", "equity"],
            max_steps=4,
        ),
        controller_fn=lambda _state: next(decisions),
    )

    assert response.status == "completed"
    assert response.final_response.strip()
    assert response.report_path
    assert response.trace_path
