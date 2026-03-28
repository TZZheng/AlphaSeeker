from __future__ import annotations

import pytest

from src.harness.skills import core as core_skills
from src.harness.types import EvidenceItem, HarnessRequest, HarnessState, ResearchPlan
from src.harness.writer import write_draft

pytestmark = pytest.mark.unit


def test_core_search_defaults_use_wider_fan_out(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, int] = {}

    def _fake_search_web(query: str, max_results: int = 0):
        captured["search_web"] = max_results
        return []

    def _fake_search_news(query: str, max_results: int = 0):
        captured["search_news"] = max_results
        return []

    def _fake_deep_search(
        queries: list[str],
        urls_per_query: int = 0,
        max_chars_per_url: int = 0,
        use_news: bool = False,
    ):
        captured["urls_per_query"] = urls_per_query
        captured["max_chars_per_url"] = max_chars_per_url
        return []

    monkeypatch.setattr(core_skills, "search_web", _fake_search_web)
    monkeypatch.setattr(core_skills, "search_news", _fake_search_news)
    monkeypatch.setattr(core_skills, "deep_search", _fake_deep_search)

    state = HarnessState(request=HarnessRequest(user_prompt="Test"))
    core_skills.search_web_skill({"query": "macro"}, state)
    core_skills.search_news_skill({"query": "macro"}, state)
    core_skills.search_and_read_skill({"queries": ["macro"]}, state)

    assert captured["search_web"] == core_skills.DEFAULT_SEARCH_MAX_RESULTS
    assert captured["search_news"] == core_skills.DEFAULT_NEWS_MAX_RESULTS
    assert captured["urls_per_query"] == core_skills.DEFAULT_URLS_PER_QUERY
    assert captured["max_chars_per_url"] == core_skills.DEFAULT_MAX_CHARS_PER_URL


def test_writer_adds_title_and_normalizes_bold_section_headings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeResponse:
        content = "**Executive Summary**\nGrounded summary [E1].\n\n**Sources**\n- [E1]"

    class _FakeLLM:
        def invoke(self, _messages):
            return _FakeResponse()

    import src.harness.writer as writer_module

    monkeypatch.setattr(writer_module, "get_llm", lambda _model: _FakeLLM())
    state = HarnessState(
        request=HarnessRequest(user_prompt="US macro outlook for the next 12 months."),
        research_plan=ResearchPlan(
            primary_question="US macro outlook for the next 12 months.",
            domain_packs=[],
            required_sections=["Executive Summary", "Sources"],
        ),
        evidence_ledger=[
            EvidenceItem(
                id="E1",
                skill_name="search_and_read",
                source_type="url",
                summary="Macro evidence",
            )
        ],
    )

    draft = write_draft(state)

    assert draft.startswith("# US macro outlook for the next 12 months")
    assert "## Executive Summary" in draft
    assert "## Sources" in draft
