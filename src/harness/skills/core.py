"""Core harness skills that are not domain-specific."""

from __future__ import annotations

from typing import Any

from src.harness.skills.common import (
    ensure_str_list,
    json_preview,
    make_result,
    note_evidence,
    safe_read,
    url_evidence,
)
from src.harness.types import HarnessState, SkillResult, SkillSpec
from src.shared.text_utils import condense_context, read_file_safe
from src.shared.web_search import deep_search, search_news, search_web


def search_web_skill(arguments: dict[str, Any], _state: HarnessState) -> SkillResult:
    query = str(arguments.get("query") or "").strip()
    max_results = int(arguments.get("max_results", 5))
    if not query:
        return make_result(
            "search_web",
            arguments,
            status="failed",
            summary="search_web requires a non-empty query.",
            error="Missing query.",
        )

    results = search_web(query, max_results=max_results)
    evidence = [
        url_evidence(
            "search_web",
            item.get("title", query),
            item.get("href", ""),
            content=item.get("body", ""),
            metadata={"query": query},
        )
        for item in results
    ]
    return make_result(
        "search_web",
        arguments,
        status="ok",
        summary=f"Found {len(results)} web results for '{query}'.",
        structured_data={"query": query, "results": results},
        output_text=json_preview(results),
        evidence=evidence,
    )


def search_news_skill(arguments: dict[str, Any], _state: HarnessState) -> SkillResult:
    query = str(arguments.get("query") or "").strip()
    max_results = int(arguments.get("max_results", 5))
    if not query:
        return make_result(
            "search_news",
            arguments,
            status="failed",
            summary="search_news requires a non-empty query.",
            error="Missing query.",
        )

    results = search_news(query, max_results=max_results)
    evidence = [
        url_evidence(
            "search_news",
            item.get("title", query),
            item.get("href", ""),
            content=item.get("body", ""),
            metadata={"query": query, "date": item.get("date", ""), "source": item.get("source", "")},
        )
        for item in results
    ]
    return make_result(
        "search_news",
        arguments,
        status="ok",
        summary=f"Found {len(results)} news results for '{query}'.",
        structured_data={"query": query, "results": results},
        output_text=json_preview(results),
        evidence=evidence,
    )


def search_and_read_skill(arguments: dict[str, Any], _state: HarnessState) -> SkillResult:
    queries = ensure_str_list(arguments.get("queries") or arguments.get("query"))
    if not queries:
        return make_result(
            "search_and_read",
            arguments,
            status="failed",
            summary="search_and_read requires at least one query.",
            error="Missing queries.",
        )

    urls_per_query = int(arguments.get("urls_per_query", 2))
    use_news = bool(arguments.get("use_news", False))
    max_chars_per_url = int(arguments.get("max_chars_per_url", 8000))
    results = deep_search(
        queries=queries,
        urls_per_query=urls_per_query,
        max_chars_per_url=max_chars_per_url,
        use_news=use_news,
    )

    evidence = []
    for item in results:
        content = item.get("full_text") or item.get("snippet") or ""
        evidence.append(
            url_evidence(
                "search_and_read",
                item.get("title", item.get("query", "search result")),
                item.get("url", ""),
                content=content,
                metadata={"query": item.get("query", ""), "use_news": use_news},
            )
        )

    output_chunks = []
    for item in results:
        output_chunks.append(
            f"### {item.get('title', '')}\n"
            f"Query: {item.get('query', '')}\n"
            f"URL: {item.get('url', '')}\n\n"
            f"{item.get('full_text') or item.get('snippet') or ''}"
        )

    return make_result(
        "search_and_read",
        arguments,
        status="ok",
        summary=f"Read {len(results)} search results across {len(queries)} querie(s).",
        structured_data={"queries": queries, "results": results, "use_news": use_news},
        output_text="\n\n".join(output_chunks),
        evidence=evidence,
    )


def condense_context_skill(arguments: dict[str, Any], state: HarnessState) -> SkillResult:
    text = str(arguments.get("text") or "").strip()
    if not text and state.skill_history:
        text = state.skill_history[-1].output_text or ""
    if not text:
        text = "\n".join(state.working_memory[-4:])
    max_chars = int(arguments.get("max_chars", state.request.max_chars_before_condense))
    purpose = str(arguments.get("purpose") or "harness review")
    focus_areas = str(arguments.get("focus_areas") or "")
    condensed = condense_context(
        text=text,
        max_chars=max_chars,
        agent="harness",
        purpose=purpose,
        focus_areas=focus_areas,
    )
    evidence = [
        note_evidence(
            "condense_context",
            f"Condensed text for {purpose}.",
            content=condensed,
            metadata={"input_chars": len(text), "output_chars": len(condensed)},
        )
    ]
    return make_result(
        "condense_context",
        arguments,
        status="ok",
        summary=f"Condensed text from {len(text)} to {len(condensed)} characters.",
        structured_data={"input_chars": len(text), "output_chars": len(condensed)},
        output_text=condensed,
        evidence=evidence,
    )


def read_artifact_skill(arguments: dict[str, Any], _state: HarnessState) -> SkillResult:
    path = str(arguments.get("path") or "").strip()
    max_chars = int(arguments.get("max_chars", 5000))
    if not path:
        return make_result(
            "read_artifact",
            arguments,
            status="failed",
            summary="read_artifact requires a file path.",
            error="Missing path.",
        )

    text = read_file_safe(path, max_chars=max_chars, agent="harness", condense_purpose="artifact review")
    if text == "N/A":
        text = safe_read(path, max_chars=max_chars)
    if not text:
        return make_result(
            "read_artifact",
            arguments,
            status="failed",
            summary=f"Could not read artifact at {path}.",
            error="Artifact missing or unreadable.",
        )

    return make_result(
        "read_artifact",
        arguments,
        status="ok",
        summary=f"Read local artifact {path}.",
        structured_data={"path": path},
        output_text=text,
        artifacts=[path],
        evidence=[note_evidence("read_artifact", f"Artifact contents from {path}.", content=text)],
    )


CORE_SKILLS = [
    SkillSpec(
        name="search_web",
        description="Discover web URLs and snippets for a query.",
        pack="core",
        input_schema={"query": "string", "max_results": "integer"},
        executor=search_web_skill,
    ),
    SkillSpec(
        name="search_news",
        description="Discover recent news URLs and snippets for a query.",
        pack="core",
        input_schema={"query": "string", "max_results": "integer"},
        executor=search_news_skill,
    ),
    SkillSpec(
        name="search_and_read",
        description="Search the web and read extracted full text from the top URLs.",
        pack="core",
        input_schema={
            "queries": "string[]",
            "urls_per_query": "integer",
            "use_news": "boolean",
            "max_chars_per_url": "integer",
        },
        executor=search_and_read_skill,
    ),
    SkillSpec(
        name="condense_context",
        description="Condense long text while preserving names, numbers, and key facts.",
        pack="core",
        input_schema={
            "text": "string",
            "max_chars": "integer",
            "purpose": "string",
            "focus_areas": "string",
        },
        executor=condense_context_skill,
    ),
    SkillSpec(
        name="read_artifact",
        description="Read a saved local file into the harness context.",
        pack="core",
        input_schema={"path": "string", "max_chars": "integer"},
        produces_artifacts=False,
        executor=read_artifact_skill,
    ),
]
