"""Core harness skills that are not domain-specific."""

from __future__ import annotations

from typing import Any

from src.harness.artifacts import sync_reduction_artifacts
from src.harness.deep_research import (
    build_query_buckets,
    build_read_queue,
    build_stage_output,
    discover_sources,
    extract_source_cards,
    ingest_read_queue,
    rank_discovered_sources,
    refresh_reduction_state,
)
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


DEFAULT_SEARCH_MAX_RESULTS = 8
DEFAULT_NEWS_MAX_RESULTS = 8
DEFAULT_URLS_PER_QUERY = 4
DEFAULT_MAX_CHARS_PER_URL = 12000
DEEP_RETRIEVAL_STAGES = {
    "plan_queries",
    "discover",
    "rank",
    "build_read_queue",
    "ingest_batch",
    "extract_batch",
    "refresh_coverage",
    "run_wave",
}


def search_web_skill(arguments: dict[str, Any], _state: HarnessState) -> SkillResult:
    query = str(arguments.get("query") or "").strip()
    max_results = int(arguments.get("max_results", DEFAULT_SEARCH_MAX_RESULTS))
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
    max_results = int(arguments.get("max_results", DEFAULT_NEWS_MAX_RESULTS))
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

    urls_per_query = int(arguments.get("urls_per_query", DEFAULT_URLS_PER_QUERY))
    use_news = bool(arguments.get("use_news", False))
    max_chars_per_url = int(arguments.get("max_chars_per_url", DEFAULT_MAX_CHARS_PER_URL))
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


def deep_retrieval_skill(arguments: dict[str, Any], state: HarnessState) -> SkillResult:
    """Composite retrieval skill for the opt-in deep research profile.

    A "composite skill" in backend terms is one higher-level entry point that
    coordinates several smaller operations behind a typed interface.
    """

    stage = str(arguments.get("stage") or "run_wave").strip()
    if stage not in DEEP_RETRIEVAL_STAGES:
        return make_result(
            "deep_retrieval",
            arguments,
            status="failed",
            summary="deep_retrieval received an illegal stage.",
            error=f"Illegal stage '{stage}'.",
        )

    prompt = str(arguments.get("prompt") or state.request.user_prompt).strip()
    query_target = int(arguments.get("query_target", state.request.deep_query_target))
    candidate_target = int(arguments.get("candidate_target", state.request.deep_candidate_target))
    read_queue_target = int(arguments.get("read_queue_target", state.request.deep_read_queue_target))
    batch_size = int(arguments.get("ingest_batch_size", state.request.deep_read_batch_size))
    max_chars_per_url = int(arguments.get("max_chars_per_url", DEFAULT_MAX_CHARS_PER_URL))
    artifact_paths = [
        state.dossier_paths.get("discovered_sources", ""),
        state.dossier_paths.get("read_queue", ""),
        state.dossier_paths.get("read_results", ""),
        state.dossier_paths.get("source_cards", ""),
        state.dossier_paths.get("fact_index", ""),
        state.dossier_paths.get("section_briefs", ""),
        state.dossier_paths.get("coverage_matrix", ""),
    ]
    artifact_paths = [path for path in artifact_paths if path]

    if stage in {"plan_queries", "discover", "run_wave"} and not state.deep_query_buckets:
        state.deep_query_buckets = build_query_buckets(
            prompt,
            [pack for pack in state.enabled_packs if pack != "core"],
            query_target=query_target,
        )

    if stage == "plan_queries":
        sync_reduction_artifacts(state)
        stage_output = build_stage_output(state, stage, artifact_paths)
        return make_result(
            "deep_retrieval",
            arguments,
            status="ok",
            summary=f"Planned {stage_output.query_count} deep retrieval querie(s).",
            structured_data=stage_output.model_dump(mode="json"),
            artifacts=artifact_paths,
        )

    if stage in {"discover", "run_wave"} and not state.discovered_sources:
        candidates = discover_sources(
            prompt,
            state.deep_query_buckets,
            candidate_target=candidate_target,
        )
        state.discovered_sources = rank_discovered_sources(candidates, prompt)

    if stage in {"rank", "run_wave"} and state.discovered_sources:
        state.discovered_sources = rank_discovered_sources(state.discovered_sources, prompt)

    if stage in {"build_read_queue", "run_wave"} and not state.read_queue:
        state.read_queue = build_read_queue(state.discovered_sources, queue_target=read_queue_target)

    if stage in {"ingest_batch", "run_wave"} and state.read_queue:
        new_results = ingest_read_queue(
            state.read_queue,
            state.discovered_sources,
            state.read_results,
            batch_size=batch_size,
            max_chars_per_url=max_chars_per_url,
        )
        if new_results:
            state.read_results.extend(new_results)
            state.deep_retrieval_wave_count += 1

    if stage in {"extract_batch", "run_wave"} and state.read_results:
        new_cards = extract_source_cards(
            state.read_results,
            state.discovered_sources,
            state.research_plan.required_sections if state.research_plan else [],
            existing_cards=state.source_cards,
        )
        if new_cards:
            state.source_cards.extend(new_cards)

    if stage in {"extract_batch", "refresh_coverage", "run_wave"}:
        refresh_reduction_state(state)
    else:
        sync_reduction_artifacts(state)

    stage_output = build_stage_output(state, stage, artifact_paths)
    evidence = [
        note_evidence(
            "deep_retrieval",
            (
                f"Deep retrieval stage '{stage}' now has "
                f"{stage_output.discovered_count} discovered candidates, "
                f"{stage_output.read_queue_count} queued reads, and "
                f"{stage_output.source_card_count} source cards."
            ),
            metadata={
                "stage": stage,
                "coverage_status": stage_output.coverage_status,
                "successful_read_count": stage_output.successful_read_count,
            },
        )
    ]
    return make_result(
        "deep_retrieval",
        arguments,
        status="ok",
        summary=(
            f"Deep retrieval stage '{stage}' completed with "
            f"{stage_output.discovered_count} candidates and "
            f"{stage_output.successful_read_count} successful reads."
        ),
        structured_data=stage_output.model_dump(mode="json"),
        artifacts=artifact_paths,
        evidence=evidence,
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
    SkillSpec(
        name="deep_retrieval",
        description="Build, rank, ingest, and reduce a staged deep-research corpus.",
        pack="core",
        input_schema={
            "stage": "string",
            "prompt": "string",
            "query_target": "integer",
            "candidate_target": "integer",
            "read_queue_target": "integer",
            "ingest_batch_size": "integer",
            "max_chars_per_url": "integer",
        },
        produces_artifacts=True,
        executor=deep_retrieval_skill,
    ),
]
