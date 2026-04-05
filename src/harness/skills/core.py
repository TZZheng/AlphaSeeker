"""Core harness skills that are not domain-specific."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from src.harness.artifacts import sync_reduction_artifacts
from src.harness.retrieval import (
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
from src.harness.types import HarnessState, SkillMetrics, SkillResult, SkillSpec
from src.shared.text_utils import condense_context
from src.shared.web_search import read_urls_parallel, search_news, search_web


DEFAULT_SEARCH_MAX_RESULTS = 8
DEFAULT_NEWS_MAX_RESULTS = 8
DEFAULT_READ_WEB_MAX_URLS = 6
DEFAULT_MAX_CHARS_PER_URL = 12000
DEFAULT_CONDENSE_MAX_CHARS = 6000
DEFAULT_QUERY_TARGET = 24
DEFAULT_CANDIDATE_TARGET = 120
DEFAULT_READ_QUEUE_TARGET = 40
DEFAULT_INGEST_BATCH_SIZE = 10
DEFAULT_FILE_SEARCH_MAX_RESULTS = 20
RETRIEVAL_STAGES = {
    "plan_queries",
    "discover",
    "rank",
    "build_read_queue",
    "ingest_batch",
    "extract_batch",
    "refresh_coverage",
    "run_wave",
}


def _resolve_search_paths(state: HarnessState, raw_paths: list[str]) -> list[str]:
    resolved: list[str] = []
    for raw_path in raw_paths:
        candidate = Path(raw_path).expanduser()
        if not candidate.is_absolute():
            candidate = Path(state.workspace_path) / candidate
        if candidate.exists():
            resolved.append(str(candidate))
    return resolved


def _python_search_fallback(
    *,
    pattern: str,
    targets: list[str],
    max_results: int,
    fixed_strings: bool,
    ignore_case: bool,
) -> list[dict[str, Any]]:
    needle = pattern if not ignore_case else pattern.lower()
    results: list[dict[str, Any]] = []
    for raw_target in targets:
        target = Path(raw_target)
        candidate_files = [target] if target.is_file() else [path for path in target.rglob("*") if path.is_file()]
        for file_path in candidate_files:
            try:
                for line_number, raw_line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
                    haystack = raw_line if not ignore_case else raw_line.lower()
                    matched = needle in haystack if fixed_strings else __import__("re").search(pattern, raw_line, __import__("re").IGNORECASE if ignore_case else 0)
                    if not matched:
                        continue
                    results.append(
                        {
                            "path": str(file_path),
                            "line_number": line_number,
                            "snippet": raw_line[:240],
                        }
                    )
                    if len(results) >= max_results:
                        return results
            except (OSError, UnicodeDecodeError):
                continue
    return results


def search_in_files_skill(arguments: dict[str, Any], state: HarnessState) -> SkillResult:
    pattern = str(arguments.get("pattern") or arguments.get("query") or "").strip()
    if not pattern:
        return make_result(
            "search_in_files",
            arguments,
            status="failed",
            summary="search_in_files requires a non-empty pattern.",
            error="Missing pattern.",
        )

    max_results = max(1, min(int(arguments.get("max_results", DEFAULT_FILE_SEARCH_MAX_RESULTS)), 100))
    if "fixed_strings" in arguments:
        fixed_strings = bool(arguments.get("fixed_strings"))
    else:
        fixed_strings = not any(char in pattern for char in ".^$*+?{}[]|()\\")
    ignore_case = bool(arguments.get("ignore_case", True))
    requested_paths = ensure_str_list(arguments.get("paths")) or [state.workspace_path]
    resolved_paths = _resolve_search_paths(state, requested_paths)
    if not resolved_paths:
        return make_result(
            "search_in_files",
            arguments,
            status="failed",
            summary="search_in_files could not find any readable target paths.",
            error="No readable files or directories were provided.",
        )

    matches: list[dict[str, Any]] = []
    rg_error = ""
    command = [
        "rg",
        "--json",
        "--line-number",
        "--no-heading",
        "--color",
        "never",
    ]
    if fixed_strings:
        command.append("--fixed-strings")
    if ignore_case:
        command.append("--ignore-case")
    command.extend([pattern, *resolved_paths])

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if completed.returncode not in {0, 1}:
            rg_error = completed.stderr.strip() or f"rg exited with code {completed.returncode}."
        else:
            for raw_line in completed.stdout.splitlines():
                if len(matches) >= max_results:
                    break
                try:
                    payload = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                if payload.get("type") != "match":
                    continue
                data = payload.get("data") or {}
                path_info = data.get("path") or {}
                lines_info = data.get("lines") or {}
                matches.append(
                    {
                        "path": str(path_info.get("text") or ""),
                        "line_number": int(data.get("line_number") or 0),
                        "snippet": str(lines_info.get("text") or "").rstrip("\n")[:240],
                    }
                )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        rg_error = "rg unavailable or timed out."

    if rg_error:
        matches = _python_search_fallback(
            pattern=pattern,
            targets=resolved_paths,
            max_results=max_results,
            fixed_strings=fixed_strings,
            ignore_case=ignore_case,
        )

    output_lines = [
        f"{item['path']}:{item['line_number']}\n{item['snippet']}"
        for item in matches
    ]
    evidence = []
    if matches:
        evidence.append(
            note_evidence(
                "search_in_files",
                f"Found {len(matches)} file match(es) for '{pattern}'.",
                content="\n\n".join(output_lines),
                metadata={
                    "pattern": pattern,
                    "searched_paths": resolved_paths,
                    "used_fallback": bool(rg_error),
                },
            )
        )
    return make_result(
        "search_in_files",
        arguments,
        status="ok",
        summary=(
            f"Found {len(matches)} match(es) for '{pattern}' across {len(resolved_paths)} target path(s)."
        ),
        details={
            "pattern": pattern,
            "matches": matches,
            "searched_paths": resolved_paths,
            "used_fallback": bool(rg_error),
            "fallback_reason": rg_error,
        },
        metrics=SkillMetrics(
            evidence_count=len(evidence),
            artifact_count=0,
            extra={
                "match_count": len(matches),
                "searched_path_count": len(resolved_paths),
            },
        ),
        output_text="\n\n".join(output_lines),
        evidence=evidence,
    )


def get_current_datetime_skill(arguments: dict[str, Any], _state: HarnessState) -> SkillResult:
    requested_timezone = str(arguments.get("timezone") or "").strip()
    try:
        if requested_timezone:
            tzinfo = ZoneInfo(requested_timezone)
            timezone_name = requested_timezone
        else:
            local_now = datetime.now().astimezone()
            tzinfo = local_now.tzinfo or timezone.utc
            timezone_name = getattr(tzinfo, "key", None) or local_now.tzname() or "local"
    except ZoneInfoNotFoundError:
        return make_result(
            "get_current_datetime",
            arguments,
            status="failed",
            summary=f"Unknown timezone '{requested_timezone}'.",
            error="Invalid timezone.",
        )

    local_now = datetime.now(tzinfo)
    utc_now = local_now.astimezone(timezone.utc)
    utc_offset = local_now.utcoffset()
    utc_offset_minutes = int(utc_offset.total_seconds() // 60) if utc_offset is not None else 0
    payload = {
        "local_iso": local_now.isoformat(),
        "local_date": local_now.date().isoformat(),
        "local_time": local_now.time().isoformat(timespec="seconds"),
        "local_day_of_week": local_now.strftime("%A"),
        "timezone": timezone_name,
        "utc_offset_minutes": utc_offset_minutes,
        "utc_iso": utc_now.isoformat(),
        "utc_date": utc_now.date().isoformat(),
        "unix_timestamp": int(local_now.timestamp()),
    }
    output_text = json_preview(payload)
    evidence = [
        note_evidence(
            "get_current_datetime",
            f"Current datetime resolved as {payload['local_iso']} in {timezone_name}.",
            content=output_text,
            metadata={
                "date": payload["local_date"],
                "timezone": timezone_name,
                "utc_iso": payload["utc_iso"],
            },
        )
    ]
    return make_result(
        "get_current_datetime",
        arguments,
        status="ok",
        summary=f"Current datetime is {payload['local_iso']} in {timezone_name}.",
        details=payload,
        metrics=SkillMetrics(
            evidence_count=1,
            dated_evidence_count=1,
            extra={"timezone": timezone_name, "utc_offset_minutes": utc_offset_minutes},
        ),
        output_text=output_text,
        evidence=evidence,
    )


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
            metadata={"query": query, "date": item.get("date", "")},
        )
        for item in results
    ]
    return make_result(
        "search_web",
        arguments,
        status="ok",
        summary=f"Found {len(results)} web results for '{query}'.",
        details={"query": query, "results": results},
        metrics=SkillMetrics(
            evidence_count=len(evidence),
            urls_discovered=len(results),
            dated_evidence_count=sum(1 for item in results if item.get("date")),
        ),
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
        details={"query": query, "results": results},
        metrics=SkillMetrics(
            evidence_count=len(evidence),
            urls_discovered=len(results),
            fresh_evidence_count=len(results),
            dated_evidence_count=sum(1 for item in results if item.get("date")),
        ),
        output_text=json_preview(results),
        evidence=evidence,
    )


def read_web_pages_skill(arguments: dict[str, Any], _state: HarnessState) -> SkillResult:
    urls = [str(item).strip() for item in arguments.get("urls") or [] if str(item).strip()]
    if not urls:
        return make_result(
            "read_web_pages",
            arguments,
            status="failed",
            summary="read_web_pages requires at least one URL.",
            error="Missing urls.",
        )

    max_urls = max(1, min(int(arguments.get("max_urls", DEFAULT_READ_WEB_MAX_URLS)), 20))
    max_chars_per_url = max(500, int(arguments.get("max_chars_per_url", DEFAULT_MAX_CHARS_PER_URL)))
    selected_urls = urls[:max_urls]
    text_by_url = read_urls_parallel(
        selected_urls,
        max_workers=min(6, max(2, len(selected_urls))),
        max_chars_per_url=max_chars_per_url,
    )

    pages: list[dict[str, Any]] = []
    output_chunks: list[str] = []
    evidence = []
    for url in selected_urls:
        text = text_by_url.get(url, "")
        if not text:
            continue
        pages.append(
            {
                "url": url,
                "content": text,
                "chars": len(text),
            }
        )
        output_chunks.append(f"### {url}\n\n{text}")
        evidence.append(
            url_evidence(
                "read_web_pages",
                f"Read web page {url}",
                url,
                content=text,
                metadata={"chars": len(text)},
            )
        )

    return make_result(
        "read_web_pages",
        arguments,
        status="ok",
        summary=f"Read {len(pages)} web page(s) from {len(selected_urls)} requested URL(s).",
        details={
            "requested_urls": selected_urls,
            "pages": pages,
            "max_chars_per_url": max_chars_per_url,
        },
        metrics=SkillMetrics(
            evidence_count=len(evidence),
            urls_read=len(pages),
        ),
        output_text="\n\n".join(output_chunks),
        evidence=evidence,
    )


def search_web_resources_skill(arguments: dict[str, Any], state: HarnessState) -> SkillResult:
    """Run a complete deterministic web-retrieval wave without exposing internal stages."""

    prompt = str(arguments.get("prompt") or arguments.get("query") or state.request.user_prompt).strip()
    if not prompt:
        return make_result(
            "search_web_resources",
            arguments,
            status="failed",
            summary="search_web_resources requires a non-empty prompt.",
            error="Missing prompt.",
        )

    required_sections = ensure_str_list(arguments.get("required_sections")) or list(state.required_sections or [])
    result = retrieve_sources_skill(
        {
            "stage": "run_wave",
            "prompt": prompt,
            "required_sections": required_sections,
        },
        state,
    )
    return result.model_copy(
        update={
            "skill_name": "search_web_resources",
            "arguments": dict(arguments),
            "summary": (
                f"Completed a full web-resource search wave for '{prompt}' with "
                f"{result.metrics.urls_read} successful reads."
                if result.status == "ok"
                else result.summary
            ),
            "metrics": result.metrics.model_copy(
                update={
                    "extra": {
                        **dict(result.metrics.extra),
                        "internal_stage": "run_wave",
                    }
                }
            ),
        }
    )


def condense_context_skill(arguments: dict[str, Any], state: HarnessState) -> SkillResult:
    text = str(arguments.get("text") or "").strip()
    if not text and state.skill_history:
        text = state.skill_history[-1].output_text or ""
    if not text:
        recent = [item.summary for item in state.observations[-4:]]
        text = "\n".join(recent)
    max_chars = int(arguments.get("max_chars", DEFAULT_CONDENSE_MAX_CHARS))
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
        details={"input_chars": len(text), "output_chars": len(condensed), "purpose": purpose},
        metrics=SkillMetrics(evidence_count=1, extra={"input_chars": len(text), "output_chars": len(condensed)}),
        output_text=condensed,
        evidence=evidence,
    )


def read_file_skill(arguments: dict[str, Any], _state: HarnessState) -> SkillResult:
    path = str(arguments.get("path") or "").strip()
    max_chars_raw = arguments.get("max_chars")
    max_chars = int(max_chars_raw) if max_chars_raw is not None else DEFAULT_MAX_CHARS_PER_URL
    start_char = max(0, int(arguments.get("start_char", 0)))
    start_line_raw = arguments.get("start_line")
    max_lines_raw = arguments.get("max_lines")
    if not path:
        return make_result(
            "read_file",
            arguments,
            status="failed",
            summary="read_file requires a file path.",
            error="Missing path.",
        )

    file_path = Path(path).expanduser()
    if not file_path.exists() or not file_path.is_file():
        return make_result(
            "read_file",
            arguments,
            status="failed",
            summary=f"Could not read file at {path}.",
            error="File missing or unreadable.",
        )

    try:
        full_text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        full_text = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return make_result(
            "read_file",
            arguments,
            status="failed",
            summary=f"Could not read file at {path}.",
            error="File unreadable.",
        )

    if start_line_raw is not None or max_lines_raw is not None:
        lines = full_text.splitlines(keepends=True)
        start_line = max(1, int(start_line_raw or 1))
        start_index = start_line - 1
        max_lines = int(max_lines_raw) if max_lines_raw is not None else len(lines)
        if max_lines <= 0:
            end_index = len(lines)
        else:
            end_index = min(len(lines), start_index + max_lines)
        text = "".join(lines[start_index:end_index])
        truncated = end_index < len(lines)
        return make_result(
            "read_file",
            arguments,
            status="truncated" if truncated else "ok",
            summary=(
                f"Read {end_index - start_index} line(s) from {path} starting at line {start_line}."
                + (" Content was truncated." if truncated else "")
            ),
            details={
                "path": path,
                "start_line": start_line,
                "returned_lines": max(0, end_index - start_index),
                "total_lines": len(lines),
            },
            metrics=SkillMetrics(evidence_count=1, artifact_count=1),
            output_text=text,
            artifacts=[path],
            evidence=[note_evidence("read_file", f"File contents from {path}.", content=text)],
        )

    if max_chars <= 0:
        end_char = len(full_text)
    else:
        end_char = min(len(full_text), start_char + max_chars)
    text = full_text[start_char:end_char]
    truncated = end_char < len(full_text)

    return make_result(
        "read_file",
        arguments,
        status="truncated" if truncated else "ok",
        summary=(
            f"Read {len(text)} character(s) from {path} starting at offset {start_char}."
            + (" Content was truncated." if truncated else "")
        ),
        details={
            "path": path,
            "start_char": start_char,
            "returned_chars": len(text),
            "total_chars": len(full_text),
        },
        metrics=SkillMetrics(evidence_count=1, artifact_count=1),
        output_text=text,
        artifacts=[path],
        evidence=[note_evidence("read_file", f"File contents from {path}.", content=text)],
    )


def retrieve_sources_skill(arguments: dict[str, Any], state: HarnessState) -> SkillResult:
    """Build, ingest, and reduce a retrieval corpus for the controller."""

    stage = str(arguments.get("stage") or "run_wave").strip()
    if stage not in RETRIEVAL_STAGES:
        return make_result(
            "retrieve_sources",
            arguments,
            status="failed",
            summary="retrieve_sources received an illegal stage.",
            error=f"Illegal stage '{stage}'.",
        )

    prompt = str(arguments.get("prompt") or state.request.user_prompt).strip()
    query_target = int(arguments.get("query_target", DEFAULT_QUERY_TARGET))
    candidate_target = int(arguments.get("candidate_target", DEFAULT_CANDIDATE_TARGET))
    read_queue_target = int(arguments.get("read_queue_target", DEFAULT_READ_QUEUE_TARGET))
    batch_size = int(arguments.get("ingest_batch_size", DEFAULT_INGEST_BATCH_SIZE))
    max_chars_per_url = int(arguments.get("max_chars_per_url", DEFAULT_MAX_CHARS_PER_URL))
    required_sections = ensure_str_list(arguments.get("required_sections")) or list(state.required_sections or [])
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

    if stage in {"plan_queries", "discover", "run_wave"} and not state.query_buckets:
        state.query_buckets = build_query_buckets(
            prompt,
            [pack for pack in state.enabled_packs if pack != "core"],
            query_target=query_target,
        )

    if stage == "plan_queries":
        sync_reduction_artifacts(state)
        stage_output = build_stage_output(state, stage, artifact_paths)
        return make_result(
            "retrieve_sources",
            arguments,
            status="ok",
            summary=f"Planned {stage_output.query_count} retrieval querie(s).",
            details=stage_output.model_dump(mode="json"),
            metrics=SkillMetrics(
                artifact_count=len(artifact_paths),
                urls_discovered=stage_output.discovered_count,
                urls_read=stage_output.successful_read_count,
                extra={"coverage_status": stage_output.coverage_status},
            ),
            artifacts=artifact_paths,
        )

    if stage in {"discover", "run_wave"} and not state.discovered_sources:
        candidates = discover_sources(
            prompt,
            state.query_buckets,
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
            state.retrieval_wave_count += 1

    source_evidence = []
    if stage in {"extract_batch", "run_wave"} and state.read_results:
        new_cards = extract_source_cards(
            state.read_results,
            state.discovered_sources,
            required_sections,
            existing_cards=state.source_cards,
        )
        if new_cards:
            next_evidence_index = len(state.evidence_ledger) + 2
            for offset, card in enumerate(new_cards):
                evidence_item = url_evidence(
                    "retrieve_sources",
                    card.title or card.summary[:120],
                    card.canonical_url,
                    content="\n".join(card.extracted_facts[:6]) or card.summary,
                    metadata={
                        "source_id": card.source_id,
                        "date": card.publication_date,
                        "domain": card.domain,
                    },
                )
                evidence_item.id = f"E{next_evidence_index + offset}"
                card.evidence_ids = [evidence_item.id]
                source_evidence.append(evidence_item)
            state.source_cards.extend(new_cards)

    if stage in {"extract_batch", "refresh_coverage", "run_wave"}:
        refresh_reduction_state(state)
    else:
        sync_reduction_artifacts(state)

    stage_output = build_stage_output(state, stage, artifact_paths)
    summary_evidence = note_evidence(
        "retrieve_sources",
        (
            f"Retrieval stage '{stage}' now has {stage_output.discovered_count} discovered candidates, "
            f"{stage_output.read_queue_count} queued reads, and {stage_output.source_card_count} source cards."
        ),
        metadata={
            "stage": stage,
            "coverage_status": stage_output.coverage_status,
            "successful_read_count": stage_output.successful_read_count,
        },
    )
    summary_evidence.id = f"E{len(state.evidence_ledger) + 1}"
    evidence = [summary_evidence, *source_evidence]
    return make_result(
        "retrieve_sources",
        arguments,
        status="ok",
        summary=(
            f"Retrieval stage '{stage}' completed with {stage_output.discovered_count} candidates and "
            f"{stage_output.successful_read_count} successful reads."
        ),
        details=stage_output.model_dump(mode="json"),
        metrics=SkillMetrics(
            evidence_count=len(evidence),
            artifact_count=len(artifact_paths),
            urls_discovered=stage_output.discovered_count,
            urls_read=stage_output.successful_read_count,
            fresh_evidence_count=sum(1 for item in source_evidence if item.metadata.get("date")),
            dated_evidence_count=sum(1 for item in source_evidence if item.metadata.get("date")),
            sections_touched=required_sections,
            extra={
                "stage": stage,
                "coverage_status": stage_output.coverage_status,
                "source_card_count": stage_output.source_card_count,
            },
        ),
        artifacts=artifact_paths,
        evidence=evidence,
    )


CORE_SKILLS = [
    SkillSpec(
        name="search_in_files",
        description="Search local files or directories for a keyword or exact text match, returning file paths, line numbers, and snippets.",
        pack="core",
        input_schema={
            "pattern": "string",
            "paths": "string[]",
            "max_results": "integer",
            "fixed_strings": "boolean",
            "ignore_case": "boolean",
        },
        executor=search_in_files_skill,
    ),
    SkillSpec(
        name="get_current_datetime",
        description="Return the current local and UTC datetime so you can anchor time-sensitive work to an explicit date.",
        pack="core",
        input_schema={"timezone": "string"},
        executor=get_current_datetime_skill,
    ),
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
        name="read_web_pages",
        description="Read extracted text from specific web URLs returned by prior search results.",
        pack="core",
        input_schema={
            "urls": "string[]",
            "max_urls": "integer",
            "max_chars_per_url": "integer",
        },
        executor=read_web_pages_skill,
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
        name="read_file",
        description="Read an exact local file path and return its content directly without hidden summarization.",
        pack="core",
        input_schema={
            "path": "string",
            "max_chars": "integer",
            "start_char": "integer",
            "start_line": "integer",
            "max_lines": "integer",
        },
        produces_artifacts=False,
        executor=read_file_skill,
    ),
    SkillSpec(
        name="retrieve_sources",
        description="Build, rank, ingest, and reduce a retrieval corpus.",
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
        executor=retrieve_sources_skill,
    ),
]
