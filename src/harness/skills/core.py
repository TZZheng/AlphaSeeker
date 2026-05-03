"""Core harness skills that are not domain-specific."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from src.harness.artifacts import agent_workspace_paths, write_json_atomic
from src.harness.skills.common import (
    ensure_str_list,
    json_preview,
    make_result,
    note_evidence,
    safe_read,
    url_evidence,
)
from src.harness.types import HarnessState, SkillMetrics, SkillResult, SkillSpec
from src.harness.visibility import (
    VisibilityError,
    default_search_targets,
    resolve_visible_read_file,
    resolve_visible_search_target,
)
from src.shared.text_utils import condense_context
from src.shared.web_search import read_urls_parallel, search_news, search_web


DEFAULT_SEARCH_MAX_RESULTS = 8
DEFAULT_READ_WEB_MAX_URLS = 6
DEFAULT_MAX_CHARS_PER_URL = 12000
DEFAULT_CONDENSE_MAX_CHARS = 6000
DEFAULT_FILE_SEARCH_MAX_RESULTS = 20


def _resolve_search_paths(state: HarnessState, raw_paths: list[str]) -> list[str]:
    if state.run_root and state.agent_id:
        return [
            str(resolve_visible_search_target(state.run_root, state.agent_id, raw_path))
            for raw_path in raw_paths
        ]
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


def grep_skill(arguments: dict[str, Any], state: HarnessState) -> SkillResult:
    pattern = str(arguments.get("pattern") or arguments.get("query") or "").strip()
    if not pattern:
        return make_result(
            "grep",
            arguments,
            status="failed",
            summary="grep requires a non-empty pattern.",
            error="Missing pattern.",
        )

    max_results = max(1, min(int(arguments.get("max_results", DEFAULT_FILE_SEARCH_MAX_RESULTS)), 100))
    if "fixed_strings" in arguments:
        fixed_strings = bool(arguments.get("fixed_strings"))
    else:
        fixed_strings = not any(char in pattern for char in ".^$*+?{}[]|()\\")
    ignore_case = bool(arguments.get("ignore_case", True))
    requested_paths = ensure_str_list(arguments.get("paths"))
    try:
        if requested_paths:
            resolved_paths = _resolve_search_paths(state, requested_paths)
        elif state.run_root and state.agent_id:
            resolved_paths = [str(path) for path in default_search_targets(state.run_root, state.agent_id)]
        else:
            resolved_paths = _resolve_search_paths(state, [str(state.workspace_path or "")])
    except VisibilityError as exc:
        return make_result(
            "grep",
            arguments,
            status="failed",
            summary=str(exc),
            error=str(exc),
        )
    if not resolved_paths:
        return make_result(
            "grep",
            arguments,
            status="failed",
            summary="grep could not find any readable target paths.",
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
                "grep",
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
        "grep",
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


def _write_search_results(state: HarnessState, results: list[dict[str, Any]], prefix: str) -> str:
    """Write search/news results to a JSON artifact. Returns the artifact path."""
    import time

    slug = prefix[:40].replace("/", "_").replace(" ", "_")
    filename = f"{slug}_{int(time.time() * 1000)}.json"
    if state.run_root and state.agent_id:
        path = agent_workspace_paths(state.run_root, state.agent_id)["search_artifacts_root"] / filename
    else:
        path = Path(state.workspace_path or ".") / "artifacts" / "search" / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(str(path), results)
    return str(path)


def _search_result_url(item: dict[str, Any]) -> str:
    return str(item.get("href") or item.get("url") or "").strip()


def _compact_search_results(results: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "title": str(item.get("title") or "").strip(),
            "url": _search_result_url(item),
            "date": str(item.get("date") or "").strip(),
        }
        for item in results
    ]


def _markdown_table_cell(value: str) -> str:
    return value.replace("\n", " ").replace("|", "\\|").strip()


def _render_search_results_output(
    *,
    query: str,
    results_path: str,
    results: list[dict[str, str]],
) -> str:
    lines = [
        "# Search Results",
        "",
        f"Query: {query}",
        f"Results artifact: {results_path}",
        "",
    ]
    if not results:
        lines.append("No results found.")
        return "\n".join(lines) + "\n"

    lines.extend(["| # | Title | URL | Date |", "|---:|---|---|---|"])
    for index, item in enumerate(results, start=1):
        title = _markdown_table_cell(item["title"])
        url = _markdown_table_cell(item["url"])
        date = _markdown_table_cell(item["date"])
        lines.append(f"| {index} | {title} | {url} | {date} |")
    return "\n".join(lines) + "\n"


def search_web_skill(arguments: dict[str, Any], state: HarnessState) -> SkillResult:
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
    results_path = _write_search_results(state, results, query)
    compact_results = _compact_search_results(results)
    evidence = [
        url_evidence(
            "search_web",
            item.get("title", query),
            _search_result_url(item),
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
        details={
            "query": query,
            "type": "web",
            "results_path": results_path,
            "count": len(results),
            "results": compact_results,
        },
        metrics=SkillMetrics(
            evidence_count=len(evidence),
            urls_discovered=len(results),
            dated_evidence_count=sum(1 for item in results if item.get("date")),
        ),
        output_text=_render_search_results_output(
            query=query,
            results_path=results_path,
            results=compact_results,
        ),
        artifacts=[results_path],
        evidence=evidence,
    )


def search_news_skill(arguments: dict[str, Any], state: HarnessState) -> SkillResult:
    query = str(arguments.get("query") or "").strip()
    max_results = int(arguments.get("max_results", DEFAULT_SEARCH_MAX_RESULTS))
    if not query:
        return make_result(
            "search_news",
            arguments,
            status="failed",
            summary="search_news requires a non-empty query.",
            error="Missing query.",
        )

    results = search_news(query, max_results=max_results)
    results_path = _write_search_results(state, results, f"news_{query}")
    compact_results = _compact_search_results(results)
    evidence = [
        url_evidence(
            "search_news",
            item.get("title", query),
            _search_result_url(item),
            content=item.get("body", ""),
            metadata={"query": query, "date": item.get("date", ""), "type": "news"},
        )
        for item in results
    ]
    return make_result(
        "search_news",
        arguments,
        status="ok",
        summary=f"Found {len(results)} news results for '{query}'.",
        details={
            "query": query,
            "type": "news",
            "results_path": results_path,
            "count": len(results),
            "results": compact_results,
        },
        metrics=SkillMetrics(
            evidence_count=len(evidence),
            urls_discovered=len(results),
            dated_evidence_count=sum(1 for item in results if item.get("date")),
        ),
        output_text=_render_search_results_output(
            query=query,
            results_path=results_path,
            results=compact_results,
        ),
        artifacts=[results_path],
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

    # Strip raw text from details — content is already in output.md and evidence items.
    pages_for_details = [
        {"url": p["url"], "chars": p["chars"]}
        for p in pages
    ]

    return make_result(
        "read_web_pages",
        arguments,
        status="ok",
        summary=f"Read {len(pages)} web page(s) from {len(selected_urls)} requested URL(s).",
        details={
            "requested_urls": selected_urls,
            "pages": pages_for_details
        },
        metrics=SkillMetrics(
            evidence_count=len(evidence),
            urls_read=len(pages),
        ),
        output_text="\n\n".join(output_chunks),
        evidence=evidence,
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


def read_skill(arguments: dict[str, Any], _state: HarnessState) -> SkillResult:
    path = str(arguments.get("path") or "").strip()
    max_chars_raw = arguments.get("max_chars")
    max_chars = int(max_chars_raw) if max_chars_raw is not None else DEFAULT_MAX_CHARS_PER_URL
    start_char = max(0, int(arguments.get("start_char", 0)))
    start_line_raw = arguments.get("start_line")
    max_lines_raw = arguments.get("max_lines")
    if not path:
        return make_result(
            "read",
            arguments,
            status="failed",
            summary="read requires a file path.",
            error="Missing path.",
        )

    if _state.run_root and _state.agent_id:
        try:
            file_path = resolve_visible_read_file(_state.run_root, _state.agent_id, path)
        except VisibilityError as exc:
            return make_result(
                "read",
                arguments,
                status="failed",
                summary=str(exc),
                error=str(exc),
            )
    else:
        file_path = Path(path).expanduser()
        if not file_path.is_absolute() and _state.workspace_path:
            # Resolve relative paths against the agent workspace so that
            # read("publish/final.md") finds the same file that
            # write("publish/final.md") wrote.
            workspace_candidate = Path(_state.workspace_path) / file_path
            if workspace_candidate.exists():
                file_path = workspace_candidate
    if not file_path.exists() or not file_path.is_file():
        return make_result(
            "read",
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
            "read",
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
            "read",
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
            evidence=[note_evidence("read", f"File contents from {path}.", content=text)],
        )

    if max_chars <= 0:
        end_char = len(full_text)
    else:
        end_char = min(len(full_text), start_char + max_chars)
    text = full_text[start_char:end_char]
    truncated = end_char < len(full_text)

    return make_result(
        "read",
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
        evidence=[note_evidence("read", f"File contents from {path}.", content=text)],
    )


CORE_SKILLS = [
    SkillSpec(
        name="grep",
        description="Search visible local files for exact text or regex-like patterns. Use before reading large local files.",
        pack="core",
        input_schema={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Text or pattern to search for."},
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional visible files or directories. Defaults to the agent's visible workspace.",
                },
                "max_results": {"type": "integer", "default": DEFAULT_FILE_SEARCH_MAX_RESULTS, "minimum": 1},
                "fixed_strings": {"type": "boolean", "description": "Treat pattern as literal text."},
                "ignore_case": {"type": "boolean", "default": True},
            },
            "required": ["pattern"],
        },
        executor=grep_skill,
    ),
    SkillSpec(
        name="get_current_datetime",
        description="Return the current local and UTC datetime so you can anchor time-sensitive work to an explicit date.",
        pack="core",
        input_schema={
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "Optional IANA timezone, for example America/New_York.",
                }
            },
        },
        executor=get_current_datetime_skill,
    ),
    SkillSpec(
        name="search_web",
        description="Discover general web URLs and dates for a query. Use search_news instead for current news coverage.",
        pack="core",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "max_results": {"type": "integer", "default": DEFAULT_SEARCH_MAX_RESULTS, "minimum": 1},
            },
            "required": ["query"],
        },
        executor=search_web_skill,
    ),
    SkillSpec(
        name="search_news",
        description="Discover news URLs and publication dates for current or time-sensitive events.",
        pack="core",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "News search query."},
                "max_results": {"type": "integer", "default": DEFAULT_SEARCH_MAX_RESULTS, "minimum": 1},
            },
            "required": ["query"],
        },
        executor=search_news_skill,
    ),
    SkillSpec(
        name="read_web_pages",
        description="Read extracted text from specific URLs returned by search_web or search_news.",
        pack="core",
        input_schema={
            "type": "object",
            "properties": {
                "urls": {"type": "array", "items": {"type": "string"}, "description": "URLs to read."},
                "max_urls": {"type": "integer", "default": DEFAULT_READ_WEB_MAX_URLS, "minimum": 1},
                "max_chars_per_url": {
                    "type": "integer",
                    "default": DEFAULT_MAX_CHARS_PER_URL,
                    "minimum": 500,
                },
            },
            "required": ["urls"],
        },
        executor=read_web_pages_skill,
    ),
    SkillSpec(
        name="condense_context",
        description="Condense long text while preserving names, numbers, and key facts.",
        pack="core",
        input_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Text to condense. Defaults to recent skill output."},
                "max_chars": {"type": "integer", "default": DEFAULT_CONDENSE_MAX_CHARS, "minimum": 200},
                "purpose": {"type": "string", "default": "harness review"},
                "focus_areas": {"type": "string", "description": "Optional facts or sections to preserve."},
            },
        },
        executor=condense_context_skill,
    ),
    SkillSpec(
        name="read",
        description="Read an exact local file path and return its content directly without hidden summarization.",
        pack="core",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Exact visible file path."},
                "max_chars": {"type": "integer", "default": DEFAULT_MAX_CHARS_PER_URL},
                "start_char": {"type": "integer", "default": 0, "minimum": 0},
                "start_line": {"type": "integer", "minimum": 1},
                "max_lines": {"type": "integer", "description": "Line count to read from start_line."},
            },
            "required": ["path"],
        },
        produces_artifacts=False,
        executor=read_skill,
    ),
]
