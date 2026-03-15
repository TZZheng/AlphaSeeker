from __future__ import annotations

import os
import re
import time
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import Any

import pytest

from src.agents.equity import nodes as equity_nodes
from src.agents.macro import nodes as macro_nodes
from src.shared.model_config import get_missing_provider_env_vars
from src.shared import web_search
from src.supervisor.graph import app as supervisor_app

pytestmark = [pytest.mark.live, pytest.mark.network]


def _assert_live_model_env_ready() -> None:
    missing = get_missing_provider_env_vars()
    if missing:
        formatted = ", ".join(f"{provider}: {env_req}" for provider, env_req in missing.items())
        pytest.fail(
            "Live pipeline test cannot run because required model-provider keys are missing: "
            f"{formatted}"
        )


def _assert_agent_result_contract(agent_results: dict[str, str], required_agents: list[str]) -> None:
    assert isinstance(agent_results, dict)
    for agent in required_agents:
        assert agent in agent_results, f"Expected agent '{agent}' in agent_results, got {list(agent_results.keys())}"
        content = agent_results[agent]
        assert isinstance(content, str) and content.strip(), f"Agent '{agent}' output is empty"
        assert not content.startswith(f"**{agent.title()} Agent Error:**"), (
            f"Agent '{agent}' returned an explicit error payload: {content[:200]}"
        )
        assert "pipeline failed" not in content.lower(), (
            f"Agent '{agent}' returned pipeline failure content: {content[:200]}"
        )


def _assert_progress_markers(progress_log: Path, markers: list[str]) -> None:
    text = progress_log.read_text(encoding="utf-8")
    missing = [marker for marker in markers if marker not in text]
    assert not missing, (
        "Live pipeline did not reach required nodes: "
        f"{missing}. Progress log: {progress_log}"
    )


def _assert_progress_patterns(progress_log: Path, patterns: list[str]) -> None:
    text = progress_log.read_text(encoding="utf-8")
    missing = [pattern for pattern in patterns if not re.search(pattern, text)]
    assert not missing, (
        "Live pipeline did not match required node patterns: "
        f"{missing}. Progress log: {progress_log}"
    )


def _configure_live_smoke_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Keep live tests practical by capping search fan-out.
    This still runs the full pipeline, but with reduced crawl breadth.
    """
    original_deep_search = web_search.deep_search

    def _capped_deep_search(
        queries: list[str],
        urls_per_query: int = 3,
        max_workers: int = 15,
        max_chars_per_url: int = 8000,
        search_delay: float = 0.3,
        use_news: bool = False,
    ) -> list[dict[str, str]]:
        capped_queries = queries[:2]
        started = time.time()
        print(
            "[live-smoke] deep_search start: "
            f"queries={len(capped_queries)} use_news={use_news} "
            f"urls_per_query={min(urls_per_query, 1)}",
            flush=True,
        )
        results = original_deep_search(
            queries=capped_queries,
            urls_per_query=min(urls_per_query, 1),
            max_workers=min(max_workers, 2),
            max_chars_per_url=min(max_chars_per_url, 2500),
            search_delay=max(search_delay, 0.2),
            use_news=use_news,
            download_timeout_seconds=8,
            extraction_timeout_seconds=8,
        )
        print(
            "[live-smoke] deep_search done: "
            f"results={len(results)} elapsed={time.time() - started:.2f}s",
            flush=True,
        )
        return results

    monkeypatch.setattr(web_search, "deep_search", _capped_deep_search)
    monkeypatch.setattr(equity_nodes, "deep_search", _capped_deep_search)
    monkeypatch.setattr(macro_nodes, "deep_search", _capped_deep_search)


def _merge_state_update(final_state: dict[str, Any], delta: dict[str, Any]) -> None:
    if "agent_results" in delta:
        final_state.setdefault("agent_results", {})
        final_state["agent_results"].update(delta["agent_results"])
    for key, value in delta.items():
        if key == "agent_results":
            continue
        final_state[key] = value


def _format_namespace(namespace: Any) -> str:
    if not namespace:
        return "supervisor"
    if isinstance(namespace, tuple):
        items = [str(item) for item in namespace if item]
    elif isinstance(namespace, list):
        items = [str(item) for item in namespace if item]
    else:
        items = [str(namespace)]
    return " > ".join(items) if items else "supervisor"


def _stream_worker(prompt: str, queue: Queue[tuple[str, Any]]) -> None:
    try:
        for update in supervisor_app.stream(
            {"user_prompt": prompt},
            stream_mode="updates",
            subgraphs=True,
        ):
            queue.put(("update", update))
        queue.put(("done", None))
    except Exception as exc:  # pragma: no cover - live only
        queue.put(("error", exc))


def _invoke_with_progress(prompt: str, run_label: str) -> tuple[dict[str, Any], Path]:
    data_dir = Path.cwd() / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    progress_log = data_dir / f"live_test_progress_{run_label}_{timestamp}.log"

    final_state: dict[str, Any] = {"user_prompt": prompt}
    start = time.time()
    last = start
    timeline: list[tuple[str, float, float]] = []

    hard_timeout_seconds = int(os.getenv("LIVE_TEST_TIMEOUT_SECONDS", "1800"))
    heartbeat_seconds = int(os.getenv("LIVE_TEST_HEARTBEAT_SECONDS", "20"))

    with progress_log.open("w", encoding="utf-8", buffering=1) as log:
        log.write(f"START {run_label}\n")
        log.write(f"PROMPT: {prompt}\n\n")
        log.flush()

        stream_queue: Queue[tuple[str, Any]] = Queue()
        worker = Thread(target=_stream_worker, args=(prompt, stream_queue), daemon=True)
        worker.start()
        last_node = "supervisor:start"

        while True:
            now = time.time()
            elapsed = now - start
            if elapsed > hard_timeout_seconds:
                timeout_line = (
                    f"\nTIMEOUT after {elapsed:.2f}s while waiting after '{last_node}'.\n"
                )
                print(timeout_line.strip(), flush=True)
                log.write(timeout_line)
                pytest.fail(
                    f"Live pipeline exceeded timeout ({hard_timeout_seconds}s). "
                    f"Last completed node: {last_node}. Progress log: {progress_log}"
                )

            try:
                kind, payload = stream_queue.get(timeout=heartbeat_seconds)
            except Empty:
                heartbeat_now = time.time()
                since_last = heartbeat_now - last
                heartbeat_line = (
                    f"[{elapsed:7.2f}s | +{since_last:6.2f}s] heartbeat: "
                    f"still running after '{last_node}'\n"
                )
                print(heartbeat_line.strip(), flush=True)
                log.write(heartbeat_line)
                continue

            if kind == "error":
                raise payload
            if kind == "done":
                break
            if kind != "update":
                continue

            update = payload
            now = time.time()
            gap = now - last
            elapsed = now - start
            last = now

            namespace: Any = ()
            delta_by_node: dict[str, Any] = {}
            if isinstance(update, tuple) and len(update) == 2:
                namespace, delta = update
                if isinstance(delta, dict):
                    delta_by_node = delta
            elif isinstance(update, dict):
                delta_by_node = update
            else:
                continue

            namespace_label = _format_namespace(namespace)
            for node_name, delta in delta_by_node.items():
                node_label = f"{namespace_label} > {node_name}"
                last_node = node_label
                timeline.append((node_label, gap, elapsed))
                line = f"[{elapsed:7.2f}s | +{gap:6.2f}s] completed node: {node_label}\n"
                print(line.strip(), flush=True)
                log.write(line)
                if namespace_label == "supervisor" and isinstance(delta, dict):
                    _merge_state_update(final_state, delta)

        if timeline:
            slowest = max(timeline, key=lambda x: x[1])
            summary = (
                f"\nSLOWEST GAP: +{slowest[1]:.2f}s before node '{slowest[0]}' "
                f"(elapsed {slowest[2]:.2f}s)\n"
            )
            print(summary.strip(), flush=True)
            log.write(summary)
            log.flush()

    return final_state, progress_log


def test_live_full_pipeline_single_domain_passthrough(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    End-to-end live run through the supervisor graph for a single-domain request.
    Verifies at least the equity sub-agent pipeline executes and final response is non-empty.
    """
    _assert_live_model_env_ready()
    _configure_live_smoke_limits(monkeypatch)

    prompt = (
        "Analyze AAPL equity valuation, business quality, and company-specific risks only. "
        "Do not include macroeconomic or commodity analysis."
    )
    final_state, progress_log = _invoke_with_progress(
        prompt=prompt,
        run_label="single_domain",
    )

    assert not final_state.get("error"), (
        f"Supervisor returned error: {final_state.get('error')} "
        f"(progress log: {progress_log})"
    )
    final_response = final_state.get("final_response", "")
    assert isinstance(final_response, str) and final_response.strip()

    agent_results = final_state.get("agent_results", {})
    _assert_agent_result_contract(agent_results, required_agents=["equity"])
    _assert_progress_markers(
        progress_log,
        markers=[
            "supervisor > classify_intent",
            "supervisor > run_equity_agent",
            " > fetch_company_profile",
            " > fetch_financials",
            " > research_qualitative",
            " > synthesize_research",
            " > review_and_expand_peers",
            " > generate_section",
            " > generate_summary",
            " > verify_content",
            " > save_report",
        ],
    )
    _assert_progress_patterns(
        progress_log,
        patterns=[
            r"run_equity_agent:[^\n]*> synthesize_research",
        ],
    )


def test_live_full_pipeline_multi_domain_synthesis(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    End-to-end live run through the full supervisor pipeline with all three domains.
    Verifies all sub-agents run and multi-agent synthesis produces a non-empty response.
    """
    _assert_live_model_env_ready()
    _configure_live_smoke_limits(monkeypatch)

    prompt = (
        "Create one integrated investment brief that combines: "
        "(1) AAPL equity view, (2) US interest-rate macro outlook, and "
        "(3) crude-oil commodity outlook for the next 12 months."
    )
    final_state, progress_log = _invoke_with_progress(
        prompt=prompt,
        run_label="multi_domain",
    )

    assert not final_state.get("error"), (
        f"Supervisor returned error: {final_state.get('error')} "
        f"(progress log: {progress_log})"
    )
    final_response = final_state.get("final_response", "")
    assert isinstance(final_response, str) and final_response.strip()

    agent_results = final_state.get("agent_results", {})
    _assert_agent_result_contract(
        agent_results,
        required_agents=["equity", "macro", "commodity"],
    )
    _assert_progress_markers(
        progress_log,
        markers=[
            "supervisor > run_equity_agent",
            "supervisor > run_macro_agent",
            "supervisor > run_commodity_agent",
            "supervisor > synthesize_results",
        ],
    )
    _assert_progress_patterns(
        progress_log,
        patterns=[
            r"run_equity_agent:[^\n]*> synthesize_research",
            r"run_macro_agent:[^\n]*> synthesize_research",
            r"run_commodity_agent:[^\n]*> synthesize_research",
        ],
    )

    # Multi-domain prompt should route to at least 2 agents, triggering synthesis mode.
    assert len(agent_results) >= 2
    assert not final_response.startswith("# Response to:"), (
        "Expected synthesized multi-agent output, got single-agent passthrough format."
    )
