"""Live test: AnthropicNativeTransport retries correctly on RateLimitError (429)."""

from __future__ import annotations

import time
from pathlib import Path

import anthropic
import httpx
import pytest

from src.harness.artifacts import (
    create_agent_workspace,
    initialize_run_root,
    load_transcript_entries,
)
from src.harness.presets import default_tool_allowlist
from src.harness.prompt_builder import render_task_markdown, render_tools_markdown
from src.harness.registry import build_skill_registry, get_skills_for_packs
from src.harness.transport import AnthropicNativeTransport
from src.harness.types import HarnessRequest

pytestmark = pytest.mark.live

_MODEL = "claude-sonnet-4-20250514"

_TOOL_SPECS = [
    {
        "name": "get_current_datetime",
        "description": "Return the current datetime.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    }
]


def _make_rate_limit_error() -> anthropic.RateLimitError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(429, request=request)
    return anthropic.RateLimitError(
        message="This request would exceed your organization's rate limit of 30,000 input tokens per minute.",
        response=response,
        body={
            "type": "error",
            "error": {"type": "rate_limit_error", "message": "rate limited"},
        },
    )


def _create_transport(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AnthropicNativeTransport:
    monkeypatch.chdir(tmp_path)
    request = HarnessRequest(user_prompt="What is 2+2?", run_id="rate-limit-retry-test")
    run_root, agent_id = initialize_run_root(request)
    registry = build_skill_registry()
    create_agent_workspace(
        run_root,
        agent_id=agent_id,
        parent_id="",
        preset="orchestrator",
        task_name="Root Task",
        description="Test rate limit retry",
        task_markdown=render_task_markdown(request.user_prompt),
        tools_markdown=render_tools_markdown(
            preset="orchestrator",
            available_tools=default_tool_allowlist("orchestrator"),
            available_skills=get_skills_for_packs(registry, ["core"]),
        ),
    )
    transport = AnthropicNativeTransport(
        run_root=run_root,
        agent_id=agent_id,
        model_name=_MODEL,
        system_prompt="You are a helpful assistant. Always call a tool.",
    )
    transport.ensure_initialized("What is 2+2? Call get_current_datetime.")
    return transport


def test_anthropic_transport_retries_once_on_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Transport retries after one 429, returns correct result, logs one request+response pair."""
    transport = _create_transport(tmp_path, monkeypatch)

    sleep_calls: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: sleep_calls.append(s))

    call_count = 0
    original_stream = transport.client.messages.stream
    rate_limit_error = _make_rate_limit_error()

    def _failing_then_real(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise rate_limit_error
        return original_stream(**kwargs)

    monkeypatch.setattr(transport.client.messages, "stream", _failing_then_real)

    result = transport.execute_turn(_TOOL_SPECS)

    # Retried exactly once
    assert call_count == 2, f"Expected 2 stream calls (1 fail + 1 success), got {call_count}"

    # Slept with the initial backoff delay
    assert len(sleep_calls) == 1
    assert sleep_calls[0] == 5.0, f"Expected initial delay 5.0s, got {sleep_calls[0]}"

    # Result is valid
    assert result.stop_reason is not None, "stop_reason should be set after successful retry"

    # Transcript has exactly one model_request and one assistant_response (no duplicate from retry)
    entries = load_transcript_entries(transport.run_root, transport.agent_id)
    request_entries = [e for e in entries if e["kind"] == "model_request"]
    response_entries = [e for e in entries if e["kind"] == "assistant_response"]
    assert len(request_entries) == 1, f"Expected 1 model_request, got {len(request_entries)}"
    assert len(response_entries) == 1, f"Expected 1 assistant_response, got {len(response_entries)}"


def test_anthropic_transport_reraises_after_max_retries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Transport re-raises RateLimitError after exhausting all retries."""
    from src.harness.transport import _RATE_LIMIT_MAX_RETRIES

    transport = _create_transport(tmp_path, monkeypatch)

    monkeypatch.setattr(time, "sleep", lambda _s: None)

    call_count = 0
    rate_limit_error = _make_rate_limit_error()

    def _always_rate_limited(**kwargs):  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1
        raise rate_limit_error

    monkeypatch.setattr(transport.client.messages, "stream", _always_rate_limited)

    with pytest.raises(anthropic.RateLimitError):
        transport.execute_turn(_TOOL_SPECS)

    assert call_count == _RATE_LIMIT_MAX_RETRIES + 1, (
        f"Expected {_RATE_LIMIT_MAX_RETRIES + 1} total attempts, got {call_count}"
    )
