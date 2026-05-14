"""AnthropicNativeTransport rate-limit retry behavior."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import time

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
from src.harness.transport import AnthropicNativeTransport, _RATE_LIMIT_MAX_RETRIES
from src.harness.types import HarnessRequest

pytestmark = pytest.mark.unit

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
        message="rate limited",
        response=response,
        body={
            "type": "error",
            "error": {"type": "rate_limit_error", "message": "rate limited"},
        },
    )


class _SuccessfulMessageStream:
    def __enter__(self) -> "_SuccessfulMessageStream":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def __iter__(self):
        return iter([SimpleNamespace(type="message_delta", delta=SimpleNamespace(stop_reason="tool_use"))])

    def get_final_message(self) -> object:
        return SimpleNamespace(
            content=[
                SimpleNamespace(
                    type="tool_use",
                    id="call_1",
                    name="get_current_datetime",
                    input={},
                )
            ],
        )


class _FakeAnthropicMessages:
    def __init__(self, *, failures_before_success: int | None) -> None:
        self.failures_before_success = failures_before_success
        self.call_count = 0
        self.rate_limit_error = _make_rate_limit_error()

    def stream(self, **_kwargs: object) -> _SuccessfulMessageStream:
        self.call_count += 1
        if self.failures_before_success is None or self.call_count <= self.failures_before_success:
            raise self.rate_limit_error
        return _SuccessfulMessageStream()


class _FakeAnthropicClient:
    def __init__(self, messages: _FakeAnthropicMessages) -> None:
        self.messages = messages


def _create_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    messages: _FakeAnthropicMessages,
) -> AnthropicNativeTransport:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        "src.harness.transport.anthropic.Anthropic",
        lambda **_kwargs: _FakeAnthropicClient(messages),
    )
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
    messages = _FakeAnthropicMessages(failures_before_success=1)
    transport = _create_transport(tmp_path, monkeypatch, messages)

    sleep_calls: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda seconds: sleep_calls.append(seconds))

    result = transport.execute_turn(_TOOL_SPECS)

    assert messages.call_count == 2
    assert sleep_calls == [5.0]
    assert result.stop_reason == "tool_use"
    assert result.tool_calls[0].name == "get_current_datetime"

    entries = load_transcript_entries(transport.run_root, transport.agent_id)
    request_entries = [entry for entry in entries if entry["kind"] == "model_request"]
    response_entries = [entry for entry in entries if entry["kind"] == "assistant_response"]
    assert len(request_entries) == 1
    assert len(response_entries) == 1


def test_anthropic_transport_reraises_after_max_retries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    messages = _FakeAnthropicMessages(failures_before_success=None)
    transport = _create_transport(tmp_path, monkeypatch, messages)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    with pytest.raises(anthropic.RateLimitError):
        transport.execute_turn(_TOOL_SPECS)

    assert messages.call_count == _RATE_LIMIT_MAX_RETRIES + 1
