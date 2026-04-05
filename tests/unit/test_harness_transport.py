from __future__ import annotations

from pathlib import Path

import pytest

from src.harness.artifacts import (
    append_transcript_entry,
    create_agent_workspace,
    initialize_run_root,
    load_transcript_entries,
    read_json,
)
from src.harness.presets import default_tool_allowlist, render_root_task_markdown, render_tools_markdown
from src.harness.registry import build_skill_registry, get_skills_for_packs
from src.harness.transport import BaseAgentTransport, MiniMaxAnthropicTransport
from src.harness.transport import (
    minimax_openai_base_url,
    minimax_anthropic_base_url,
    normalize_minimax_model_name,
    resolve_agent_transport,
)
from src.harness.types import HarnessRequest


class DummyTransport(BaseAgentTransport):
    def append_user_text(self, text: str) -> None:
        append_transcript_entry(
            self.run_root,
            self.agent_id,
            {"kind": "user_message", "message": {"role": "user", "content": text}},
        )

    def execute_turn(self, tool_specs: list[dict[str, object]]):  # pragma: no cover - not used in these tests.
        raise NotImplementedError

    def append_tool_results(self, tool_results: list[dict[str, object]]) -> None:  # pragma: no cover - not used.
        return None


class _FakeAnthropicBlock:
    def __init__(self, block_type: str, **payload: object) -> None:
        self.type = block_type
        for key, value in payload.items():
            setattr(self, key, value)


class _FakeAnthropicResponse:
    def __init__(self) -> None:
        self.content = [
            _FakeAnthropicBlock("thinking", thinking="Plan the next step."),
            _FakeAnthropicBlock(
                "tool_use",
                id="call_1",
                name="search_web",
                input={"query": "XOM valuation"},
            ),
        ]
        self.stop_reason = "tool_use"

    def model_dump(self, mode: str = "json") -> dict[str, object]:
        assert mode == "json"
        return {
            "content": [
                {"type": "thinking", "thinking": "Plan the next step."},
                {"type": "tool_use", "id": "call_1", "name": "search_web", "input": {"query": "XOM valuation"}},
            ],
            "stop_reason": "tool_use",
        }


class _FakeAnthropicMessages:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> _FakeAnthropicResponse:
        self.calls.append(dict(kwargs))
        return _FakeAnthropicResponse()

    def stream(self, **kwargs: object) -> "_FakeMessageStream":
        self.calls.append(dict(kwargs))
        return _FakeMessageStream()


class _FakeMessageStream:
    """Fake streaming message iterator matching the Anthropic messages.stream() API."""

    def __init__(self) -> None:
        self._events = self._build_events()

    def _build_events(self) -> list[object]:
        """Build a minimal event sequence that produces a tool_use stop_reason."""
        import types

        events = []
        # thinking block start
        events.append(types.SimpleNamespace(type="content_block_start", name="thinking", index=0))
        # tool_use block start - name="tool_use" (block type), actual tool name is in input.name
        events.append(types.SimpleNamespace(type="content_block_start", name="tool_use", index=1, id="call_1", input=types.SimpleNamespace(name="search_web", query="test")))
        # tool input delta - accumulates into current_tool_input._raw
        delta = types.SimpleNamespace(type="input_json_delta", input_json='{"name":"search_web","query":"test"}')
        events.append(types.SimpleNamespace(type="content_block_delta", delta=delta, index=2))
        # tool_use block stop
        events.append(types.SimpleNamespace(type="content_block_stop", name="tool_use", index=1))
        # thinking block stop
        events.append(types.SimpleNamespace(type="content_block_stop", name="thinking", index=0))
        # message stop
        events.append(types.SimpleNamespace(type="message_stop", stop_reason="tool_use", index=3))
        return events

    def __enter__(self) -> "_FakeMessageStream":
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def __iter__(self) -> "_FakeMessageStream":
        return self

    def __next__(self) -> object:
        if self._events:
            return self._events.pop(0)
        raise StopIteration

    def get_last_message(self) -> dict[str, object]:
        return {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Plan the next step."},
                {"type": "tool_use", "id": "call_1", "name": "search_web", "input": {"query": "test"}},
            ],
            "stop_reason": "tool_use",
        }

    def get_final_message(self) -> object:
        """Return a mock message with content blocks for final extraction."""
        import types

        class _FakeContentBlock:
            def __init__(self, block_type: str, **kwargs: object) -> None:
                self.type = block_type
                for k, v in kwargs.items():
                    setattr(self, k, v)

        return types.SimpleNamespace(
            id="msg_1",
            type="message",
            role="assistant",
            content=[
                _FakeContentBlock("thinking", thinking="Plan the next step."),
                _FakeContentBlock("tool_use", id="call_1", name="search_web", input={"query": "test"}),
            ],
            stop_reason="tool_use",
        )


class _FakeAnthropicClient:
    def __init__(self) -> None:
        self.messages = _FakeAnthropicMessages()


def _create_agent_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str]:
    monkeypatch.chdir(tmp_path)
    request = HarnessRequest(user_prompt="Analyze XOM", run_id="transport-test")
    run_root, agent_id = initialize_run_root(request)
    registry = build_skill_registry()
    create_agent_workspace(
        run_root,
        agent_id=agent_id,
        parent_id="",
        preset="research",
        task_name="Root Task",
        description="Analyze XOM",
        task_markdown=render_root_task_markdown(request.user_prompt),
        tools_markdown=render_tools_markdown(
            preset="research",
            available_tools=default_tool_allowlist("research"),
            available_skills=get_skills_for_packs(registry, ["core"]),
        ),
    )
    return Path(run_root), agent_id


def test_auto_transport_prefers_anthropic_for_minimax(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MINIMAX_BASE_URL", raising=False)

    assert resolve_agent_transport("auto", "minimax/MiniMax-M2.7") == "minimax_anthropic"
    assert resolve_agent_transport("auto", "gpt-4o") == "text_json"


def test_minimax_default_base_urls_match_official_docs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MINIMAX_BASE_URL", raising=False)
    monkeypatch.delenv("MINIMAX_ANTHROPIC_BASE_URL", raising=False)

    assert minimax_openai_base_url() == "https://api.minimaxi.com/v1"
    assert minimax_anthropic_base_url() == "https://api.minimaxi.com/anthropic"


def test_minimax_anthropic_base_url_derives_from_openai_base(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1")
    monkeypatch.delenv("MINIMAX_ANTHROPIC_BASE_URL", raising=False)

    assert minimax_anthropic_base_url() == "https://api.minimaxi.com/anthropic"


def test_normalize_minimax_model_name_strips_prefix() -> None:
    assert normalize_minimax_model_name("minimax/MiniMax-M2.7") == "MiniMax-M2.7"
    assert normalize_minimax_model_name("MiniMax-M2.7") == "MiniMax-M2.7"


def test_transport_initialization_logs_system_prompt_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, agent_id = _create_agent_workspace(tmp_path, monkeypatch)
    transport = DummyTransport(
        run_root=str(run_root),
        agent_id=agent_id,
        model_name="minimax/MiniMax-M2.7",
        system_prompt="System v1",
    )

    transport.ensure_initialized("Initial prompt")
    transport.update_system_prompt("System v2")

    entries = load_transcript_entries(run_root, agent_id)
    assert [entry["kind"] for entry in entries[:3]] == [
        "system_prompt_snapshot",
        "user_message",
        "system_prompt_snapshot",
    ]
    assert entries[0]["system_prompt"] == "System v1"
    assert entries[2]["system_prompt"] == "System v2"
    assert Path(entries[0]["artifact_path"]).exists()
    assert Path(entries[2]["artifact_path"]).exists()


def test_anthropic_transport_logs_full_turn_request_and_decision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, agent_id = _create_agent_workspace(tmp_path, monkeypatch)
    fake_client = _FakeAnthropicClient()
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
    monkeypatch.setattr(
        "src.harness.transport.anthropic.Anthropic",
        lambda **_: fake_client,
    )
    transport = MiniMaxAnthropicTransport(
        run_root=str(run_root),
        agent_id=agent_id,
        model_name="minimax/MiniMax-M2.7",
        system_prompt="System v1",
    )
    tool_specs = [
        {
            "name": "search_web",
            "description": "Search the web",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }
    ]

    transport.ensure_initialized("Initial prompt")
    result = transport.execute_turn(tool_specs)

    assert result.stop_reason == "tool_use"
    entries = load_transcript_entries(run_root, agent_id)
    request_entry = next(entry for entry in entries if entry["kind"] == "model_request")
    response_entry = next(entry for entry in entries if entry["kind"] == "assistant_response")
    request_payload = read_json(request_entry["artifact_path"])

    assert request_payload["request"]["system"] == "System v1"
    assert request_payload["request"]["messages"][0]["role"] == "user"
    assert request_payload["request"]["tools"][0]["name"] == "search_web"
    assert response_entry["decision"]["tool_calls"][0]["name"] == "search_web"
    assert response_entry["decision"]["provider_thinking_blocks"] == 1
    assert Path(response_entry["artifact_path"]).exists()
    response_payload = read_json(response_entry["artifact_path"])
    assert response_payload["decision"]["stop_reason"] == "tool_use"
