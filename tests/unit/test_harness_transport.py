from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.harness.artifacts import (
    append_transcript_entry,
    create_agent_workspace,
    initialize_run_root,
    load_transcript_entries,
    read_json,
)
from src.harness.presets import default_tool_allowlist
from src.harness.prompt_builder import render_task_markdown, render_tools_markdown
from src.harness.registry import build_skill_registry, get_skills_for_packs
from src.harness.transport import BaseAgentTransport, MiniMaxAnthropicTransport, OpenAINativeTransport
from src.harness.transport import (
    _persist_history_compaction_state,
    _transcript_messages,
    minimax_openai_base_url,
    minimax_anthropic_base_url,
    normalize_minimax_model_name,
    preflight_history_compaction,
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


class _FakeOpenAICompletions:
    def create(self, **_kwargs: object) -> object:
        class _FakeChoice:
            finish_reason = "stop"

            class message:
                content = "done"
                tool_calls: list[object] = []

        class _FakeResponse:
            choices = [_FakeChoice()]

        return _FakeResponse()


class _FakeOpenAIClient:
    def __init__(self) -> None:
        class _FakeChat:
            completions = _FakeOpenAICompletions()

        self.chat = _FakeChat()


class _FakeSummaryLLM:
    def invoke(self, _messages: object) -> object:
        class _Response:
            content = (
                "## Objective\n"
                "- Keep improving the draft.\n\n"
                "## Decisions\n"
                "- Compacted older turns.\n\n"
                "## Evidence\n"
                "- Older findings preserved.\n\n"
                "## Files\n"
                "- publish/final.md\n\n"
                "## Open Issues\n"
                "- None\n\n"
                "## Reviewer Feedback\n"
                "- None\n\n"
                "## Recent Failures\n"
                "- None\n"
            )

        return _Response()


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
        task_markdown=render_task_markdown(request.user_prompt),
        tools_markdown=render_tools_markdown(
            preset="research",
            available_tools=default_tool_allowlist("research"),
            available_skills=get_skills_for_packs(registry, ["core"]),
        ),
    )
    return Path(run_root), agent_id


def test_auto_transport_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MINIMAX_BASE_URL", raising=False)

    assert resolve_agent_transport("auto", "minimax/MiniMax-M2.7") == "minimax_anthropic"
    assert resolve_agent_transport("auto", "claude-sonnet-4-6") == "anthropic"
    assert resolve_agent_transport("auto", "gpt-4o") == "openai"
    assert resolve_agent_transport("auto", "o3-mini") == "openai"
    assert resolve_agent_transport("auto", "o4-mini") == "openai"
    assert resolve_agent_transport("auto", "sf/Qwen2.5-72B") == "text_json"


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


def test_anthropic_transport_replay_strips_thinking_blocks(
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

    transport.ensure_initialized("Initial prompt")
    append_transcript_entry(
        run_root,
        agent_id,
        {
            "kind": "assistant_response",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Hidden reasoning."},
                    {"type": "text", "text": "Visible answer."},
                    {"type": "tool_use", "id": "call_old", "name": "search_web", "input": {"query": "prior"}},
                ],
            },
        },
    )
    append_transcript_entry(
        run_root,
        agent_id,
        {
            "kind": "tool_result",
            "message": {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "call_old",
                        "content": '{"status":"ok"}',
                    }
                ],
            },
        },
    )

    transport.execute_turn([])

    entries = load_transcript_entries(run_root, agent_id)
    request_entry = next(entry for entry in entries if entry["kind"] == "model_request")
    messages = read_json(request_entry["artifact_path"])["request"]["messages"]
    assistant_message = next(message for message in messages if message["role"] == "assistant")

    assert assistant_message["content"] == [
        {"type": "text", "text": "Visible answer."},
        {"type": "tool_use", "id": "call_old", "name": "search_web", "input": {"query": "prior"}},
    ]
    assert not any(block.get("type") == "thinking" for block in assistant_message["content"])


def test_anthropic_transport_skips_assistant_messages_with_only_thinking(
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

    transport.ensure_initialized("Initial prompt")
    append_transcript_entry(
        run_root,
        agent_id,
        {
            "kind": "assistant_response",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Hidden reasoning only."},
                ],
            },
        },
    )

    transport.execute_turn([])

    entries = load_transcript_entries(run_root, agent_id)
    request_entry = next(entry for entry in entries if entry["kind"] == "model_request")
    messages = read_json(request_entry["artifact_path"])["request"]["messages"]

    assert all(message["role"] != "assistant" for message in messages)


def test_openai_transport_replay_strips_reasoning_content(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, agent_id = _create_agent_workspace(tmp_path, monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "src.harness.transport.OpenAI",
        lambda **_: _FakeOpenAIClient(),
    )
    transport = OpenAINativeTransport(
        run_root=str(run_root),
        agent_id=agent_id,
        model_name="gpt-4o",
        system_prompt="System v1",
    )

    transport.ensure_initialized("Initial prompt")
    append_transcript_entry(
        run_root,
        agent_id,
        {
            "kind": "assistant_response",
            "message": {
                "role": "assistant",
                "content": "Visible answer.",
                "reasoning_content": "Hidden reasoning.",
            },
        },
    )

    transport.execute_turn([])

    entries = load_transcript_entries(run_root, agent_id)
    request_entry = next(entry for entry in entries if entry["kind"] == "model_request")
    messages = read_json(request_entry["artifact_path"])["request"]["messages"]
    assistant_message = next(message for message in messages if message["role"] == "assistant")

    assert assistant_message["content"] == "Visible answer."
    assert "reasoning_content" not in assistant_message


def test_model_request_artifact_records_compaction_preflight_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, agent_id = _create_agent_workspace(tmp_path, monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "src.harness.transport.OpenAI",
        lambda **_: _FakeOpenAIClient(),
    )
    _persist_history_compaction_state(
        str(run_root),
        agent_id,
        compacted_user_turns=2,
        estimated_input_tokens_before=180_000,
        estimated_input_tokens_after=165_000,
        compaction_applied=True,
        soft_overflow=False,
        hard_overflow=False,
    )
    transport = OpenAINativeTransport(
        run_root=str(run_root),
        agent_id=agent_id,
        model_name="gpt-4o",
        system_prompt="System v1",
    )

    transport.ensure_initialized("Initial prompt")
    transport.execute_turn([])

    entries = load_transcript_entries(run_root, agent_id)
    request_entry = next(entry for entry in entries if entry["kind"] == "model_request")
    request_payload = read_json(request_entry["artifact_path"])

    assert request_payload["preflight"]["estimated_input_tokens_before"] == 180_000
    assert request_payload["preflight"]["estimated_input_tokens_after"] == 165_000
    assert request_payload["preflight"]["compaction_applied"] is True
    assert request_entry["summary"]["estimated_input_tokens_after"] == 165_000
    assert request_entry["summary"]["compaction_applied"] is True


def test_preflight_history_compaction_keeps_full_raw_replay_under_budget(
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
    transport.ensure_initialized("Turn 1")
    for turn_number in range(1, 6):
        if turn_number > 1:
            transport.append_user_text(f"Turn {turn_number}")
        append_transcript_entry(
            run_root,
            agent_id,
            {
                "kind": "assistant_response",
                "message": {
                    "role": "assistant",
                    "content": f"Assistant response {turn_number}",
                },
            },
        )

    monkeypatch.setattr("src.harness.transport.estimate_payload_input_tokens", lambda _payload: 120_000)

    result = preflight_history_compaction(
        transport_name="openai",
        run_root=str(run_root),
        agent_id=agent_id,
        model_name="gpt-4o",
        system_prompt="System v1",
        pending_user_prompt="Turn 6",
        tool_specs=[],
    )

    replay_messages = _transcript_messages(str(run_root), agent_id)
    replay_user_messages = [message for message in replay_messages if message["role"] == "user"]

    assert not result.compaction_changed
    assert result.estimated_input_tokens_before == 120_000
    assert result.estimated_input_tokens_after == 120_000
    assert [message["content"] for message in replay_user_messages] == [
        "Turn 1",
        "Turn 2",
        "Turn 3",
        "Turn 4",
        "Turn 5",
    ]
    assert (run_root / "agents" / agent_id / "state" / "history_summary.md").read_text(encoding="utf-8") == ""


def test_preflight_history_compaction_compacts_oldest_turns_only_when_over_budget(
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
    transport.ensure_initialized("Turn 1")
    for turn_number in range(1, 6):
        if turn_number > 1:
            transport.append_user_text(f"Turn {turn_number}")
        append_transcript_entry(
            run_root,
            agent_id,
            {
                "kind": "assistant_response",
                "message": {
                    "role": "assistant",
                    "content": f"Assistant response {turn_number}",
                },
            },
        )

    def _estimate(payload: dict[str, object]) -> int:
        rendered = json.dumps(payload, ensure_ascii=True)
        if "Turn 1" in rendered or "Turn 2" in rendered:
            return 171_500
        return 160_000

    monkeypatch.setattr("src.harness.transport.estimate_payload_input_tokens", _estimate)
    monkeypatch.setattr("src.harness.transport.get_model", lambda *_args: "summary-model")
    monkeypatch.setattr("src.harness.transport.get_llm", lambda _model: _FakeSummaryLLM())

    result = preflight_history_compaction(
        transport_name="openai",
        run_root=str(run_root),
        agent_id=agent_id,
        model_name="gpt-4o",
        system_prompt="System v1",
        pending_user_prompt="Turn 6",
        tool_specs=[],
    )

    history_summary = (run_root / "agents" / agent_id / "state" / "history_summary.md").read_text(
        encoding="utf-8"
    )
    replay_messages = _transcript_messages(str(run_root), agent_id)
    replay_user_messages = [message for message in replay_messages if message["role"] == "user"]

    assert result.compaction_changed
    assert result.estimated_input_tokens_before == 171_500
    assert result.estimated_input_tokens_after == 160_000
    assert result.compacted_user_turns == 2
    assert "Keep improving the draft." in history_summary
    assert [message["content"] for message in replay_user_messages] == ["Turn 3", "Turn 4", "Turn 5"]


def test_preflight_history_compaction_reports_hard_overflow_after_full_compaction(
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
    transport.ensure_initialized("Turn 1")
    transport.append_user_text("Turn 2")

    monkeypatch.setattr("src.harness.transport.estimate_payload_input_tokens", lambda _payload: 205_000)
    monkeypatch.setattr("src.harness.transport.get_model", lambda *_args: "summary-model")
    monkeypatch.setattr("src.harness.transport.get_llm", lambda _model: _FakeSummaryLLM())

    result = preflight_history_compaction(
        transport_name="openai",
        run_root=str(run_root),
        agent_id=agent_id,
        model_name="gpt-4o",
        system_prompt="System v1",
        pending_user_prompt="Turn 3",
        tool_specs=[],
    )

    assert result.soft_overflow
    assert result.hard_overflow
    assert result.compacted_user_turns == 1


def test_replay_drops_orphan_assistant_tool_calls_before_next_turn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, agent_id = _create_agent_workspace(tmp_path, monkeypatch)
    transport = DummyTransport(
        run_root=str(run_root),
        agent_id=agent_id,
        model_name="gpt-4o",
        system_prompt="System v1",
    )
    transport.ensure_initialized("Turn 1")
    append_transcript_entry(
        run_root,
        agent_id,
        {
            "kind": "assistant_response",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_stale",
                        "type": "function",
                        "function": {
                            "name": "edit_file",
                            "arguments": '{"path":"publish/final.md","target_text":"stale body"}',
                        },
                    }
                ],
            },
        },
    )
    transport.append_user_text("Turn 2")

    replay_messages = _transcript_messages(str(run_root), agent_id)

    assert [message["role"] for message in replay_messages] == ["user", "user"]
    assert replay_messages[-1]["content"] == "Turn 2"
