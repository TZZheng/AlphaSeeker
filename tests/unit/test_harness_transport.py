from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.harness.artifacts import (
    agent_workspace_paths,
    append_conversation_entry,
    append_transcript_entry,
    create_agent_workspace,
    initialize_run_root,
    load_conversation_entries,
    load_transcript_entries,
    read_json,
)
from src.harness.presets import default_tool_allowlist
from src.harness.prompt_builder import render_task_markdown, render_tools_markdown
from src.harness.registry import build_skill_registry, get_skills_for_packs
from src.harness.transport import (
    AnthropicNativeTransport,
    BaseAgentTransport,
    MiniMaxAnthropicTransport,
    MiniMaxOpenAITransport,
    OpenAINativeTransport,
    CodexNativeTransport,
)
from src.harness.transport import (
    _build_request_payload,
    _codex_responses_tool_specs,
    _canonicalize_message_for_conversation,
    _conversation_messages,
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
        message = {"role": "user", "content": text}
        append_transcript_entry(
            self.run_root,
            self.agent_id,
            {"kind": "user_message", "message": message},
        )
        self._append_conversation_message(kind="user_message", message=message)

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
    def __init__(self, responses: list[object] | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self._responses = list(responses or [])

    def create(self, **_kwargs: object) -> object:
        self.calls.append(dict(_kwargs))
        if self._responses:
            return self._responses.pop(0)

        class _FakeChoice:
            finish_reason = "stop"

            class message:
                content = "done"
                tool_calls: list[object] = []

        class _FakeResponse:
            choices = [_FakeChoice()]

        return _FakeResponse()


class _FakeOpenAIClient:
    def __init__(self, responses: list[object] | None = None) -> None:
        self.completions = _FakeOpenAICompletions(responses)

        class _FakeChat:
            pass

        _FakeChat.completions = self.completions
        self.chat = _FakeChat()


class _FakeCodexResponses:
    def __init__(self, events: list[object] | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self._events = list(events or [])

    def create(self, **kwargs: object) -> list[object]:
        self.calls.append(dict(kwargs))
        return list(self._events)


class _FakeCodexClient:
    def __init__(self, events: list[object] | None = None) -> None:
        self.api_key = "initial-token"
        self.responses = _FakeCodexResponses(events)


def _fake_codex_tool_stream() -> list[object]:
    return [
        SimpleNamespace(type="response.output_text.delta", delta="Need data."),
        SimpleNamespace(
            type="response.output_item.added",
            item=SimpleNamespace(type="function_call", call_id="call_1", name="read"),
        ),
        SimpleNamespace(type="response.function_call_arguments.delta", delta='{"path"'),
        SimpleNamespace(type="response.function_call_arguments.delta", delta=':"publish/final.md"}'),
        SimpleNamespace(
            type="response.output_item.done",
            item=SimpleNamespace(type="function_call", call_id="call_1", name="read", arguments='{"path":"publish/final.md"}'),
        ),
        SimpleNamespace(type="response.completed", response=SimpleNamespace(id="resp_1", usage=None)),
    ]


def _fake_openai_tool_call(call_id: str, name: str, arguments: dict[str, object]) -> object:
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments, ensure_ascii=True)),
    )


def _fake_openai_response(
    *,
    content: str | None = "done",
    tool_calls: list[object] | None = None,
    finish_reason: str = "stop",
) -> object:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content, tool_calls=tool_calls or []),
            )
        ]
    )


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


def _append_model_visible_entry(
    run_root: Path,
    agent_id: str,
    *,
    kind: str,
    message: dict[str, object],
    transport: str = "test",
) -> None:
    append_transcript_entry(run_root, agent_id, {"kind": kind, "message": message})
    canonical_message = _canonicalize_message_for_conversation(message, entry_kind=kind)
    if canonical_message is None:
        return
    append_conversation_entry(
        run_root,
        agent_id,
        {
            "kind": kind,
            "transport": transport,
            "created_at": "2026-01-01T00:00:00+00:00",
            "message": canonical_message,
        },
    )


def test_auto_transport_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MINIMAX_BASE_URL", raising=False)

    assert resolve_agent_transport("auto", "codex/gpt-5.5") == "codex"
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


def test_workspace_initializes_empty_canonical_conversation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, agent_id = _create_agent_workspace(tmp_path, monkeypatch)
    conversation_path = agent_workspace_paths(run_root, agent_id)["conversation"]

    assert conversation_path.exists()
    assert load_conversation_entries(run_root, agent_id) == []


def test_canonical_conversation_appends_model_visible_rows_in_order(
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

    transport.append_user_text("Turn 1")
    _append_model_visible_entry(
        run_root,
        agent_id,
        kind="assistant_response",
        message={
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read", "arguments": '{"path":"publish/final.md"}'},
                }
            ],
        },
    )
    _append_model_visible_entry(
        run_root,
        agent_id,
        kind="tool_result",
        message={"role": "tool", "tool_call_id": "call_1", "content": '{"status":"ok"}'},
    )

    rows = load_conversation_entries(run_root, agent_id)

    assert [row["kind"] for row in rows] == ["user_message", "assistant_response", "tool_result"]
    assert all("transport" in row and "created_at" in row and "message" in row for row in rows)


@pytest.mark.parametrize(
    ("transport_name", "assistant_message", "tool_result_message", "tool_result_role"),
    [
        (
            "minimax_anthropic",
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "call_1", "name": "read", "input": {"path": "a.md"}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": '{"status":"ok"}'}],
            },
            "user",
        ),
        (
            "anthropic",
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "call_1", "name": "read", "input": {"path": "a.md"}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "call_1", "content": '{"status":"ok"}'}],
            },
            "user",
        ),
        (
            "minimax_openai",
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read", "arguments": '{"path":"a.md"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": '{"status":"ok"}'},
            "tool",
        ),
        (
            "openai",
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read", "arguments": '{"path":"a.md"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": '{"status":"ok"}'},
            "tool",
        ),
    ],
)
def test_canonical_conversation_keeps_tool_results_paired_after_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    transport_name: str,
    assistant_message: dict[str, object],
    tool_result_message: dict[str, object],
    tool_result_role: str,
) -> None:
    run_root, agent_id = _create_agent_workspace(tmp_path, monkeypatch)
    append_conversation_entry(
        run_root,
        agent_id,
        {
            "kind": "user_message",
            "transport": transport_name,
            "created_at": "2026-01-01T00:00:00+00:00",
            "message": {"role": "user", "content": "Turn 1"},
        },
    )
    for kind, message in (("assistant_response", assistant_message), ("tool_result", tool_result_message)):
        canonical_message = _canonicalize_message_for_conversation(message, entry_kind=kind)
        assert canonical_message is not None
        append_conversation_entry(
            run_root,
            agent_id,
            {
                "kind": kind,
                "transport": transport_name,
                "created_at": "2026-01-01T00:00:01+00:00",
                "message": canonical_message,
            },
        )

    messages = _conversation_messages(str(run_root), agent_id)

    assert [message["role"] for message in messages] == ["user", "assistant", tool_result_role]
    assert messages[1]["role"] == "assistant"
    assert messages[2]["role"] == tool_result_role


@pytest.mark.parametrize(
    ("transport_name", "transport_class", "model_name", "env_key", "assistant_message", "tool_result_role"),
    [
        (
            "minimax_anthropic",
            MiniMaxAnthropicTransport,
            "minimax/MiniMax-M2.7",
            "MINIMAX_API_KEY",
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "call_write", "name": "write", "input": {"path": "publish/final.md"}}],
            },
            "user",
        ),
        (
            "anthropic",
            AnthropicNativeTransport,
            "claude-sonnet-4-6",
            "ANTHROPIC_API_KEY",
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "call_write", "name": "write", "input": {"path": "publish/final.md"}}],
            },
            "user",
        ),
        (
            "minimax_openai",
            MiniMaxOpenAITransport,
            "minimax/MiniMax-M2.7",
            "MINIMAX_API_KEY",
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_write",
                        "type": "function",
                        "function": {"name": "write", "arguments": '{"path":"publish/final.md"}'},
                    }
                ],
            },
            "tool",
        ),
        (
            "openai",
            OpenAINativeTransport,
            "gpt-4o",
            "OPENAI_API_KEY",
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_write",
                        "type": "function",
                        "function": {"name": "write", "arguments": '{"path":"publish/final.md"}'},
                    }
                ],
            },
            "tool",
        ),
        (
            "codex",
            CodexNativeTransport,
            "codex/gpt-5.5",
            "",
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_write",
                        "type": "function",
                        "function": {"name": "write", "arguments": '{"path":"publish/final.md"}'},
                    }
                ],
            },
            "tool",
        ),
    ],
)
def test_native_tool_results_use_toolview_conversation_and_full_transcript(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    transport_name: str,
    transport_class: type[BaseAgentTransport],
    model_name: str,
    env_key: str,
    assistant_message: dict[str, object],
    tool_result_role: str,
) -> None:
    run_root, agent_id = _create_agent_workspace(tmp_path, monkeypatch)
    if env_key:
        monkeypatch.setenv(env_key, "test-key")
    if transport_name in {"anthropic", "minimax_anthropic"}:
        monkeypatch.setattr("src.harness.transport.anthropic.Anthropic", lambda **_: _FakeAnthropicClient())
    elif transport_name == "codex":
        monkeypatch.setattr("src.harness.transport.CodexTokenManager", lambda: SimpleNamespace(get_access_token=lambda: "codex-token"))
        monkeypatch.setattr("src.harness.transport.OpenAI", lambda **_: _FakeCodexClient())
    else:
        monkeypatch.setattr("src.harness.transport.OpenAI", lambda **_: _FakeOpenAIClient())
    transport = transport_class(
        run_root=str(run_root),
        agent_id=agent_id,
        model_name=model_name,
        system_prompt="System v1",
    )
    full_content = "RAW_FILE_CONTENT_" + ("A" * 1800)
    conversation_result = {
        "tool_name": "write",
        "status": "ok",
        "description": "Wrote publish/final.md",
        "path": "publish/final.md",
        "operation": "overwrite",
        "line_count": 12,
    }

    transport.append_user_text("Initial prompt")
    _append_model_visible_entry(
        run_root,
        agent_id,
        kind="assistant_response",
        message=assistant_message,
        transport=transport_name,
    )
    transport.append_tool_results(
        [
            {
                "call_id": "call_write",
                "name": "write",
                "result": {"status": "ok", "content": full_content},
                "conversation_result": conversation_result,
            }
        ]
    )

    transcript_tool_result = next(
        entry for entry in reversed(load_transcript_entries(run_root, agent_id)) if entry["kind"] == "tool_result"
    )
    conversation_tool_result = next(
        entry for entry in reversed(load_conversation_entries(run_root, agent_id)) if entry["kind"] == "tool_result"
    )
    replay_messages = _conversation_messages(str(run_root), agent_id)
    request_payload = _build_request_payload(
        transport_name=transport_name,
        model_name=model_name,
        system_prompt="System v1",
        transcript_messages=replay_messages,
        tool_specs=[],
    )
    rendered_transcript = json.dumps(transcript_tool_result["message"], ensure_ascii=True)
    rendered_conversation = json.dumps(conversation_tool_result["message"], ensure_ascii=True)
    rendered_request = json.dumps(request_payload, ensure_ascii=True)
    replay_tool_message = replay_messages[2]
    if tool_result_role == "user":
        replay_tool_result = json.loads(replay_tool_message["content"][0]["content"])
    else:
        replay_tool_result = json.loads(replay_tool_message["content"])

    assert full_content in rendered_transcript
    assert full_content not in rendered_conversation
    assert full_content not in rendered_request
    assert replay_tool_result == conversation_result
    assert "Wrote publish/final.md" in rendered_conversation
    assert "Wrote publish/final.md" in rendered_request
    assert [message["role"] for message in replay_messages] == ["user", "assistant", tool_result_role]


def test_missing_canonical_conversation_is_hard_cutover_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, agent_id = _create_agent_workspace(tmp_path, monkeypatch)
    agent_workspace_paths(run_root, agent_id)["conversation"].unlink()

    with pytest.raises(FileNotFoundError, match="Canonical conversation state is missing"):
        _conversation_messages(str(run_root), agent_id)


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
    _append_model_visible_entry(
        run_root,
        agent_id,
        kind="assistant_response",
        message={
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Hidden reasoning."},
                {"type": "text", "text": "Visible answer."},
                {"type": "tool_use", "id": "call_old", "name": "search_web", "input": {"query": "prior"}},
            ],
        },
    )
    _append_model_visible_entry(
        run_root,
        agent_id,
        kind="tool_result",
        message={
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_old",
                    "content": '{"status":"ok"}',
                }
            ],
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
    _append_model_visible_entry(
        run_root,
        agent_id,
        kind="assistant_response",
        message={
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Hidden reasoning only."},
            ],
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
    _append_model_visible_entry(
        run_root,
        agent_id,
        kind="assistant_response",
        message={
            "role": "assistant",
            "content": "Visible answer.",
            "reasoning_content": "Hidden reasoning.",
        },
    )

    transport.execute_turn([])

    entries = load_transcript_entries(run_root, agent_id)
    request_entry = next(entry for entry in entries if entry["kind"] == "model_request")
    messages = read_json(request_entry["artifact_path"])["request"]["messages"]
    assistant_message = next(message for message in messages if message["role"] == "assistant")

    assert assistant_message["content"] == "Visible answer."
    assert "reasoning_content" not in assistant_message


def test_openai_transport_request_uses_canonical_conversation_not_transcript(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, agent_id = _create_agent_workspace(tmp_path, monkeypatch)
    fake_client = _FakeOpenAIClient()
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("src.harness.transport.OpenAI", lambda **_: fake_client)
    append_transcript_entry(
        run_root,
        agent_id,
        {"kind": "user_message", "message": {"role": "user", "content": "transcript-only"}},
    )
    append_conversation_entry(
        run_root,
        agent_id,
        {
            "kind": "user_message",
            "transport": "openai",
            "created_at": "2026-01-01T00:00:00+00:00",
            "message": {"role": "user", "content": "canonical-only"},
        },
    )
    transport = OpenAINativeTransport(
        run_root=str(run_root),
        agent_id=agent_id,
        model_name="gpt-4o",
        system_prompt="System v1",
    )

    transport.execute_turn([])

    rendered_messages = json.dumps(fake_client.completions.calls[0]["messages"], ensure_ascii=True)
    assert "canonical-only" in rendered_messages
    assert "transcript-only" not in rendered_messages


def test_condense_context_result_cuts_prior_canonical_content(
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
    old_page_body = "OLD_RAW_WEB_CONTENT_SHOULD_DISAPPEAR"

    transport.append_user_text("Old turn")
    _append_model_visible_entry(
        run_root,
        agent_id,
        kind="assistant_response",
        message={
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_read_web",
                    "type": "function",
                    "function": {
                        "name": "read_web_pages",
                        "arguments": '{"urls":["https://example.com/old"]}',
                    },
                }
            ],
        },
    )
    _append_model_visible_entry(
        run_root,
        agent_id,
        kind="tool_result",
        message={
            "role": "tool",
            "tool_call_id": "call_read_web",
            "content": json.dumps(
                {
                    "tool_name": "read_web_pages",
                    "status": "ok",
                    "description": "Read page.",
                    "content": old_page_body,
                },
                ensure_ascii=True,
            ),
        },
    )
    transport.append_user_text("Condense now")
    _append_model_visible_entry(
        run_root,
        agent_id,
        kind="assistant_response",
        message={
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_condense",
                    "type": "function",
                    "function": {
                        "name": "condense_context",
                        "arguments": '{"purpose":"continue research"}',
                    },
                }
            ],
        },
    )
    _append_model_visible_entry(
        run_root,
        agent_id,
        kind="tool_result",
        message={
            "role": "tool",
            "tool_call_id": "call_condense",
            "content": json.dumps(
                {
                    "tool_name": "condense_context",
                    "status": "ok",
                    "description": "Condensed context.",
                    "content": "Condensed durable context.",
                },
                ensure_ascii=True,
            ),
        },
    )
    transport.append_user_text("Continue")

    messages = _conversation_messages(str(run_root), agent_id)
    rendered = json.dumps(messages, ensure_ascii=True)

    assert [message["role"] for message in messages] == ["user", "assistant", "tool", "user"]
    assert "Condensed durable context." in rendered
    assert old_page_body not in rendered
    assert "Old turn" not in rendered
    assert "Continue" in rendered


def test_openai_transport_canonical_request_omits_large_write_arguments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, agent_id = _create_agent_workspace(tmp_path, monkeypatch)
    huge_content = "A" * 1800 + "FULL_CONTENT_END"
    fake_client = _FakeOpenAIClient(
        responses=[
            _fake_openai_response(
                content=None,
                tool_calls=[
                    _fake_openai_tool_call(
                        "call_write",
                        "write",
                        {"path": "publish/final.md", "content": huge_content},
                    )
                ],
                finish_reason="tool_calls",
            ),
            _fake_openai_response(content="done"),
        ]
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("src.harness.transport.OpenAI", lambda **_: fake_client)
    transport = OpenAINativeTransport(
        run_root=str(run_root),
        agent_id=agent_id,
        model_name="gpt-4o",
        system_prompt="System v1",
    )

    transport.append_user_text("Initial prompt")
    result = transport.execute_turn([])
    transport.append_tool_results([{"call_id": "call_write", "result": {"status": "ok"}}])
    transport.append_user_text("Revise")
    transport.execute_turn([])

    transcript_response = next(
        entry for entry in load_transcript_entries(run_root, agent_id) if entry["kind"] == "assistant_response"
    )
    raw_arguments = transcript_response["message"]["tool_calls"][0]["function"]["arguments"]
    second_request_messages = fake_client.completions.calls[1]["messages"]
    rendered_second_request = json.dumps(second_request_messages, ensure_ascii=True)

    assert result.tool_calls[0].arguments["content"] == huge_content
    assert "FULL_CONTENT_END" in raw_arguments
    assert "[omitted " in rendered_second_request
    assert "read the current file before revising" in rendered_second_request
    assert "FULL_CONTENT_END" not in rendered_second_request


def test_codex_tool_specs_are_flat_and_scrub_top_level_combinators() -> None:
    tools = _codex_responses_tool_specs(
        [
            {
                "name": "read",
                "description": "Read a file",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string", "enum": ["a.md"]}},
                    "required": ["path"],
                    "oneOf": [],
                    "enum": [],
                },
            }
        ]
    )

    assert tools == [
        {
            "type": "function",
            "name": "read",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "enum": ["a.md"]}},
                "required": ["path"],
            },
        }
    ]


def test_codex_native_transport_sends_stateless_streaming_responses_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, agent_id = _create_agent_workspace(tmp_path, monkeypatch)
    fake_client = _FakeCodexClient(_fake_codex_tool_stream())
    monkeypatch.setattr("src.harness.transport.CodexTokenManager", lambda: SimpleNamespace(get_access_token=lambda: "fresh-token"))
    monkeypatch.setattr("src.harness.transport.OpenAI", lambda **kwargs: fake_client)
    transport = CodexNativeTransport(
        run_root=str(run_root),
        agent_id=agent_id,
        model_name="codex/gpt-5.5",
        system_prompt="System v1",
    )
    tool_specs = [
        {
            "name": "read",
            "description": "Read a file",
            "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
        }
    ]

    transport.ensure_initialized("Initial prompt")
    result = transport.execute_turn(tool_specs)

    call = fake_client.responses.calls[0]
    assert call["model"] == "gpt-5.5"
    assert call["instructions"] == "System v1"
    assert call["stream"] is True
    assert call["store"] is False
    assert "previous_response_id" not in call
    assert "context_management" not in call
    assert call["tools"][0]["name"] == "read"
    assert call["input"][0] == {"role": "user", "content": "Initial prompt"}
    assert result.text_blocks == ["Need data."]
    assert result.tool_calls[0].call_id == "call_1"
    assert result.tool_calls[0].name == "read"
    assert result.tool_calls[0].arguments == {"path": "publish/final.md"}


def test_codex_native_transport_replays_tool_outputs_as_function_call_outputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_root, agent_id = _create_agent_workspace(tmp_path, monkeypatch)
    fake_client = _FakeCodexClient([SimpleNamespace(type="response.completed", response=SimpleNamespace(id="resp_2", usage=None))])
    monkeypatch.setattr("src.harness.transport.CodexTokenManager", lambda: SimpleNamespace(get_access_token=lambda: "fresh-token"))
    monkeypatch.setattr("src.harness.transport.OpenAI", lambda **kwargs: fake_client)
    transport = CodexNativeTransport(
        run_root=str(run_root),
        agent_id=agent_id,
        model_name="codex/gpt-5.5",
        system_prompt="System v1",
    )

    transport.append_user_text("Initial prompt")
    _append_model_visible_entry(
        run_root,
        agent_id,
        kind="assistant_response",
        message={
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_read",
                    "type": "function",
                    "function": {"name": "read", "arguments": '{"path":"a.md"}'},
                }
            ],
        },
        transport="codex",
    )
    transport.append_tool_results(
        [
            {
                "call_id": "call_read",
                "result": {"status": "ok", "content": "RAW"},
                "conversation_result": {"status": "ok", "description": "Read a.md"},
            }
        ]
    )
    transport.append_user_text("Continue")

    transport.execute_turn([])

    input_items = fake_client.responses.calls[0]["input"]
    assert {
        "type": "function_call",
        "call_id": "call_read",
        "name": "read",
        "arguments": '{"path": "a.md"}',
    } in input_items
    assert {
        "type": "function_call_output",
        "call_id": "call_read",
        "output": json.dumps({"status": "ok", "description": "Read a.md"}, ensure_ascii=True),
    } in input_items
    assert {"role": "user", "content": "Continue"} in input_items


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
        _append_model_visible_entry(
            run_root,
            agent_id,
            kind="assistant_response",
            message={
                "role": "assistant",
                "content": f"Assistant response {turn_number}",
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

    replay_messages = _conversation_messages(str(run_root), agent_id)
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
    assert agent_workspace_paths(run_root, agent_id)["history_summary"].read_text(encoding="utf-8") == ""


def test_preflight_history_compaction_allows_transcript_only_next_turn(
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

    def _estimate(payload: dict[str, object]) -> int:
        rendered = json.dumps(payload, ensure_ascii=True)
        assert "Pending prompt" not in rendered
        return 120_000

    monkeypatch.setattr("src.harness.transport.estimate_payload_input_tokens", _estimate)

    result = preflight_history_compaction(
        transport_name="openai",
        run_root=str(run_root),
        agent_id=agent_id,
        model_name="gpt-4o",
        system_prompt="System v1",
        pending_user_prompt=None,
        tool_specs=[],
    )

    assert not result.compaction_changed
    assert result.estimated_input_tokens_before == 120_000
    assert result.estimated_input_tokens_after == 120_000


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
        _append_model_visible_entry(
            run_root,
            agent_id,
            kind="assistant_response",
            message={
                "role": "assistant",
                "content": f"Assistant response {turn_number}",
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

    history_summary = agent_workspace_paths(run_root, agent_id)["history_summary"].read_text(encoding="utf-8")
    replay_messages = _conversation_messages(str(run_root), agent_id)
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
                            "name": "edit",
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


def test_replay_preserves_latest_unconsumed_tool_results_for_next_turn(
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
                "content": [
                    {"type": "tool_use", "id": "call_a", "name": "read", "input": {"path": "a.md"}},
                    {"type": "tool_use", "id": "call_b", "name": "read", "input": {"path": "b.md"}},
                ],
            },
        },
    )
    first_body = json.dumps({"status": "ok", "content": "A" * 1600 + "A_END"}, ensure_ascii=True)
    second_body = json.dumps({"status": "ok", "content": "B" * 1600 + "B_END"}, ensure_ascii=True)
    append_transcript_entry(
        run_root,
        agent_id,
        {
            "kind": "tool_result",
            "message": {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call_a", "content": first_body},
                    {"type": "tool_result", "tool_use_id": "call_b", "content": second_body},
                ],
            },
        },
    )
    transport.append_user_text("Turn 2")

    replay_messages = _transcript_messages(str(run_root), agent_id)
    result_message = next(
        message
        for message in replay_messages
        if isinstance(message.get("content"), list)
        and any(block.get("type") == "tool_result" for block in message["content"])
    )
    result_blocks = result_message["content"]

    assert len(result_blocks) == 2
    assert result_blocks[0]["content"] == first_body
    assert result_blocks[1]["content"] == second_body
    assert "A_END" in result_blocks[0]["content"]
    assert "B_END" in result_blocks[1]["content"]
    assert replay_messages[-1]["content"] == "Turn 2"


def test_replay_truncates_tool_result_after_model_consumes_it(
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
                "content": [
                    {"type": "tool_use", "id": "call_a", "name": "read", "input": {"path": "a.md"}},
                ],
            },
        },
    )
    full_body = json.dumps({"status": "ok", "content": "A" * 1600 + "A_END"}, ensure_ascii=True)
    append_transcript_entry(
        run_root,
        agent_id,
        {
            "kind": "tool_result",
            "message": {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "call_a", "content": full_body},
                ],
            },
        },
    )
    transport.append_user_text("Turn 2")
    append_transcript_entry(
        run_root,
        agent_id,
        {
            "kind": "assistant_response",
            "message": {"role": "assistant", "content": "Used the file content."},
        },
    )

    replay_messages = _transcript_messages(str(run_root), agent_id)
    result_message = next(
        message
        for message in replay_messages
        if isinstance(message.get("content"), list)
        and any(block.get("type") == "tool_result" for block in message["content"])
    )
    result_content = result_message["content"][0]["content"]

    assert result_content != full_body
    assert "A_END" not in result_content
    assert "... [" in result_content


def test_transcript_replay_strips_stale_budget_lines_from_historical_string_content(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Historical user message with string content has budget lines stripped."""
    run_root, agent_id = _create_agent_workspace(tmp_path, monkeypatch)

    # Turn 1: user message with budget lines (historical)
    append_transcript_entry(
        run_root, agent_id,
        {
            "kind": "user_message",
            "message": {
                "role": "user",
                "content": (
                    "# Runtime Capacity Snapshot\n"
                    "- remaining run time: ~598s\n"
                    "- remaining agent time: ~598s\n"
                    "- live agents: 0/16\n"
                ),
            },
        },
    )

    # Turn 1: assistant response (makes the above historical)
    append_transcript_entry(
        run_root, agent_id,
        {"kind": "assistant_response", "message": {"role": "assistant", "content": "Let me research XOM."}},
    )

    # Turn 2: newest user message, no model activity after it (pending)
    append_transcript_entry(
        run_root, agent_id,
        {
            "kind": "user_message",
            "message": {
                "role": "user",
                "content": (
                    "# Runtime Delta\n"
                    "## Soft-Stop Mode\n"
                    "- remaining run time: ~120s\n"
                    "- Finish your work.\n"
                ),
            },
        },
    )

    messages = _transcript_messages(str(run_root), agent_id)
    historical = messages[0]
    pending = messages[-1]

    # Historical: budget lines stripped
    assert historical["role"] == "user"
    assert isinstance(historical["content"], str)
    assert "remaining run time" not in historical["content"]
    assert "remaining agent time" not in historical["content"]
    assert "live agents: 0/16" in historical["content"]  # non-budget lines preserved

    # Pending: budget lines preserved
    assert pending["role"] == "user"
    assert isinstance(pending["content"], str)
    assert "remaining run time: ~120s" in pending["content"]


def test_transcript_replay_strips_stale_budget_lines_from_historical_list_content(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Historical user message with Anthropic-style list content has budget lines stripped."""
    run_root, agent_id = _create_agent_workspace(tmp_path, monkeypatch)

    # Turn 1: user message with budget lines in list content (Anthropic style)
    append_transcript_entry(
        run_root, agent_id,
        {
            "kind": "user_message",
            "message": {
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        "# Runtime Capacity Snapshot\n"
                        "- remaining run time: ~598s\n"
                        "- remaining agent time: ~598s\n"
                        "- live agents: 0/16\n"
                    )},
                ],
            },
        },
    )

    # Turn 1: assistant response
    append_transcript_entry(
        run_root, agent_id,
        {"kind": "assistant_response", "message": {"role": "assistant", "content": "Researching."}},
    )

    # Turn 2: pending user message (no model activity after it)
    append_transcript_entry(
        run_root, agent_id,
        {
            "kind": "user_message",
            "message": {
                "role": "user",
                "content": [{"type": "text", "text": "# Runtime Delta\n- remaining run time: ~60s\n"}],
            },
        },
    )

    messages = _transcript_messages(str(run_root), agent_id)
    historical = messages[0]
    pending = messages[-1]

    # Historical: budget lines stripped from list text block
    assert historical["role"] == "user"
    assert isinstance(historical["content"], list)
    historical_text = historical["content"][0]["text"]
    assert "remaining run time" not in historical_text
    assert "remaining agent time" not in historical_text
    assert "live agents: 0/16" in historical_text

    # Pending: budget lines preserved
    assert pending["role"] == "user"
    assert isinstance(pending["content"], list)
    pending_text = pending["content"][0]["text"]
    assert "remaining run time: ~60s" in pending_text


def test_transcript_replay_no_pending_strips_latest_user_message(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """When the latest user_message is followed by a model_request (consumed),
    it is historical, not pending, and its budget lines get stripped."""
    run_root, agent_id = _create_agent_workspace(tmp_path, monkeypatch)

    # User message with budget lines
    append_transcript_entry(
        run_root, agent_id,
        {
            "kind": "user_message",
            "message": {
                "role": "user",
                "content": (
                    "# Runtime Capacity Snapshot\n"
                    "- remaining run time: ~598s\n"
                ),
            },
        },
    )
    # Model request consumes it — user message becomes historical, not pending
    append_transcript_entry(
        run_root, agent_id,
        {"kind": "model_request", "turn_index": 1, "transport": "test"},
    )

    messages = _transcript_messages(str(run_root), agent_id)
    assert len(messages) == 1
    assert "remaining run time" not in messages[0]["content"]


def test_transcript_replay_non_user_messages_unaffected_by_strip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Assistant and tool messages are not affected by the budget-line strip."""
    run_root, agent_id = _create_agent_workspace(tmp_path, monkeypatch)

    append_transcript_entry(
        run_root, agent_id,
        {
            "kind": "user_message",
            "message": {
                "role": "user",
                "content": "# Task\n- remaining run time: ~598s\nDo analysis.",
            },
        },
    )
    append_transcript_entry(
        run_root, agent_id,
        {
            "kind": "assistant_response",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "- remaining run time: ~598s\nResearching now."}],
            },
        },
    )

    messages = _transcript_messages(str(run_root), agent_id)
    for msg in messages:
        if msg["role"] == "assistant":
            content = msg["content"]
            if isinstance(content, list):
                content = content[0]["text"]
            assert "remaining run time" in content  # assistants NOT stripped


def test_transcript_replay_first_prompt_preserves_budget_lines(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """First user message with no prior model activity is pending — budget lines preserved."""
    run_root, agent_id = _create_agent_workspace(tmp_path, monkeypatch)

    # Only one user_message, no model activity yet — this is the pending first prompt
    append_transcript_entry(
        run_root, agent_id,
        {
            "kind": "user_message",
            "message": {
                "role": "user",
                "content": (
                    "# Runtime Capacity Snapshot\n"
                    "- remaining run time: ~598s\n"
                    "- remaining agent time: ~598s\n"
                ),
            },
        },
    )

    messages = _transcript_messages(str(run_root), agent_id)
    assert len(messages) == 1
    assert "remaining run time: ~598s" in messages[0]["content"]
    assert "remaining agent time: ~598s" in messages[0]["content"]
