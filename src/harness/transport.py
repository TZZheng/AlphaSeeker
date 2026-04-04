"""MiniMax-native agent transports for the file-based harness kernel."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

import anthropic
from openai import OpenAI

from src.harness.artifacts import (
    agent_workspace_paths,
    append_transcript_entry,
    load_transcript_entries,
    load_transport_state,
    save_transport_state,
    write_json_atomic,
)


DEFAULT_MAX_TOKENS = 8192


@dataclass
class ModelToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ModelTurnResult:
    tool_calls: list[ModelToolCall]
    text_blocks: list[str]
    stop_reason: str | None


def is_minimax_model(model_name: str) -> bool:
    normalized = model_name.lower()
    return (
        normalized.startswith("minimax/")
        or normalized.startswith("minimax-")
        or normalized.startswith("codex-minimax-")
    )


def normalize_minimax_model_name(model_name: str) -> str:
    if model_name.lower().startswith("minimax/"):
        return model_name.split("/", 1)[1]
    return model_name


def minimax_openai_base_url() -> str:
    return os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1")


def minimax_anthropic_base_url() -> str:
    explicit = os.getenv("MINIMAX_ANTHROPIC_BASE_URL")
    if explicit:
        return explicit
    openai_base = minimax_openai_base_url().rstrip("/")
    if openai_base.endswith("/v1"):
        return openai_base[:-3] + "/anthropic"
    if openai_base.endswith("/anthropic"):
        return openai_base
    return openai_base + "/anthropic"


def resolve_agent_transport(requested: str, model_name: str) -> str:
    if requested != "auto":
        return requested
    if is_minimax_model(model_name):
        return "minimax_anthropic"
    return "text_json"


def _serialize_payload(payload: Any) -> Any:
    if hasattr(payload, "model_dump"):
        return payload.model_dump(mode="json")
    if isinstance(payload, dict):
        return {key: _serialize_payload(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [_serialize_payload(item) for item in payload]
    if hasattr(payload, "__dict__"):
        return {
            key: _serialize_payload(value)
            for key, value in vars(payload).items()
            if not key.startswith("_")
        }
    return payload


def _transcript_messages(run_root: str, agent_id: str) -> list[dict[str, Any]]:
    return [
        entry["message"]
        for entry in load_transcript_entries(run_root, agent_id)
        if isinstance(entry, dict) and isinstance(entry.get("message"), dict)
    ]


class BaseAgentTransport:
    transport_name = "text_json"

    def __init__(
        self,
        *,
        run_root: str,
        agent_id: str,
        model_name: str,
        system_prompt: str,
    ) -> None:
        self.run_root = run_root
        self.agent_id = agent_id
        self.model_name = model_name
        self.system_prompt = system_prompt

    def _load_meta(self) -> dict[str, Any]:
        return load_transport_state(self.run_root, self.agent_id) or {}

    def _save_meta(self, meta: dict[str, Any]) -> None:
        meta.update(
            {
                "transport": self.transport_name,
                "model_name": self.model_name,
                "system_prompt": self.system_prompt,
            }
        )
        save_transport_state(self.run_root, self.agent_id, meta)

    def _llm_turns_root(self) -> Path:
        root = agent_workspace_paths(self.run_root, self.agent_id)["scratch_root"] / "llm_turns"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _next_counter(self, key: str) -> int:
        meta = self._load_meta()
        value = int(meta.get(key, 0)) + 1
        meta[key] = value
        self._save_meta(meta)
        return value

    def _write_turn_artifact(self, *, turn_index: int, suffix: str, payload: dict[str, Any]) -> str:
        path = self._llm_turns_root() / f"{turn_index:04d}_{suffix}.json"
        write_json_atomic(path, payload)
        return str(path)

    def _append_system_prompt_snapshot(self, *, reason: str) -> None:
        version = self._next_counter("system_prompt_version")
        artifact_path = self._write_turn_artifact(
            turn_index=version,
            suffix="system_prompt",
            payload={
                "version": version,
                "reason": reason,
                "transport": self.transport_name,
                "model_name": self.model_name,
                "system_prompt": self.system_prompt,
            },
        )
        append_transcript_entry(
            self.run_root,
            self.agent_id,
            {
                "kind": "system_prompt_snapshot",
                "version": version,
                "reason": reason,
                "transport": self.transport_name,
                "model_name": self.model_name,
                "artifact_path": artifact_path,
                "system_prompt": self.system_prompt,
            },
        )

    def _record_model_request(
        self,
        *,
        api_name: str,
        request_payload: dict[str, Any],
    ) -> tuple[int, str]:
        turn_index = self._next_counter("turn_index")
        artifact_path = self._write_turn_artifact(
            turn_index=turn_index,
            suffix="request",
            payload={
                "turn_index": turn_index,
                "transport": self.transport_name,
                "api_name": api_name,
                "request": request_payload,
            },
        )
        append_transcript_entry(
            self.run_root,
            self.agent_id,
            {
                "kind": "model_request",
                "turn_index": turn_index,
                "transport": self.transport_name,
                "model_name": self.model_name,
                "api_name": api_name,
                "artifact_path": artifact_path,
                "summary": {
                    "message_count": len(request_payload.get("messages", [])),
                    "tool_count": len(request_payload.get("tools", [])),
                    "system_prompt_chars": len(str(request_payload.get("system", self.system_prompt))),
                },
            },
        )
        return turn_index, artifact_path

    def _record_model_response(
        self,
        *,
        turn_index: int,
        request_artifact_path: str,
        assistant_message: dict[str, Any],
        raw_response: Any,
        stop_reason: str | None,
        tool_calls: list[ModelToolCall],
        text_blocks: list[str],
        provider_thinking_blocks: int = 0,
    ) -> None:
        response_payload = _serialize_payload(raw_response)
        artifact_path = self._write_turn_artifact(
            turn_index=turn_index,
            suffix="response",
            payload={
                "turn_index": turn_index,
                "transport": self.transport_name,
                "model_name": self.model_name,
                "request_artifact_path": request_artifact_path,
                "assistant_message": assistant_message,
                "raw_response": response_payload,
                "decision": {
                    "stop_reason": stop_reason,
                    "tool_calls": [
                        {"call_id": item.call_id, "name": item.name, "arguments": item.arguments}
                        for item in tool_calls
                    ],
                    "text_blocks": text_blocks,
                    "provider_thinking_blocks": provider_thinking_blocks,
                },
            },
        )
        append_transcript_entry(
            self.run_root,
            self.agent_id,
            {
                "kind": "assistant_response",
                "turn_index": turn_index,
                "message": assistant_message,
                "raw_response": response_payload,
                "artifact_path": artifact_path,
                "request_artifact_path": request_artifact_path,
                "decision": {
                    "stop_reason": stop_reason,
                    "tool_calls": [
                        {"call_id": item.call_id, "name": item.name, "arguments": item.arguments}
                        for item in tool_calls
                    ],
                    "text_blocks": text_blocks,
                    "provider_thinking_blocks": provider_thinking_blocks,
                },
            },
        )

    def ensure_initialized(self, initial_user_prompt: str) -> None:
        meta = load_transport_state(self.run_root, self.agent_id)
        if meta is None:
            self._save_meta({"turn_index": 0, "system_prompt_version": 0})
        if not load_transcript_entries(self.run_root, self.agent_id):
            self._append_system_prompt_snapshot(reason="initialized")
            self.append_user_text(initial_user_prompt)

    def update_system_prompt(self, system_prompt: str) -> None:
        if system_prompt == self.system_prompt:
            return
        self.system_prompt = system_prompt
        meta = self._load_meta()
        self._save_meta(meta)
        self._append_system_prompt_snapshot(reason="updated")

    def append_user_text(self, text: str) -> None:
        raise NotImplementedError

    def execute_turn(self, tool_specs: list[dict[str, Any]]) -> ModelTurnResult:
        raise NotImplementedError

    def append_tool_results(self, tool_results: list[dict[str, Any]]) -> None:
        raise NotImplementedError


class MiniMaxAnthropicTransport(BaseAgentTransport):
    transport_name = "minimax_anthropic"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.client = anthropic.Anthropic(
            base_url=minimax_anthropic_base_url(),
            api_key=os.environ["MINIMAX_API_KEY"],
        )

    def append_user_text(self, text: str) -> None:
        message = {
            "role": "user",
            "content": [{"type": "text", "text": text}],
        }
        append_transcript_entry(
            self.run_root,
            self.agent_id,
            {"kind": "user_message", "message": message},
        )

    def execute_turn(self, tool_specs: list[dict[str, Any]]) -> ModelTurnResult:
        transcript_messages = _transcript_messages(self.run_root, self.agent_id)
        request_payload = {
            "model": normalize_minimax_model_name(self.model_name),
            "max_tokens": DEFAULT_MAX_TOKENS,
            "system": self.system_prompt,
            "messages": transcript_messages,
            "tools": tool_specs,
        }
        turn_index, request_artifact_path = self._record_model_request(
            api_name="anthropic.messages.create",
            request_payload=request_payload,
        )
        response = self.client.messages.create(
            model=request_payload["model"],
            max_tokens=request_payload["max_tokens"],
            system=request_payload["system"],
            messages=request_payload["messages"],
            tools=request_payload["tools"],
        )
        assistant_message = {
            "role": "assistant",
            "content": [_serialize_payload(block) for block in response.content],
        }

        tool_calls: list[ModelToolCall] = []
        text_blocks: list[str] = []
        thinking_blocks = 0
        for block in response.content:
            block_type = getattr(block, "type", "")
            if block_type == "tool_use":
                tool_calls.append(
                    ModelToolCall(
                        call_id=str(getattr(block, "id", "")),
                        name=str(getattr(block, "name", "")),
                        arguments=dict(getattr(block, "input", {}) or {}),
                    )
                )
            elif block_type == "text":
                text_blocks.append(str(getattr(block, "text", "")))
            elif block_type == "thinking":
                thinking_blocks += 1
        self._record_model_response(
            turn_index=turn_index,
            request_artifact_path=request_artifact_path,
            assistant_message=assistant_message,
            raw_response=response,
            stop_reason=getattr(response, "stop_reason", None),
            tool_calls=tool_calls,
            text_blocks=text_blocks,
            provider_thinking_blocks=thinking_blocks,
        )
        return ModelTurnResult(
            tool_calls=tool_calls,
            text_blocks=text_blocks,
            stop_reason=getattr(response, "stop_reason", None),
        )

    def append_tool_results(self, tool_results: list[dict[str, Any]]) -> None:
        content = [
            {
                "type": "tool_result",
                "tool_use_id": result["call_id"],
                "content": json.dumps(result["result"], ensure_ascii=True),
            }
            for result in tool_results
        ]
        message = {"role": "user", "content": content}
        append_transcript_entry(
            self.run_root,
            self.agent_id,
            {"kind": "tool_result", "message": message},
        )


class MiniMaxOpenAITransport(BaseAgentTransport):
    transport_name = "minimax_openai"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.client = OpenAI(
            base_url=minimax_openai_base_url(),
            api_key=os.environ["MINIMAX_API_KEY"],
        )

    def append_user_text(self, text: str) -> None:
        message = {"role": "user", "content": text}
        append_transcript_entry(
            self.run_root,
            self.agent_id,
            {"kind": "user_message", "message": message},
        )

    def execute_turn(self, tool_specs: list[dict[str, Any]]) -> ModelTurnResult:
        transcript_messages = [
            {"role": "system", "content": self.system_prompt},
            *_transcript_messages(self.run_root, self.agent_id),
        ]
        request_payload = {
            "model": normalize_minimax_model_name(self.model_name),
            "messages": transcript_messages,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": spec["name"],
                        "description": spec["description"],
                        "parameters": spec["input_schema"],
                    },
                }
                for spec in tool_specs
            ],
            "tool_choice": "auto",
            "max_tokens": DEFAULT_MAX_TOKENS,
            "extra_body": {"reasoning_split": True},
        }
        turn_index, request_artifact_path = self._record_model_request(
            api_name="openai.chat.completions.create",
            request_payload=request_payload,
        )
        response = self.client.chat.completions.create(
            model=request_payload["model"],
            messages=request_payload["messages"],
            tools=request_payload["tools"],
            tool_choice=request_payload["tool_choice"],
            max_tokens=request_payload["max_tokens"],
            extra_body=request_payload["extra_body"],
        )
        choice = response.choices[0]
        message_dict = _serialize_payload(choice.message)
        assistant_message = {
            "role": "assistant",
            **message_dict,
        }

        tool_calls: list[ModelToolCall] = []
        raw_tool_calls = getattr(choice.message, "tool_calls", None) or []
        for tool_call in raw_tool_calls:
            raw_arguments = getattr(tool_call.function, "arguments", "{}")
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                arguments = {}
            tool_calls.append(
                ModelToolCall(
                    call_id=str(getattr(tool_call, "id", "")),
                    name=str(getattr(tool_call.function, "name", "")),
                    arguments=arguments,
                )
            )
        content = getattr(choice.message, "content", None)
        text_blocks = [content] if isinstance(content, str) and content else []
        provider_thinking_blocks = 0
        reasoning_content = getattr(choice.message, "reasoning_content", None)
        if reasoning_content:
            provider_thinking_blocks = 1
        self._record_model_response(
            turn_index=turn_index,
            request_artifact_path=request_artifact_path,
            assistant_message=assistant_message,
            raw_response=response,
            stop_reason=getattr(choice, "finish_reason", None),
            tool_calls=tool_calls,
            text_blocks=text_blocks,
            provider_thinking_blocks=provider_thinking_blocks,
        )
        return ModelTurnResult(
            tool_calls=tool_calls,
            text_blocks=text_blocks,
            stop_reason=getattr(choice, "finish_reason", None),
        )

    def append_tool_results(self, tool_results: list[dict[str, Any]]) -> None:
        for result in tool_results:
            message = {
                "role": "tool",
                "tool_call_id": result["call_id"],
                "content": json.dumps(result["result"], ensure_ascii=True),
            }
            append_transcript_entry(
                self.run_root,
                self.agent_id,
                {"kind": "tool_result", "message": message},
            )


def create_transport(
    *,
    transport_name: str,
    run_root: str,
    agent_id: str,
    model_name: str,
    system_prompt: str,
) -> BaseAgentTransport:
    if transport_name == "minimax_anthropic":
        return MiniMaxAnthropicTransport(
            run_root=run_root,
            agent_id=agent_id,
            model_name=model_name,
            system_prompt=system_prompt,
        )
    if transport_name == "minimax_openai":
        return MiniMaxOpenAITransport(
            run_root=run_root,
            agent_id=agent_id,
            model_name=model_name,
            system_prompt=system_prompt,
        )
    raise ValueError(f"Unsupported agent transport '{transport_name}'.")
