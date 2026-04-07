"""Agent transports for the file-based harness kernel (MiniMax and Anthropic native)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
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
    write_text_atomic,
)


DEFAULT_MAX_TOKENS = 8192

_RATE_LIMIT_INITIAL_DELAY = 5.0
_RATE_LIMIT_MAX_DELAY = 60.0
_RATE_LIMIT_MAX_RETRIES = 6


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


def is_anthropic_model(model_name: str) -> bool:
    return model_name.lower().startswith("claude-")


def is_openai_model(model_name: str) -> bool:
    normalized = model_name.lower()
    return normalized.startswith("gpt-") or normalized.startswith("o1") or normalized.startswith("o3") or normalized.startswith("o4")


def resolve_agent_transport(requested: str, model_name: str) -> str:
    if requested != "auto":
        return requested
    if is_minimax_model(model_name):
        return "minimax_anthropic"
    if is_anthropic_model(model_name):
        return "anthropic"
    if is_openai_model(model_name):
        return "openai"
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

    def _write_thinking_block(self, *, turn_index: int, content: list[str]) -> None:
        """Write thinking block content to a human-readable text file for live observability."""
        root = self._llm_turns_root()
        # Per-turn thinking file: scratch/llm_turns/0001_thinking.txt
        turn_path = root / f"{turn_index:04d}_thinking.txt"
        text = "\n\n---\n\n".join(content) if content else ""
        write_text_atomic(turn_path, text)
        # Latest thinking symlink so you can always `cat` the most recent one
        latest_path = root / "thinking_current.txt"
        write_text_atomic(latest_path, text)

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
                "created_at": datetime.now(timezone.utc).isoformat(),
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
        # Initialize file paths before streaming
        thinking_stream_path = self._llm_turns_root() / f"{turn_index:04d}_thinking.txt"
        thinking_stream_path.write_text("", encoding="utf-8")
        latest_thinking_path = self._llm_turns_root() / "thinking_current.txt"
        latest_thinking_path.write_text("", encoding="utf-8")

        import sys as _sys

        _rate_limit_delay = _RATE_LIMIT_INITIAL_DELAY
        for _attempt in range(_RATE_LIMIT_MAX_RETRIES + 1):
            tool_calls: list[ModelToolCall] = []
            text_blocks: list[str] = []
            thinking_blocks = 0
            thinking_content: list[str] = []

            # Per-thought-chunk entries: (iso_timestamp, block_index, thinking_text)
            thinking_log_lines: list[str] = []
            block_index = 0
            current_tool_call_id: str | None = None
            current_tool_name: str | None = None
            current_tool_input: dict[str, Any] = {}
            current_text: str = ""
            current_thinking: str = ""
            stop_reason: str | None = None
            _saw_standalone_thinking = False  # set True when full thinking arrives as standalone event

            try:
                # Use streaming with context manager (SDK 0.84.0 requires `with` for MessageStreamManager)
                with self.client.messages.stream(
                    model=request_payload["model"],
                    max_tokens=request_payload["max_tokens"],
                    system=request_payload["system"],
                    messages=request_payload["messages"],
                    tools=request_payload["tools"],
                ) as stream:
                    for event in stream:
                        event_type = getattr(event, "type", "")
                        # Debug: print unknown event types to stderr
                        if event_type not in (
                            "content_block_start",
                            "content_block_delta",
                            "content_block_stop",
                            "message_delta",
                            "message_stop",
                            "thinking",
                            "text",
                            "input_json",
                            "signature",
                            "message_start",
                        ):
                            _sys.stderr.write(f"[STREAM DEBUG] unknown event_type={event_type!r} attrs={list(vars(event).keys())}\n")
                            _sys.stderr.flush()
                        if event_type == "content_block_start":
                            block_name = getattr(event, "name", "") or getattr(event, "type", "")
                            if block_name == "thinking":
                                thinking_blocks += 1
                                block_index += 1
                                current_thinking = ""
                                _saw_standalone_thinking = False  # reset per thinking block
                                thinking_log_lines.append(
                                    f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] thinking_block #{block_index} START"
                                )
                            elif block_name == "tool_use":
                                block_index += 1
                                current_tool_call_id = str(getattr(event, "id", ""))
                                current_tool_name = str(getattr(event.input, "name", "tool_use"))
                                current_tool_input = {}
                            elif block_name == "text":
                                block_index += 1
                                current_text = ""

                        elif event_type == "thinking":
                            # Standalone thinking block event: MiniMax sends the full thinking
                            # content as its own event type (separate from content_block_delta chunks).
                            raw_thinking = getattr(event, "thinking", None)
                            if raw_thinking:
                                current_thinking = str(raw_thinking)
                                _saw_standalone_thinking = True
                                thinking_log_lines.append(
                                    f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] thinking_block #{block_index} FULL standalone ({len(current_thinking)} chars)"
                                )

                        elif event_type == "content_block_delta":
                            delta = getattr(event, "delta", None)
                            if delta is None:
                                continue
                            delta_type = getattr(delta, "type", "")
                            if delta_type == "thinking":
                                chunk = str(getattr(delta, "thinking", ""))
                                if not _saw_standalone_thinking:
                                    current_thinking += chunk
                                thinking_log_lines.append(
                                    f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] thinking_block #{block_index} chunk ({len(chunk)} chars)"
                                )
                                # Write incrementally so `tail -f` shows live progress
                                thinking_stream_path.write_text(
                                    "\n".join(thinking_log_lines) + f"\n\n--- thinking_block #{block_index} accumulated ---\n{current_thinking}\n",
                                    encoding="utf-8",
                                )
                                latest_thinking_path.write_text(
                                    f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] LIVE thinking_block #{block_index} ({len(current_thinking)} chars so far)\n{current_thinking}",
                                    encoding="utf-8",
                                )
                            elif delta_type == "text_delta":
                                chunk = str(getattr(delta, "text", ""))
                                current_text += chunk
                            elif delta_type == "input_json_delta":
                                # Accumulate tool input as chunks arrive
                                chunk = str(getattr(delta, "input_json", ""))
                                current_tool_input_str = current_tool_input.get("_raw", "") + chunk
                                current_tool_input["_raw"] = current_tool_input_str

                        elif event_type == "content_block_stop":
                            block_name = getattr(event, "name", "") or getattr(event, "type", "")
                            if block_name == "thinking":
                                # Only append if we didn't already capture via standalone thinking event
                                thinking_content.append(current_thinking)
                                thinking_log_lines.append(
                                    f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] thinking_block #{block_index} END ({len(current_thinking)} total chars)"
                                )
                            elif block_name == "text":
                                text_blocks.append(current_text)
                            elif block_name == "tool_use":
                                # Parse accumulated tool input
                                try:
                                    tool_input = json.loads(current_tool_input.get("_raw", "{}"))
                                except json.JSONDecodeError:
                                    tool_input = {}
                                if current_tool_call_id and current_tool_name:
                                    tool_calls.append(
                                        ModelToolCall(
                                            call_id=current_tool_call_id,
                                            name=current_tool_name,
                                            arguments=tool_input,
                                        )
                                    )

                        elif event_type == "message_delta":
                            stop_reason = getattr(getattr(event, "delta", None), "stop_reason", None)

                        elif event_type == "message_stop":
                            stop_reason = getattr(event, "stop_reason", None)

                        # MiniMax API sends these as standalone events (not content_block_delta):
                        elif event_type == "thinking":
                            # Standalone thinking content event
                            raw_thinking = getattr(event, "thinking", None)
                            if raw_thinking:
                                current_thinking = str(raw_thinking)
                                thinking_log_lines.append(
                                    f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] thinking FULL ({len(current_thinking)} chars)"
                                )
                                # Write incrementally
                                thinking_stream_path.write_text(
                                    "\n".join(thinking_log_lines) + f"\n\n--- accumulated thinking ({len(current_thinking)} chars) ---\n{current_thinking}\n",
                                    encoding="utf-8",
                                )
                                latest_thinking_path.write_text(
                                    f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] LIVE thinking ({len(current_thinking)} chars)\n{current_thinking}",
                                    encoding="utf-8",
                                )

                        elif event_type == "text":
                            # Standalone text content event
                            raw_text = getattr(event, "text", None)
                            if raw_text:
                                current_text += str(raw_text)

                        elif event_type == "input_json":
                            # Tool input streaming event
                            partial = getattr(event, "partial_json", None)
                            if partial:
                                current_tool_input_str = current_tool_input.get("_raw", "") + str(partial)
                                current_tool_input["_raw"] = current_tool_input_str

                        elif event_type == "signature":
                            # Signature event - might be related to thinking block end
                            # No action needed for now
                            pass

                        elif event_type == "message_start":
                            # Message start event - no content action needed
                            pass

                # Get the final accumulated message from the stream
                final_message = stream.get_final_message()

                # Extract thinking blocks and tool_use blocks from the final message content
                # The final_message.content contains all content blocks including thinking and tool_use
                # Reset thinking_blocks since streaming loop already counted them
                thinking_blocks = 0
                for block in final_message.content:
                    block_type = getattr(block, "type", None)
                    if block_type == "thinking":
                        thinking_blocks += 1
                        thinking_content.append(str(block.thinking))
                        thinking_log_lines.append(
                            f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] FINAL extracted thinking_block #{thinking_blocks} ({len(str(block.thinking))} chars)"
                        )
                    elif block_type == "tool_use":
                        # Extract tool_use blocks from the final message
                        tool_calls.append(
                            ModelToolCall(
                                call_id=str(getattr(block, "id", "")),
                                name=str(getattr(block, "name", "")),
                                arguments=dict(getattr(block, "input", {})),
                            )
                        )
                    elif block_type == "text":
                        text_blocks.append(str(getattr(block, "text", "")))
                break  # success, exit retry loop
            except anthropic.RateLimitError:
                if _attempt >= _RATE_LIMIT_MAX_RETRIES:
                    raise
                sys.stderr.write(
                    f"[transport] rate-limited (429), retrying in {_rate_limit_delay:.0f}s"
                    f" (attempt {_attempt + 1}/{_RATE_LIMIT_MAX_RETRIES})\n"
                )
                sys.stderr.flush()
                time.sleep(_rate_limit_delay)
                _rate_limit_delay = min(_rate_limit_delay * 2, _RATE_LIMIT_MAX_DELAY)

        # Write final thinking log (outside the with block)
        thinking_stream_path.write_text(
            "\n".join(thinking_log_lines)
            + f"\n\n--- FINAL thinking blocks ({thinking_blocks} total) ---\n"
            + "\n\n".join(f"=== thinking_block #{i+1} ===\n{ct}" for i, ct in enumerate(thinking_content)),
            encoding="utf-8",
        )
        latest_thinking_path.write_text(
            f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] DONE\n"
            + "\n".join(f"=== thinking_block #{i+1} ===\n{ct}" for i, ct in enumerate(thinking_content)),
            encoding="utf-8",
        )

        assistant_message = {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": tc} for tc in thinking_content
            ]
            + ([{"type": "text", "text": tb} for tb in text_blocks if tb])
            + [
                {"type": "tool_use", "id": tc.call_id, "name": tc.name, "input": tc.arguments}
                for tc in tool_calls
            ],
        }
        self._record_model_response(
            turn_index=turn_index,
            request_artifact_path=request_artifact_path,
            assistant_message=assistant_message,
            raw_response=final_message,
            stop_reason=stop_reason,
            tool_calls=tool_calls,
            text_blocks=text_blocks,
            provider_thinking_blocks=thinking_blocks,
        )
        return ModelTurnResult(
            tool_calls=tool_calls,
            text_blocks=text_blocks,
            stop_reason=stop_reason,
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


class AnthropicNativeTransport(BaseAgentTransport):
    """Native Anthropic transport using the standard Anthropic SDK and ANTHROPIC_API_KEY."""

    transport_name = "anthropic"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.client = anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"],
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
            "model": self.model_name,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "system": self.system_prompt,
            "messages": transcript_messages,
            "tools": tool_specs,
        }
        turn_index, request_artifact_path = self._record_model_request(
            api_name="anthropic.messages.create",
            request_payload=request_payload,
        )
        thinking_stream_path = self._llm_turns_root() / f"{turn_index:04d}_thinking.txt"
        thinking_stream_path.write_text("", encoding="utf-8")
        latest_thinking_path = self._llm_turns_root() / "thinking_current.txt"
        latest_thinking_path.write_text("", encoding="utf-8")

        _rate_limit_delay = _RATE_LIMIT_INITIAL_DELAY
        for _attempt in range(_RATE_LIMIT_MAX_RETRIES + 1):
            tool_calls: list[ModelToolCall] = []
            text_blocks: list[str] = []
            thinking_content: list[str] = []
            thinking_blocks = 0
            thinking_log_lines: list[str] = []
            stop_reason: str | None = None

            # Track which block type is open at each index, since content_block_stop
            # only carries the index — unlike MiniMax which carries the block name.
            block_type_by_index: dict[int, str] = {}
            current_tool_call_id: str | None = None
            current_tool_name: str | None = None
            current_tool_input_raw: str = ""
            current_text: str = ""
            current_thinking: str = ""

            try:
                with self.client.messages.stream(
                    model=request_payload["model"],
                    max_tokens=request_payload["max_tokens"],
                    system=request_payload["system"],
                    messages=request_payload["messages"],
                    tools=request_payload["tools"],
                ) as stream:
                    for event in stream:
                        event_type = getattr(event, "type", "")

                        if event_type == "content_block_start":
                            cb = getattr(event, "content_block", None)
                            cb_type = getattr(cb, "type", "")
                            block_idx = getattr(event, "index", -1)
                            block_type_by_index[block_idx] = cb_type

                            if cb_type == "thinking":
                                thinking_blocks += 1
                                current_thinking = ""
                                thinking_log_lines.append(
                                    f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] thinking_block #{thinking_blocks} START"
                                )
                            elif cb_type == "tool_use":
                                current_tool_call_id = str(getattr(cb, "id", ""))
                                current_tool_name = str(getattr(cb, "name", ""))
                                current_tool_input_raw = ""
                            elif cb_type == "text":
                                current_text = ""

                        elif event_type == "content_block_delta":
                            delta = getattr(event, "delta", None)
                            if delta is None:
                                continue
                            delta_type = getattr(delta, "type", "")

                            if delta_type == "thinking_delta":
                                # Real Anthropic uses "thinking_delta" (MiniMax uses "thinking")
                                chunk = str(getattr(delta, "thinking", ""))
                                current_thinking += chunk
                                thinking_log_lines.append(
                                    f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] thinking_block #{thinking_blocks} chunk ({len(chunk)} chars)"
                                )
                                thinking_stream_path.write_text(
                                    "\n".join(thinking_log_lines)
                                    + f"\n\n--- thinking_block #{thinking_blocks} accumulated ---\n{current_thinking}\n",
                                    encoding="utf-8",
                                )
                                latest_thinking_path.write_text(
                                    f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] LIVE thinking_block #{thinking_blocks} ({len(current_thinking)} chars so far)\n{current_thinking}",
                                    encoding="utf-8",
                                )
                            elif delta_type == "text_delta":
                                current_text += str(getattr(delta, "text", ""))
                            elif delta_type == "input_json_delta":
                                # Real Anthropic uses "partial_json" (MiniMax uses "input_json")
                                current_tool_input_raw += str(getattr(delta, "partial_json", ""))

                        elif event_type == "content_block_stop":
                            block_idx = getattr(event, "index", -1)
                            cb_type = block_type_by_index.get(block_idx, "")

                            if cb_type == "thinking":
                                thinking_content.append(current_thinking)
                                thinking_log_lines.append(
                                    f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] thinking_block #{thinking_blocks} END ({len(current_thinking)} total chars)"
                                )
                            elif cb_type == "text":
                                if current_text:
                                    text_blocks.append(current_text)
                            elif cb_type == "tool_use":
                                try:
                                    tool_input = json.loads(current_tool_input_raw or "{}")
                                except json.JSONDecodeError:
                                    tool_input = {}
                                if current_tool_call_id and current_tool_name:
                                    tool_calls.append(
                                        ModelToolCall(
                                            call_id=current_tool_call_id,
                                            name=current_tool_name,
                                            arguments=tool_input,
                                        )
                                    )

                        elif event_type == "message_delta":
                            stop_reason = getattr(getattr(event, "delta", None), "stop_reason", None)

                # Authoritative extraction from the final assembled message.
                # Overrides streaming-accumulated values to avoid partial/duplicate entries.
                final_message = stream.get_final_message()
                tool_calls = []
                text_blocks = []
                thinking_content = []
                thinking_blocks = 0
                for block in final_message.content:
                    block_type = getattr(block, "type", None)
                    if block_type == "thinking":
                        thinking_blocks += 1
                        thinking_content.append(str(getattr(block, "thinking", "")))
                        thinking_log_lines.append(
                            f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] FINAL extracted thinking_block #{thinking_blocks} ({len(str(getattr(block, 'thinking', '')))} chars)"
                        )
                    elif block_type == "tool_use":
                        tool_calls.append(
                            ModelToolCall(
                                call_id=str(getattr(block, "id", "")),
                                name=str(getattr(block, "name", "")),
                                arguments=dict(getattr(block, "input", {})),
                            )
                        )
                    elif block_type == "text":
                        text_blocks.append(str(getattr(block, "text", "")))
                break  # success, exit retry loop
            except anthropic.RateLimitError:
                if _attempt >= _RATE_LIMIT_MAX_RETRIES:
                    raise
                sys.stderr.write(
                    f"[transport] rate-limited (429), retrying in {_rate_limit_delay:.0f}s"
                    f" (attempt {_attempt + 1}/{_RATE_LIMIT_MAX_RETRIES})\n"
                )
                sys.stderr.flush()
                time.sleep(_rate_limit_delay)
                _rate_limit_delay = min(_rate_limit_delay * 2, _RATE_LIMIT_MAX_DELAY)

        thinking_stream_path.write_text(
            "\n".join(thinking_log_lines)
            + f"\n\n--- FINAL thinking blocks ({thinking_blocks} total) ---\n"
            + "\n\n".join(f"=== thinking_block #{i + 1} ===\n{ct}" for i, ct in enumerate(thinking_content)),
            encoding="utf-8",
        )
        latest_thinking_path.write_text(
            f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ')}] DONE\n"
            + "\n".join(f"=== thinking_block #{i + 1} ===\n{ct}" for i, ct in enumerate(thinking_content)),
            encoding="utf-8",
        )

        assistant_message = {
            "role": "assistant",
            "content": [{"type": "thinking", "thinking": tc} for tc in thinking_content]
            + ([{"type": "text", "text": tb} for tb in text_blocks if tb])
            + [
                {"type": "tool_use", "id": tc.call_id, "name": tc.name, "input": tc.arguments}
                for tc in tool_calls
            ],
        }
        self._record_model_response(
            turn_index=turn_index,
            request_artifact_path=request_artifact_path,
            assistant_message=assistant_message,
            raw_response=final_message,
            stop_reason=stop_reason,
            tool_calls=tool_calls,
            text_blocks=text_blocks,
            provider_thinking_blocks=thinking_blocks,
        )
        return ModelTurnResult(
            tool_calls=tool_calls,
            text_blocks=text_blocks,
            stop_reason=stop_reason,
        )

    def append_tool_results(self, tool_results: list[dict[str, Any]]) -> None:
        # Anthropic format: tool_result blocks with tool_use_id
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


class OpenAINativeTransport(BaseAgentTransport):
    """Native OpenAI transport using the standard OpenAI SDK and OPENAI_API_KEY."""

    transport_name = "openai"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @staticmethod
    def _to_openai_tools(tool_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": spec["name"],
                    "description": spec["description"],
                    "parameters": spec["input_schema"],
                },
            }
            for spec in tool_specs
        ]

    def append_user_text(self, text: str) -> None:
        # OpenAI format: plain string content, not a list of content blocks
        message = {"role": "user", "content": text}
        append_transcript_entry(
            self.run_root,
            self.agent_id,
            {"kind": "user_message", "message": message},
        )

    def execute_turn(self, tool_specs: list[dict[str, Any]]) -> ModelTurnResult:
        # System prompt goes as the first message in OpenAI format
        transcript_messages = [
            {"role": "system", "content": self.system_prompt},
            *_transcript_messages(self.run_root, self.agent_id),
        ]
        openai_tools = self._to_openai_tools(tool_specs)
        request_payload = {
            "model": self.model_name,
            "messages": transcript_messages,
            "tools": openai_tools,
            "tool_choice": "auto",
            "max_tokens": DEFAULT_MAX_TOKENS,
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
        )
        choice = response.choices[0]

        tool_calls: list[ModelToolCall] = []
        for tc in getattr(choice.message, "tool_calls", None) or []:
            try:
                arguments = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                arguments = {}
            tool_calls.append(
                ModelToolCall(
                    call_id=str(tc.id),
                    name=str(tc.function.name),
                    arguments=arguments,
                )
            )

        content = getattr(choice.message, "content", None)
        text_blocks = [content] if isinstance(content, str) and content else []

        # Build assistant message in OpenAI format so it replays correctly
        assistant_message: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": tc.call_id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments, ensure_ascii=True),
                    },
                }
                for tc in tool_calls
            ]

        self._record_model_response(
            turn_index=turn_index,
            request_artifact_path=request_artifact_path,
            assistant_message=assistant_message,
            raw_response=response,
            stop_reason=getattr(choice, "finish_reason", None),
            tool_calls=tool_calls,
            text_blocks=text_blocks,
        )
        return ModelTurnResult(
            tool_calls=tool_calls,
            text_blocks=text_blocks,
            stop_reason=getattr(choice, "finish_reason", None),
        )

    def append_tool_results(self, tool_results: list[dict[str, Any]]) -> None:
        # OpenAI format: one role=tool message per result with tool_call_id
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
    if transport_name == "anthropic":
        return AnthropicNativeTransport(
            run_root=run_root,
            agent_id=agent_id,
            model_name=model_name,
            system_prompt=system_prompt,
        )
    if transport_name == "openai":
        return OpenAINativeTransport(
            run_root=run_root,
            agent_id=agent_id,
            model_name=model_name,
            system_prompt=system_prompt,
        )
    raise ValueError(f"Unsupported agent transport '{transport_name}'.")
