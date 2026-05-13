"""
LLM Manager — centralized model registry for AlphaSeeker.

Provides a single `get_llm(model_name)` function that returns a
configured LangChain ChatModel instance. Models are lazy-initialized
and cached after first use.

To add a new provider, add an ``elif`` branch in ``_build_model``.

Usage::

    from src.shared.llm_manager import get_llm

    llm = get_llm("gemini-3-flash-preview")   # extraction tasks
    llm = get_llm("kimi-k2.5")                # writing tasks
    llm = get_llm("minimax/MiniMax-M2.5")     # MiniMax via OpenAI-compatible API
    llm = get_llm("codex/gpt-5.5")             # Native ChatGPT Codex Responses backend
"""

import os
from types import SimpleNamespace
from typing import Any, Callable, Dict, Protocol, cast
from pydantic import SecretStr

from openai import OpenAI

from src.shared.codex_auth import CodexTokenManager

# ---------------------------------------------------------------------------
# Rate Limit Handling (Wait & Alert)
# ---------------------------------------------------------------------------
from tenacity import (
    stop_after_attempt,
    wait_exponential,
)
import logging
import google.api_core.exceptions

# Configure logger for rate limit alerts
logger = logging.getLogger(__name__)

# Fallback chain for Gemini models
FALLBACK_CHAIN = [
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-exp-1206",
]

def _is_rate_limit_error(exception):
    """Check if exception is a rate limit error (429)."""
    if isinstance(exception, google.api_core.exceptions.ResourceExhausted):
        return True
        
    # Fallback: Check string content for ANY exception type
    # (LangChain might wrap it in different ways)
    msg = str(exception)
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
        return True
        
    return False

def _is_rpd_error(exception):
    """Check if the rate limit error is due to Requests Per Day (RPD)."""
    # Extract details from exception if possible
    msg = str(exception).lower()
    if "day" in msg:
        return True
        
    # Check inner exception cause if present
    if hasattr(exception, "__cause__") and exception.__cause__:
        inner_msg = str(exception.__cause__).lower()
        if "day" in inner_msg:
            return True
            
    return False

def _log_rate_limit(retry_state):
    """Alert user when rate limit is hit."""
    print(f"⚠️ RATE LIMIT HIT: Waiting {retry_state.next_action.sleep}s before retry...", flush=True)


def _secret_from_env(var_name: str) -> SecretStr | None:
    value = os.getenv(var_name)
    return SecretStr(value) if value else None


def _is_minimax_model(name: str) -> bool:
    normalized = name.lower()
    return (
        normalized.startswith("minimax/")
        or normalized.startswith("minimax-")
        or normalized.startswith("codex-minimax-")
    )


def _normalize_minimax_model_name(name: str) -> str:
    if name.lower().startswith("minimax/"):
        return name.split("/", 1)[1]
    return name


def _minimax_base_url() -> str:
    return os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1")


CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex"


def _is_codex_model(name: str) -> bool:
    return name.lower().startswith("codex/")


def _normalize_codex_model_name(name: str) -> str:
    if name.lower().startswith("codex/"):
        return name.split("/", 1)[1]
    return name


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
            else:
                text = getattr(item, "text", None) or getattr(item, "content", None)
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part)
    if content is None:
        return ""
    return str(content)


def _codex_messages_to_responses_input(messages: Any) -> tuple[str, list[dict[str, Any]]]:
    if isinstance(messages, str):
        return "", [{"role": "user", "content": messages}]

    instructions: list[str] = []
    input_items: list[dict[str, Any]] = []
    for message in messages:
        role = getattr(message, "type", None) or getattr(message, "role", None)
        if isinstance(message, dict):
            role = message.get("type") or message.get("role") or role
            content = message.get("content")
            tool_call_id = message.get("tool_call_id")
        else:
            content = getattr(message, "content", "")
            tool_call_id = getattr(message, "tool_call_id", None)

        text = _message_content_to_text(content)
        if role in {"system", "developer"}:
            if text:
                instructions.append(text)
        elif role in {"human", "user"}:
            input_items.append({"role": "user", "content": text})
        elif role in {"ai", "assistant"}:
            input_items.append({"role": "assistant", "content": text})
        elif role == "tool":
            input_items.append({
                "type": "function_call_output",
                "call_id": str(tool_call_id or ""),
                "output": text,
            })
        elif text:
            input_items.append({"role": "user", "content": text})

    return "\n\n".join(instructions), input_items


def _completed_response_text(response: Any) -> str:
    output = getattr(response, "output", None)
    if isinstance(output, list):
        parts: list[str] = []
        for item in output:
            content = getattr(item, "content", None)
            if isinstance(content, list):
                for block in content:
                    text = getattr(block, "text", None)
                    if isinstance(text, str):
                        parts.append(text)
            text = getattr(item, "text", None)
            if isinstance(text, str):
                parts.append(text)
        if parts:
            return "\n".join(parts)
    output_text = getattr(response, "output_text", None)
    return output_text if isinstance(output_text, str) else ""


class CodexNativeChatModel:
    """Minimal invoke-compatible native Codex model for non-agent LLM calls.

    Used by condense/summarization code paths that go through get_llm(), while
    harness agents use CodexNativeTransport directly. This uses the same
    ChatGPT Codex Responses backend and OAuth token file; it does not use the
    OpenAI API-key path or Codex CLI.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.token_manager = CodexTokenManager()
        self.client = OpenAI(
            api_key=self.token_manager.get_access_token(),
            base_url=CODEX_BASE_URL,
        )

    def invoke(self, messages: Any, **_: Any) -> Any:
        instructions, input_items = _codex_messages_to_responses_input(messages)
        self.client.api_key = self.token_manager.get_access_token()
        payload: dict[str, Any] = {
            "model": _normalize_codex_model_name(self.model_name),
            "input": input_items,
            "instructions": instructions or "You are a helpful text-processing assistant.",
            "stream": True,
            "store": False,
        }

        text_parts: list[str] = []
        completed_response: Any = None
        for event in self.client.responses.create(**payload):
            event_type = getattr(event, "type", "")
            if event_type == "response.output_text.delta":
                delta = getattr(event, "delta", "")
                if delta:
                    text_parts.append(str(delta))
            elif event_type == "response.completed":
                completed_response = getattr(event, "response", None)

        text = "".join(text_parts)
        if not text and completed_response is not None:
            text = _completed_response_text(completed_response)
        return SimpleNamespace(content=text)

    def stream(self, messages: Any, **kwargs: Any) -> Any:
        yield self.invoke(messages, **kwargs)


def _normalize_structured_output_kwargs_for_model(
    model_name: str,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    normalized = dict(kwargs)
    if _is_minimax_model(model_name) and normalized.get("method") == "json_mode":
        # MiniMax-M2.7 supports OpenAI-compatible tool calling, but its documented
        # native JSON schema support is not for the reasoning model family.
        normalized["method"] = "function_calling"
    return normalized


class SupportsModelOps(Protocol):
    """Minimal interface used by RateLimitWrapper to execute or bind models."""

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        ...


def _bind_model_method(
    model: SupportsModelOps,
    method_name: str,
    *args: Any,
    **kwargs: Any,
) -> SupportsModelOps:
    method = getattr(model, method_name, None)
    if not callable(method):
        raise TypeError(f"Model does not support '{method_name}' binding")
    return cast(SupportsModelOps, method(*args, **kwargs))

class RateLimitWrapper:
    """
    Wrapper for LLM to handle 429 errors with wait-and-alert logic and model fallback.
    Uses a Factory Pattern to allow reconstructing the model chain (including 
    structured output bindings) when switching underlying models.
    """
    def __init__(self, model_factory: Callable[[str], SupportsModelOps], current_model_name: str):
        self.model_factory = model_factory
        self.current_model_name = current_model_name
        self._instance: SupportsModelOps | None = None

    @property
    def model(self):
        """Lazy-load or return cached instance."""
        if self._instance is None:
            self._instance = self.model_factory(self.current_model_name)
        return self._instance

    def _switch_model(self):
        """Switch to the next model in the fallback chain."""
        try:
            current_idx = FALLBACK_CHAIN.index(self.current_model_name)
            next_idx = current_idx + 1
            if next_idx < len(FALLBACK_CHAIN):
                new_model_name = FALLBACK_CHAIN[next_idx]
                print(f"⚠️ RPD Limit exceeded on {self.current_model_name}. Switching to backup: {new_model_name}...", flush=True)
                
                self.current_model_name = new_model_name
                self._instance = None # Invalidate cache to force rebuild with new model
                return True
            else:
                print(f"❌ All models in fallback chain exhausted.", flush=True)
                return False
        except ValueError:
            # Current model not in chain? generic fallback
            return False

    def _check_retry(self, retry_state):
        """Custom retry predicate that handles model switching side-effect."""
        exc = retry_state.outcome.exception()
        
        if not _is_rate_limit_error(exc):
            return False
        
        # Check if it's an RPD error
        if _is_rpd_error(exc):
            if self._switch_model():
                return True
            else:
                return False 
        
        return True

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        return self._retry_invoke(*args, **kwargs)
    
    def stream(self, *args: Any, **kwargs: Any) -> Any:
        return self._retry_stream(*args, **kwargs)

    def _retry_invoke(self, *args: Any, **kwargs: Any) -> Any:
        from tenacity import Retrying
        for attempt in Retrying(
            retry=self._check_retry,
            wait=wait_exponential(multiplier=1, min=4, max=60),
            stop=stop_after_attempt(10),
            before_sleep=_log_rate_limit,
            reraise=True
        ):
            with attempt:
                result = self.model.invoke(*args, **kwargs)
                _notify_observer(self.current_model_name, args, result)
                return result
        raise RuntimeError("Retry loop exited without invoking model")

    def _retry_stream(self, *args: Any, **kwargs: Any) -> Any:
        from tenacity import Retrying
        for attempt in Retrying(
            retry=self._check_retry,
            wait=wait_exponential(multiplier=1, min=4, max=60),
            stop=stop_after_attempt(10),
            before_sleep=_log_rate_limit,
            reraise=True
        ):
            with attempt:
                result = self.model.stream(*args, **kwargs)
                return result
        raise RuntimeError("Retry loop exited without invoking model")

    def with_structured_output(self, *args, **kwargs):
        """
        Bind structured output configuration to a NEW factory.
        This ensures the binding is re-applied if we switch models.
        """
        # print(f"DEBUG: RateLimitWrapper.with_structured_output factory creation", flush=True)
        def new_factory(name: str) -> SupportsModelOps:
            base = self.model_factory(name)
            normalized_kwargs = _normalize_structured_output_kwargs_for_model(name, kwargs)
            return _bind_model_method(base, "with_structured_output", *args, **normalized_kwargs)
        return RateLimitWrapper(new_factory, self.current_model_name)

    def bind_tools(self, *args, **kwargs):
        """Bind tools to a NEW factory."""
        # print(f"DEBUG: RateLimitWrapper.bind_tools factory creation", flush=True)
        def new_factory(name: str) -> SupportsModelOps:
            base = self.model_factory(name)
            return _bind_model_method(base, "bind_tools", *args, **kwargs)
        return RateLimitWrapper(new_factory, self.current_model_name)

    def __getattr__(self, name):
        """Proxy other attribute access to the underlying model."""
        return getattr(self.model, name)



# ---------------------------------------------------------------------------
# Global LLM observer hook (for TUI logging / thinking capture)
# ---------------------------------------------------------------------------
_llm_observer: Callable[[str, Any, Any, Any], None] | None = None
"""Optional global callback invoked on every LLM invoke call.

Args:
    model_name: The model identifier string.
    prompt: The input prompt/messages passed to the model.
    response: The raw model response object.
    thinking: Any thinking block content if available (None otherwise).
"""


def set_llm_observer(callback: Callable[[str, Any, Any, Any], None] | None) -> None:
    """Register or clear the global LLM observer callback."""
    global _llm_observer
    _llm_observer = callback


def _notify_observer(model_name: str, prompt: Any, response: Any, thinking: Any = None) -> None:
    if _llm_observer is not None:
        try:
            _llm_observer(model_name, prompt, response, thinking)
        except Exception:
            pass  # Observer must not affect LLM calls


# ---------------------------------------------------------------------------
# Registry: model name → configured ChatModel instance (lazy-initialized)
# ---------------------------------------------------------------------------
_registry: Dict[str, RateLimitWrapper] = {}


def _build_model(model_name: str) -> RateLimitWrapper:
    """Construct a ChatModel for a given model name.

    Args:
        model_name: The model identifier string (e.g. ``"kimi-k2.5"``).

    Returns:
        A configured model wrapped in ``RateLimitWrapper``.

    Raises:
        ValueError: If the model name is not recognized.
    """
    def _factory(name: str) -> SupportsModelOps:
        if name.startswith("kimi-"):
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=name,
                temperature=1,
                base_url="https://api.moonshot.ai/v1",
                api_key=_secret_from_env("KIMI_API_KEY"),
                max_retries=2,
            )
        elif _is_codex_model(name):
            return CodexNativeChatModel(name)
        elif name.startswith("gpt-") or name.startswith("o1") or name.startswith("o3") or name.startswith("o4"):
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=name,
                temperature=0.3,
                api_key=_secret_from_env("OPENAI_API_KEY"),
                max_retries=2,
            )
        elif name.startswith("claude-"):
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model_name=name,
                timeout=None,
                stop=None,
                temperature=0.3,
                api_key=_secret_from_env("ANTHROPIC_API_KEY") or SecretStr(""),
                max_retries=2,
            )
        elif name.startswith("sf/"):
            from langchain_openai import ChatOpenAI
            # Strip the "sf/" prefix to get the SiliconFlow model ID
            sf_model = name[3:]
            return ChatOpenAI(
                model=sf_model,
                temperature=0.3,
                base_url="https://api.siliconflow.cn/v1",
                api_key=_secret_from_env("SILICONFLOW_API_KEY"),
                max_retries=2,
            )
        elif _is_minimax_model(name):
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=_normalize_minimax_model_name(name),
                temperature=0.3,
                base_url=_minimax_base_url(),
                api_key=_secret_from_env("MINIMAX_API_KEY"),
                max_retries=2,
            )
        elif name.startswith("gemini-"):
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=name,
                temperature=0.3,
                max_retries=1,  # Fail fast internally so wrapper catches it
                google_api_key=os.getenv("GOOGLE_API_KEY"),
            )
        else:
            raise ValueError(
                f"Unknown model '{name}'. "
                "Add a builder branch in src/shared/llm_manager.py."
            )
            
    return RateLimitWrapper(_factory, model_name)


def get_llm(model_name: str) -> RateLimitWrapper:
    """Get a configured LLM instance by model name. Cached after first use.

    Args:
        model_name: The model identifier string.

    Returns:
        A configured rate-limit-safe model wrapper.
    """
    if model_name not in _registry:
        _registry[model_name] = _build_model(model_name)
    return _registry[model_name]
