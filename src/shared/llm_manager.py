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
"""

import os
from typing import Any, Callable, Dict, Protocol, cast
from pydantic import SecretStr

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
                return self.model.invoke(*args, **kwargs)
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
                return self.model.stream(*args, **kwargs)
        raise RuntimeError("Retry loop exited without invoking model")

    def with_structured_output(self, *args, **kwargs):
        """
        Bind structured output configuration to a NEW factory.
        This ensures the binding is re-applied if we switch models.
        """
        # print(f"DEBUG: RateLimitWrapper.with_structured_output factory creation", flush=True)
        def new_factory(name: str) -> SupportsModelOps:
            base = self.model_factory(name)
            return _bind_model_method(base, "with_structured_output", *args, **kwargs)
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
