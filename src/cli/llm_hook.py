"""LLM observer hook for the CLI dashboard."""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any

from src.shared.llm_manager import set_llm_observer


@dataclass
class LLMCallRecord:
    """Thread-safe record of one LLM call."""

    model_name: str
    prompt: Any
    response: Any
    thinking: Any = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class LLMObserver:
    """
    Thread-safe LLM call buffer consumed by the dashboard's LLM log tab.

    Registered as the global LLM observer via ``set_llm_observer``.
    """

    def __init__(self, max_entries: int = 500):
        self._buffer: deque[LLMCallRecord] = deque(maxlen=max_entries)
        self._lock = Lock()

    def __call__(self, model_name: str, prompt: Any, response: Any, thinking: Any = None) -> None:
        record = LLMCallRecord(
            model_name=model_name,
            prompt=prompt,
            response=response,
            thinking=thinking,
        )
        with self._lock:
            self._buffer.append(record)

    def get_recent(self, n: int = 50) -> list[LLMCallRecord]:
        """Return the n most recent records, oldest first."""
        with self._lock:
            items = list(self._buffer)[-n:]
        return items

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()


# Global singleton
_observer: LLMObserver | None = None


def install_llm_observer() -> LLMObserver:
    """Install the global LLM observer. Returns the observer for consumers."""
    global _observer
    _observer = LLMObserver()
    set_llm_observer(_observer)
    return _observer


def uninstall_llm_observer() -> None:
    """Remove the global LLM observer."""
    global _observer
    set_llm_observer(None)
    _observer = None


def get_observer() -> LLMObserver | None:
    """Return the current observer, if installed."""
    return _observer
