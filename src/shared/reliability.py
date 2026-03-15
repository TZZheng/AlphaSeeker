"""
Retry, timeout, and cache helpers for external data fetches.

The goal is pragmatic reliability:
- `tenacity` handles retry/backoff for transient dependency failures.
- an on-disk TTL cache avoids re-fetching the same payload during retries and
  repeated runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence, TypeVar

import requests
from tenacity import Retrying, retry_if_exception_type, stop_after_attempt, wait_exponential

T = TypeVar("T")

_CACHE_ROOT = Path(os.getcwd()) / ".cache" / "alphaseeker"
_CACHE_LOCK = threading.Lock()


def make_cache_key(parts: Any) -> str:
    """Create a deterministic hash key from nested Python data."""
    payload = json.dumps(parts, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_file(namespace: str, key: str) -> Path:
    namespace_dir = _CACHE_ROOT / namespace.replace(":", "_").replace("/", "_")
    namespace_dir.mkdir(parents=True, exist_ok=True)
    return namespace_dir / f"{key}.pkl"


def cached_call(
    namespace: str,
    key_parts: Any,
    loader: Callable[[], T],
    *,
    ttl_seconds: int,
) -> T:
    """Load a value from the TTL cache or compute and persist it atomically."""
    if ttl_seconds <= 0:
        return loader()

    key = make_cache_key(key_parts)
    cache_path = _cache_file(namespace, key)
    now = time.time()

    if cache_path.exists():
        try:
            with cache_path.open("rb") as fh:
                payload = pickle.load(fh)
            expires_at = float(payload.get("expires_at", 0))
            if expires_at > now:
                return payload["value"]
        except Exception:
            cache_path.unlink(missing_ok=True)

    value = loader()
    payload = {"expires_at": now + ttl_seconds, "value": value}

    with _CACHE_LOCK:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=cache_path.parent, delete=False) as tmp:
            pickle.dump(payload, tmp)
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, cache_path)

    return value


def retry_call(
    loader: Callable[[], T],
    *,
    attempts: int = 3,
    retry_exceptions: tuple[type[BaseException], ...] = (Exception,),
    min_wait_seconds: int = 1,
    max_wait_seconds: int = 8,
) -> T:
    """Retry a transient external call with exponential backoff."""
    for attempt in Retrying(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(multiplier=1, min=min_wait_seconds, max=max_wait_seconds),
        retry=retry_if_exception_type(retry_exceptions),
        reraise=True,
    ):
        with attempt:
            return loader()
    raise RuntimeError("Retry loop exited without returning a value")


def cached_retry_call(
    namespace: str,
    key_parts: Any,
    loader: Callable[[], T],
    *,
    ttl_seconds: int,
    attempts: int = 3,
    retry_exceptions: tuple[type[BaseException], ...] = (Exception,),
    min_wait_seconds: int = 1,
    max_wait_seconds: int = 8,
) -> T:
    """Combine retry/backoff with a TTL cache around the successful result."""
    return cached_call(
        namespace,
        key_parts,
        lambda: retry_call(
            loader,
            attempts=attempts,
            retry_exceptions=retry_exceptions,
            min_wait_seconds=min_wait_seconds,
            max_wait_seconds=max_wait_seconds,
        ),
        ttl_seconds=ttl_seconds,
    )


def request_json(
    url: str,
    *,
    params: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: int = 15,
    ttl_seconds: int = 900,
    attempts: int = 3,
) -> Any:
    """HTTP GET returning JSON, protected by retry/backoff and the TTL cache."""

    def _load() -> Any:
        response = requests.get(url, params=params, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.json()

    return cached_retry_call(
        "http_json",
        {"url": url, "params": params, "headers": headers, "timeout": timeout},
        _load,
        ttl_seconds=ttl_seconds,
        attempts=attempts,
        retry_exceptions=(requests.RequestException,),
    )


def request_text(
    url: str,
    *,
    params: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: int = 15,
    ttl_seconds: int = 900,
    attempts: int = 3,
) -> str:
    """HTTP GET returning response text, protected by retry/backoff and cache."""

    def _load() -> str:
        response = requests.get(url, params=params, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.text

    return cached_retry_call(
        "http_text",
        {"url": url, "params": params, "headers": headers, "timeout": timeout},
        _load,
        ttl_seconds=ttl_seconds,
        attempts=attempts,
        retry_exceptions=(requests.RequestException,),
    )


def request_bytes(
    url: str,
    *,
    params: Mapping[str, Any] | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: int = 30,
    ttl_seconds: int = 3600,
    attempts: int = 3,
) -> bytes:
    """HTTP GET returning raw bytes, protected by retry/backoff and cache."""

    def _load() -> bytes:
        response = requests.get(url, params=params, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.content

    return cached_retry_call(
        "http_bytes",
        {"url": url, "params": params, "headers": headers, "timeout": timeout},
        _load,
        ttl_seconds=ttl_seconds,
        attempts=attempts,
        retry_exceptions=(requests.RequestException,),
        min_wait_seconds=2,
        max_wait_seconds=12,
    )
