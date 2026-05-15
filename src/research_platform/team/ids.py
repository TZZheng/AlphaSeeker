from __future__ import annotations

import re
import secrets
from collections.abc import Callable
from datetime import datetime, timezone
from random import Random
from typing import Protocol

_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_ALLOWED_CORRELATION_PREFIXES = {"c", "root", "issue", "source", "escalation"}

MESSAGE_ID_RE = r"^m-\d{8}T\d{9}-[0-9A-Z]{8}$"
CORRELATION_ID_RE = r"^[a-z]{1,16}-\d{8}T\d{9}-[0-9A-Z]{8}$"


class _ChoiceRng(Protocol):
    def choice(self, seq: str) -> str: ...


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def now_iso(clock: Callable[[], datetime] | None = None) -> str:
    return iso_utc((clock or utc_now)())


def _now_iso_compact(clock: Callable[[], datetime] | None = None) -> str:
    now = (clock or utc_now)()
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%f")[:-3]


def _random_suffix(rng: _ChoiceRng | None = None, n: int = 8) -> str:
    chooser = rng or secrets.SystemRandom()
    return "".join(chooser.choice(_ALPHABET) for _ in range(n))


def seeded_rng(seed: int) -> Random:
    return Random(seed)


def new_message_id(*, clock: Callable[[], datetime] | None = None, rng: _ChoiceRng | None = None) -> str:
    return f"m-{_now_iso_compact(clock)}-{_random_suffix(rng)}"


def new_correlation_id(prefix: str = "c", *, clock: Callable[[], datetime] | None = None, rng: _ChoiceRng | None = None) -> str:
    if prefix not in _ALLOWED_CORRELATION_PREFIXES:
        raise ValueError(f"invalid correlation prefix: {prefix}")
    return f"{prefix}-{_now_iso_compact(clock)}-{_random_suffix(rng)}"


def is_message_id(value: str) -> bool:
    return re.fullmatch(MESSAGE_ID_RE, value) is not None


def is_correlation_id(value: str) -> bool:
    return re.fullmatch(CORRELATION_ID_RE, value) is not None
