from __future__ import annotations

from datetime import datetime, timezone, timedelta
from random import Random

import pytest
from pydantic import ValidationError

from src.research_platform.team.envelope import MessageEnvelope, envelope_from_json, envelope_to_json
from src.research_platform.team.ids import new_correlation_id, new_message_id


def _clock(ts: datetime):
    return lambda: ts


def _env(**kwargs):
    base = dict(
        message_id="m-20260515T150000000-ABCDEFGH",
        correlation_id="root-20260515T150000000-12345678",
        from_role="writer",
        to_role="reviewer",
        kind="draft_ready",
        refs=(),
        body_path="messages/m-20260515T150000000-ABCDEFGH/body.md",
        body_sha256="a" * 64,
        body_bytes=10,
        created_at="2026-05-15T15:00:00Z",
        turn_index=0,
    )
    base.update(kwargs)
    return MessageEnvelope(**base)


def test_new_message_id_format_and_lex_sortable_with_injected_clock():
    rng = Random(1)
    first = new_message_id(clock=_clock(datetime(2026, 5, 15, 15, 0, 0, tzinfo=timezone.utc)), rng=rng)
    second = new_message_id(clock=_clock(datetime(2026, 5, 15, 15, 0, 1, tzinfo=timezone.utc)), rng=rng)
    assert first.startswith("m-20260515T150000000-")
    assert first < second


def test_correlation_id_prefix_allowlist():
    cid = new_correlation_id("issue", clock=_clock(datetime(2026, 5, 15, 15, 0, 0, tzinfo=timezone.utc)), rng=Random(2))
    assert cid.startswith("issue-20260515T150000000-")
    with pytest.raises(ValueError):
        new_correlation_id("bad")


def test_envelope_refs_dedupe_preserve_order_and_reject_whitespace():
    env = _env(refs=("I-1", "S2", "I-1"))
    assert env.refs == ("I-1", "S2")
    with pytest.raises(ValidationError):
        _env(refs=("bad ref",))


def test_envelope_route_matrix_accepts_and_rejects():
    assert _env(from_role="reviewer", to_role="coordinator", kind="accept").kind == "accept"
    with pytest.raises(ValidationError):
        _env(from_role="writer", to_role="coordinator", kind="accept")
    with pytest.raises(ValidationError):
        _env(from_role="human", to_role="writer", kind="review_note")


def test_envelope_round_trip_json_and_frozen():
    env = _env()
    assert envelope_from_json(envelope_to_json(env)) == env
    with pytest.raises(ValidationError):
        env.message_id = "x"  # type: ignore[misc]


def test_envelope_rejects_invalid_message_id_sha_and_ref_cap():
    with pytest.raises(ValidationError):
        _env(message_id="bad")
    with pytest.raises(ValidationError):
        _env(body_sha256="ABC")
    with pytest.raises(ValidationError):
        _env(refs=tuple(f"R{i}" for i in range(33)))
