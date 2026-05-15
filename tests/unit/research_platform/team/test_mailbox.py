from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from random import Random

import pytest

from src.research_platform.team.envelope import envelope_to_json
from src.research_platform.team.ids import new_message_id
from src.research_platform.team.mailbox import DuplicateMessageIdError, Mailbox, UnknownMessageIdError


def _clock():
    return datetime(2026, 5, 15, 15, 0, 0, tzinfo=timezone.utc)


def _send(mailbox: Mailbox, *, mid: str, to="reviewer", kind="draft_ready", cid="root-20260515T150000000-12345678"):
    return mailbox.send(
        from_role="coordinator" if kind == "kick_off" else ("reviewer" if kind == "review_note" else "writer"),
        to_role=to,
        kind=kind,
        body_md=f"body {mid}",
        correlation_id=cid,
        turn_index=0,
        clock=_clock,
        rng=Random(1),
        message_id=mid,
    )


def test_send_writes_envelope_body_index_and_sha(tmp_path):
    mailbox = Mailbox(tmp_path / "mailbox")
    mid = "m-20260515T150000000-ABCDEFGH"
    env = _send(mailbox, mid=mid)
    assert (tmp_path / "mailbox" / "messages" / mid / "body.md").read_text() == f"body {mid}"
    assert (tmp_path / "mailbox" / "messages" / mid / "envelope.json").exists()
    assert (tmp_path / "mailbox" / "index.jsonl").read_text().strip()
    assert env.body_sha256 == hashlib.sha256(f"body {mid}".encode()).hexdigest()


def test_duplicate_message_id_raises(tmp_path):
    mailbox = Mailbox(tmp_path / "mailbox")
    mid = "m-20260515T150000000-ABCDEFGH"
    _send(mailbox, mid=mid)
    with pytest.raises(DuplicateMessageIdError):
        _send(mailbox, mid=mid)


def test_inbox_filters_consumed_and_orders_by_message_id(tmp_path):
    mailbox = Mailbox(tmp_path / "mailbox")
    later = "m-20260515T150001000-ABCDEFGH"
    earlier = "m-20260515T150000000-ABCDEFGH"
    other = "m-20260515T150002000-ABCDEFGH"
    _send(mailbox, mid=later)
    _send(mailbox, mid=earlier)
    _send(mailbox, mid=other, to="writer", kind="review_note")
    assert [m.envelope.message_id for m in mailbox.inbox_for("reviewer")] == [earlier, later]
    assert mailbox.mark_consumed(earlier, "reviewer") is True
    assert mailbox.mark_consumed(earlier, "reviewer") is False
    assert [m.envelope.message_id for m in mailbox.inbox_for("reviewer")] == [later]
    restarted = Mailbox(tmp_path / "mailbox")
    assert restarted.consumed_set("reviewer") == {earlier}


def test_thread_and_stats(tmp_path):
    mailbox = Mailbox(tmp_path / "mailbox")
    cid1 = "root-20260515T150000000-12345678"
    cid2 = "root-20260515T150000000-ABCDEFGH"
    _send(mailbox, mid="m-20260515T150000000-ABCDEFGH", cid=cid1)
    _send(mailbox, mid="m-20260515T150001000-ABCDEFGH", cid=cid2)
    assert [m.envelope.correlation_id for m in mailbox.thread(cid1)] == [cid1]
    stats = mailbox.stats()
    assert stats.total_messages == 2
    assert stats.by_to_role["reviewer"] == 2
    assert stats.by_correlation_id[cid1] == 1


def test_get_unknown_and_recover_index_from_messages_dir(tmp_path):
    mailbox = Mailbox(tmp_path / "mailbox")
    with pytest.raises(UnknownMessageIdError):
        mailbox.get("m-20260515T150000000-XXXXXXXX")
    mid = "m-20260515T150000000-ABCDEFGH"
    env = _send(mailbox, mid=mid)
    (tmp_path / "mailbox" / "index.jsonl").unlink()
    recovered = Mailbox(tmp_path / "mailbox")
    assert [e.message_id for e in recovered.list_all()] == [mid]
