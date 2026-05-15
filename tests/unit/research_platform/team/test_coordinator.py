from __future__ import annotations

from datetime import datetime, timedelta, timezone
from random import Random

import pytest

from src.research_platform.roles.base import NoProgressNote, OutboundMessage, RoleTurnResult, PadAppend
from src.research_platform.roles.testing import CallbackRole, ScriptedRole, ScriptedTurn, out
from src.research_platform.team.coordinator import Coordinator
from src.research_platform.team.mailbox import Mailbox
from src.research_platform.team.pad import Pad
from src.research_platform.team.state import RoleBudget, TeamConfig


def _clock_factory(start: datetime | None = None):
    current = [start or datetime(2026, 5, 15, 15, 0, 0, tzinfo=timezone.utc)]

    def clock():
        return current[0]

    def advance(seconds: int):
        current[0] = current[0] + timedelta(seconds=seconds)

    return clock, advance


def _coord(tmp_path, roles, *, config=None, clock=None):
    pads = {role: Pad(tmp_path / "pads" / role / "pad.md", clock=clock) for role in ("writer", "reviewer", "source_maintainer")}
    role_docs = {role: f"# {role}" for role in pads}
    return Coordinator(
        mailbox=Mailbox(tmp_path / "mailbox"),
        roles=roles,
        config=config or TeamConfig.default(),
        memo_dir=tmp_path,
        role_docs=role_docs,
        pads=pads,
        clock=clock,
        rng=Random(1),
    )


def test_happy_path_reaches_accept(tmp_path):
    clock, _ = _clock_factory()
    writer = ScriptedRole("writer", [
        ScriptedTurn(expect_inbox_kinds=("kick_off",), result=RoleTurnResult(outbound=(out("reviewer", "draft_ready", "draft"),))),
    ])
    source = ScriptedRole("source_maintainer", [
        ScriptedTurn(expect_inbox_kinds=("kick_off",), result=RoleTurnResult(outbound=(
            out("writer", "source_update", "sources"),
            out("reviewer", "source_update", "sources"),
        ))),
    ])
    reviewer = ScriptedRole("reviewer", [
        RoleTurnResult(outbound=(out("coordinator", "accept", "ship"),)),
    ])
    coord = _coord(tmp_path, {"writer": writer, "reviewer": reviewer, "source_maintainer": source}, clock=clock)
    coord.kick_off(body_md="memo")
    terminal = coord.run_until_terminal()
    assert terminal.reason == "accepted"
    assert terminal.review_status == "accepted"
    assert (tmp_path / "team" / "team_state.json").exists()


def test_kick_off_creates_writer_and_source_messages(tmp_path):
    coord = _coord(tmp_path, {"writer": ScriptedRole("writer", []), "reviewer": ScriptedRole("reviewer", []), "source_maintainer": ScriptedRole("source_maintainer", [])})
    ids = coord.kick_off(body_md="start")
    assert len(ids) == 2
    assert [m.to_role for m in coord.mailbox.list_all()] == ["writer", "source_maintainer"]


def test_step_consumes_inbox_persists_outbound_and_pad(tmp_path):
    clock, _ = _clock_factory()
    writer = ScriptedRole("writer", [
        RoleTurnResult(outbound=(out("reviewer", "draft_ready", "draft"),), pad_appends=(PadAppend("note", "body"),)),
    ])
    coord = _coord(tmp_path, {"writer": writer, "reviewer": ScriptedRole("reviewer", []), "source_maintainer": ScriptedRole("source_maintainer", [])}, clock=clock)
    coord.kick_off(body_md="start", also_to_source_maintainer=False)
    result = coord.step()
    assert result.role == "writer"
    assert result.outbound_ids
    assert not coord.mailbox.inbox_for("writer")
    assert "body" in coord.pads["writer"].read()


def test_urgent_kind_priority(tmp_path):
    calls: list[str] = []

    def writer_cb(ctx, inbox):
        calls.append("writer")
        return RoleTurnResult(no_progress=NoProgressNote("done"))

    def reviewer_cb(ctx, inbox):
        calls.append("reviewer")
        return RoleTurnResult(no_progress=NoProgressNote("done"))

    coord = _coord(tmp_path, {"writer": CallbackRole("writer", writer_cb), "reviewer": CallbackRole("reviewer", reviewer_cb), "source_maintainer": ScriptedRole("source_maintainer", [])})
    cid = "root-20260515T150000000-12345678"
    coord.mailbox.send(from_role="coordinator", to_role="reviewer", kind="defer", body_md="low", correlation_id=cid, turn_index=0, rng=Random(2))
    coord.mailbox.send(from_role="reviewer", to_role="writer", kind="review_note", body_md="urgent", correlation_id=cid, turn_index=0, rng=Random(3))
    coord.step()
    assert calls == ["writer"]


def test_role_exception_and_silent_turn_are_failures(tmp_path):
    def boom(ctx, inbox):
        raise RuntimeError("boom")

    coord = _coord(tmp_path / "a", {"writer": CallbackRole("writer", boom), "reviewer": ScriptedRole("reviewer", []), "source_maintainer": ScriptedRole("source_maintainer", [])})
    coord.kick_off(body_md="start", also_to_source_maintainer=False)
    assert coord.step().terminal.reason == "give_up_role_failed"

    coord2 = _coord(tmp_path / "b", {"writer": ScriptedRole("writer", [RoleTurnResult()]), "reviewer": ScriptedRole("reviewer", []), "source_maintainer": ScriptedRole("source_maintainer", [])})
    coord2.kick_off(body_md="start", also_to_source_maintainer=False)
    assert coord2.step().terminal.reason == "give_up_role_failed"


def test_no_progress_streak_yields_stuck_escalated(tmp_path):
    config = TeamConfig(role_budgets={"writer": RoleBudget(5), "reviewer": RoleBudget(5), "source_maintainer": RoleBudget(5)}, max_no_progress_streak=2)
    writer = ScriptedRole("writer", [
        RoleTurnResult(no_progress=NoProgressNote("wait")),
        RoleTurnResult(no_progress=NoProgressNote("wait")),
    ])
    coord = _coord(tmp_path, {"writer": writer, "reviewer": ScriptedRole("reviewer", []), "source_maintainer": ScriptedRole("source_maintainer", [])}, config=config)
    coord.kick_off(body_md="start", also_to_source_maintainer=False)
    # Re-inject a second message so writer is eligible for a second no-progress turn.
    coord.step()
    coord.mailbox.send(from_role="coordinator", to_role="writer", kind="defer", body_md="again", correlation_id="root-20260515T150000000-12345678", turn_index=0, rng=Random(5))
    result = coord.step()
    assert result.terminal.reason == "stuck_escalated"


def test_wall_clock_and_total_message_caps(tmp_path):
    clock, advance = _clock_factory()
    config = TeamConfig.default()
    config = TeamConfig(role_budgets=config.role_budgets, wall_clock_budget_s=1)
    coord = _coord(tmp_path / "wall", {"writer": ScriptedRole("writer", []), "reviewer": ScriptedRole("reviewer", []), "source_maintainer": ScriptedRole("source_maintainer", [])}, config=config, clock=clock)
    coord.kick_off(body_md="start", also_to_source_maintainer=False)
    advance(2)
    assert coord.step().terminal.reason == "give_up_wall_clock"

    config2 = TeamConfig(role_budgets=TeamConfig.default().role_budgets, total_messages_cap=1)
    coord2 = _coord(tmp_path / "cap", {"writer": ScriptedRole("writer", []), "reviewer": ScriptedRole("reviewer", []), "source_maintainer": ScriptedRole("source_maintainer", [])}, config=config2)
    coord2.kick_off(body_md="start", also_to_source_maintainer=False)
    assert coord2.step().terminal.reason == "give_up_total_messages_cap"


def test_dissent_depth_injects_defer_and_accept_with_caveats(tmp_path):
    config = TeamConfig(role_budgets={"writer": RoleBudget(5), "reviewer": RoleBudget(5), "source_maintainer": RoleBudget(5)}, max_dissent_depth=0, max_issue_retry_depth=0)
    writer = ScriptedRole("writer", [RoleTurnResult(outbound=(out("reviewer", "dissent", "no", refs=("I-1",)),))])
    reviewer = ScriptedRole("reviewer", [RoleTurnResult(outbound=(out("coordinator", "accept", "ok"),))])
    coord = _coord(tmp_path, {"writer": writer, "reviewer": reviewer, "source_maintainer": ScriptedRole("source_maintainer", [])}, config=config)
    coord.kick_off(body_md="start", also_to_source_maintainer=False)
    coord.step()
    terminal = coord.run_until_terminal()
    assert terminal.review_status == "accepted_with_caveats"
