from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from src.research_platform.roles.base import NoProgressNote, OutboundMessage, RoleTurnResult, derive_correlation_id_for_reply
from src.research_platform.team.state import TeamState
from src.research_platform.team.mailbox import Mailbox


def test_role_turn_result_rejects_outbound_and_no_progress_both():
    with pytest.raises(ValueError):
        RoleTurnResult(outbound=(OutboundMessage("reviewer", "draft_ready", "body"),), no_progress=NoProgressNote("no"))


def test_role_turn_result_is_no_progress_flag():
    assert RoleTurnResult(no_progress=NoProgressNote("waiting")).is_no_progress is True
    assert RoleTurnResult(outbound=(OutboundMessage("reviewer", "draft_ready", "body"),)).is_no_progress is False


def test_outbound_message_with_correlation_helper():
    out = OutboundMessage("reviewer", "draft_ready", "body")
    assert out.with_correlation("root-20260515T150000000-12345678").correlation_id == "root-20260515T150000000-12345678"


def test_team_state_view_is_immutable():
    state = TeamState.fresh(clock=None)
    view = state.view(now=state.started_at, role_budgets={})
    with pytest.raises(FrozenInstanceError):
        view.total_messages = 9  # type: ignore[misc]
