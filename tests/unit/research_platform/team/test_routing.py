from __future__ import annotations

from src.research_platform.team.envelope import MESSAGE_KINDS
from src.research_platform.team.routing import ALLOWED_ROUTES, URGENT_KINDS, is_allowed_route


def test_routing_matrix_completeness_every_message_kind_has_route():
    assert set(ALLOWED_ROUTES) == set(MESSAGE_KINDS)
    assert all(ALLOWED_ROUTES[kind] for kind in MESSAGE_KINDS)


def test_urgent_kinds_subset_of_message_kinds():
    assert set(URGENT_KINDS).issubset(set(MESSAGE_KINDS))


def test_is_allowed_route_rejects_unknown_role_and_accepts_canonical():
    assert not is_allowed_route("bogus", "writer", "kick_off")
    assert is_allowed_route("writer", "reviewer", "draft_ready")
