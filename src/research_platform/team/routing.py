from __future__ import annotations

from src.research_platform.team.envelope import MESSAGE_KINDS, ROLE_NAMES, MessageKind, RoleName

ALLOWED_ROUTES: dict[MessageKind, set[tuple[RoleName, RoleName]]] = {
    "kick_off": {("coordinator", "writer"), ("coordinator", "source_maintainer")},
    "draft_ready": {("writer", "reviewer")},
    "review_note": {("reviewer", "writer")},
    "source_request": {("writer", "source_maintainer"), ("reviewer", "source_maintainer")},
    "source_update": {("source_maintainer", "writer"), ("source_maintainer", "reviewer")},
    "source_unavailable": {
        ("source_maintainer", "writer"),
        ("source_maintainer", "reviewer"),
        ("coordinator", "writer"),
        ("coordinator", "reviewer"),
    },
    "dissent": {("writer", "reviewer")},
    "defer": {
        ("writer", "reviewer"),
        ("reviewer", "writer"),
        ("coordinator", "reviewer"),
        ("coordinator", "writer"),
    },
    "escalate": {
        ("writer", "coordinator"),
        ("reviewer", "coordinator"),
        ("source_maintainer", "coordinator"),
        ("coordinator", "human"),
    },
    "done": {("writer", "reviewer")},
    "accept": {("reviewer", "coordinator")},
    "give_up": {
        ("coordinator", "writer"),
        ("coordinator", "reviewer"),
        ("coordinator", "source_maintainer"),
    },
}

URGENT_KINDS: frozenset[MessageKind] = frozenset({"kick_off", "source_update", "dissent", "review_note"})


def is_allowed_route(from_role: str, to_role: str, kind: str) -> bool:
    if from_role not in ROLE_NAMES or to_role not in ROLE_NAMES or kind not in MESSAGE_KINDS:
        return False
    return (from_role, to_role) in ALLOWED_ROUTES.get(kind, set())
