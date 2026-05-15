from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from src.research_platform.roles.base import OutboundMessage, Role, RoleContext, RoleTurnResult, NoProgressNote
from src.research_platform.team.envelope import CostHint, MessageKind, RoleName
from src.research_platform.team.mailbox import InboundMessage


@dataclass
class ScriptedTurn:
    expect_inbox_kinds: tuple[MessageKind, ...] | None = None
    result: RoleTurnResult = field(default_factory=RoleTurnResult)


class ScriptedRole(Role):
    def __init__(self, name: RoleName, turns: list[ScriptedTurn | RoleTurnResult]) -> None:
        self.name = name
        self._turns = list(turns)
        self._cursor = 0

    def take_turn(self, ctx: RoleContext, inbox: tuple[InboundMessage, ...]) -> RoleTurnResult:
        if self._cursor >= len(self._turns):
            return RoleTurnResult(no_progress=NoProgressNote(reason="script exhausted"))
        item = self._turns[self._cursor]
        self._cursor += 1
        if isinstance(item, ScriptedTurn):
            if item.expect_inbox_kinds is not None:
                actual = tuple(message.envelope.kind for message in inbox)
                if actual != item.expect_inbox_kinds:
                    raise AssertionError(f"inbox kinds {actual!r} != expected {item.expect_inbox_kinds!r}")
            return item.result
        return item


class CallbackRole(Role):
    def __init__(self, name: RoleName, callback: Callable[[RoleContext, tuple[InboundMessage, ...]], RoleTurnResult]) -> None:
        self.name = name
        self._callback = callback

    def take_turn(self, ctx: RoleContext, inbox: tuple[InboundMessage, ...]) -> RoleTurnResult:
        return self._callback(ctx, inbox)


def out(
    to: RoleName,
    kind: MessageKind,
    body: str = "",
    refs: tuple[str, ...] = (),
    correlation_id: str | None = None,
    cost: CostHint = "small",
) -> OutboundMessage:
    return OutboundMessage(to_role=to, kind=kind, body_md=body or kind, refs=refs, correlation_id=correlation_id, cost_hint=cost)
