from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from src.research_platform.team.envelope import CostHint, MessageKind, RoleName
from src.research_platform.team.mailbox import InboundMessage, Mailbox
from src.research_platform.team.pad import Pad
from src.research_platform.team.state import TeamStateView


@dataclass(frozen=True)
class RoleConfig:
    role_name: RoleName
    extras: dict[str, object] = field(default_factory=dict)


class TurnLogger:
    def info(self, message: str, **kv: object) -> None:  # pragma: no cover - interface default
        pass

    def warning(self, message: str, **kv: object) -> None:  # pragma: no cover
        pass

    def error(self, message: str, **kv: object) -> None:  # pragma: no cover
        pass


@dataclass(frozen=True)
class RoleContext:
    role_name: RoleName
    role_doc: str
    pad: Pad
    mailbox: Mailbox
    memo_dir: Path
    team_state: TeamStateView
    config: RoleConfig
    logger: TurnLogger
    turn_index: int


@dataclass(frozen=True)
class OutboundMessage:
    to_role: RoleName
    kind: MessageKind
    body_md: str
    refs: tuple[str, ...] = ()
    correlation_id: str | None = None
    cost_hint: CostHint = "small"

    def with_correlation(self, cid: str) -> "OutboundMessage":
        return OutboundMessage(
            to_role=self.to_role,
            kind=self.kind,
            body_md=self.body_md,
            refs=self.refs,
            correlation_id=cid,
            cost_hint=self.cost_hint,
        )


@dataclass(frozen=True)
class NoProgressNote:
    reason: str
    pad_note: str | None = None


@dataclass(frozen=True)
class PadAppend:
    title: str
    body: str


@dataclass(frozen=True)
class RoleTurnResult:
    outbound: tuple[OutboundMessage, ...] = ()
    no_progress: NoProgressNote | None = None
    pad_appends: tuple[PadAppend, ...] = ()

    def __post_init__(self) -> None:
        if self.outbound and self.no_progress is not None:
            raise ValueError("RoleTurnResult cannot have both outbound and no_progress")

    @property
    def is_no_progress(self) -> bool:
        return not self.outbound and self.no_progress is not None


class Role(ABC):
    name: RoleName

    @abstractmethod
    def take_turn(self, ctx: RoleContext, inbox: tuple[InboundMessage, ...]) -> RoleTurnResult:
        """Process inbound messages and return outbound messages plus pad updates."""


def derive_correlation_id_for_reply(inbox: tuple[InboundMessage, ...], *, kind: MessageKind) -> str | None:
    if not inbox:
        return None
    return max(inbox, key=lambda message: message.envelope.message_id).envelope.correlation_id
