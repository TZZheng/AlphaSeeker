from __future__ import annotations

# Keep this package initializer deliberately light: role base classes import
# team.envelope/team.mailbox, and eager coordinator imports would create a cycle.
from src.research_platform.team.envelope import MESSAGE_KINDS, ROLE_NAMES, MessageEnvelope, MessageKind, RoleName
from src.research_platform.team.mailbox import InboundMessage, Mailbox, MailboxStats
from src.research_platform.team.pad import Pad, PadMeta
from src.research_platform.team.state import RoleBudget, TeamConfig, TerminalResult

__all__ = [
    "Coordinator",
    "CoordinatorStepResult",
    "InboundMessage",
    "Mailbox",
    "MailboxStats",
    "MESSAGE_KINDS",
    "MessageEnvelope",
    "MessageKind",
    "Pad",
    "PadMeta",
    "ROLE_NAMES",
    "RoleBudget",
    "RoleName",
    "TeamConfig",
    "TerminalResult",
]


def __getattr__(name: str):
    if name in {"Coordinator", "CoordinatorStepResult"}:
        from src.research_platform.team.coordinator import Coordinator, CoordinatorStepResult

        return {"Coordinator": Coordinator, "CoordinatorStepResult": CoordinatorStepResult}[name]
    raise AttributeError(name)
