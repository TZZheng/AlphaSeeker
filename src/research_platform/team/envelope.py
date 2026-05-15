from __future__ import annotations

import re
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.research_platform.team.ids import CORRELATION_ID_RE, MESSAGE_ID_RE

RoleName = Literal["coordinator", "writer", "reviewer", "source_maintainer", "human"]
ROLE_NAMES: tuple[RoleName, ...] = get_args(RoleName)

MessageKind = Literal[
    "kick_off",
    "draft_ready",
    "review_note",
    "source_request",
    "source_update",
    "source_unavailable",
    "dissent",
    "defer",
    "escalate",
    "done",
    "accept",
    "give_up",
]
MESSAGE_KINDS: tuple[MessageKind, ...] = get_args(MessageKind)
CostHint = Literal["small", "medium", "large"]


class MessageEnvelope(BaseModel):
    """Thin routing envelope for v9 team messages.

    The envelope is intentionally about transport and flow control only. The body
    remains markdown and is not interpreted by the coordinator.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    message_id: str
    correlation_id: str
    from_role: RoleName
    to_role: RoleName
    kind: MessageKind
    refs: tuple[str, ...] = Field(default_factory=tuple)
    body_path: str
    body_sha256: str
    body_bytes: int = Field(ge=0)
    created_at: str
    turn_index: int = Field(ge=0)
    cost_hint: CostHint = "small"

    @field_validator("message_id")
    @classmethod
    def _validate_message_id(cls, value: str) -> str:
        if not re.fullmatch(MESSAGE_ID_RE, value):
            raise ValueError(f"invalid message_id: {value}")
        return value

    @field_validator("correlation_id")
    @classmethod
    def _validate_correlation_id(cls, value: str) -> str:
        if not re.fullmatch(CORRELATION_ID_RE, value):
            raise ValueError(f"invalid correlation_id: {value}")
        return value

    @field_validator("body_sha256")
    @classmethod
    def _validate_sha(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("body_sha256 must be 64 lowercase hex chars")
        return value

    @field_validator("refs", mode="after")
    @classmethod
    def _validate_refs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        seen: set[str] = set()
        out: list[str] = []
        for ref in value:
            if not ref or any(char.isspace() for char in ref):
                raise ValueError(f"invalid ref: {ref!r}")
            if ref in seen:
                continue
            seen.add(ref)
            out.append(ref)
        if len(out) > 32:
            raise ValueError("refs capped at 32 entries")
        return tuple(out)

    @model_validator(mode="after")
    def _validate_route(self) -> "MessageEnvelope":
        from src.research_platform.team.routing import is_allowed_route

        if not is_allowed_route(self.from_role, self.to_role, self.kind):
            raise ValueError(f"disallowed route: {self.from_role}->{self.to_role} for kind={self.kind}")
        return self


def envelope_to_json(envelope: MessageEnvelope) -> str:
    return envelope.model_dump_json(indent=2)


def envelope_from_json(text: str) -> MessageEnvelope:
    return MessageEnvelope.model_validate_json(text)
