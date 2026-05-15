from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.research_platform.team.envelope import (
    CostHint,
    MessageEnvelope,
    MessageKind,
    RoleName,
    envelope_from_json,
    envelope_to_json,
)
from src.research_platform.team.ids import new_message_id, now_iso


@dataclass(frozen=True)
class InboundMessage:
    envelope: MessageEnvelope
    body_md: str


@dataclass(frozen=True)
class MailboxStats:
    total_messages: int
    by_kind: dict[str, int]
    by_to_role: dict[str, int]
    by_correlation_id: dict[str, int]
    consumed_by_role: dict[str, int]


class DuplicateMessageIdError(ValueError):
    pass


class UnknownMessageIdError(KeyError):
    pass


class MailboxIntegrityError(RuntimeError):
    pass


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip("\n") + "\n")
        handle.flush()
        os.fsync(handle.fileno())


class Mailbox:
    def __init__(self, root: Path, *, single_writer: bool = True) -> None:
        self.root = Path(root)
        self.single_writer = single_writer
        (self.root / "messages").mkdir(parents=True, exist_ok=True)
        (self.root / "consumed").mkdir(parents=True, exist_ok=True)
        self._index_path = self.root / "index.jsonl"
        self._consumed_index_path = self.root / "index.consumed.jsonl"
        self._ordered_ids = self._load_index_ids()

    def send(
        self,
        *,
        from_role: RoleName,
        to_role: RoleName,
        kind: MessageKind,
        body_md: str,
        correlation_id: str,
        refs: tuple[str, ...] = (),
        turn_index: int,
        cost_hint: CostHint = "small",
        clock=None,
        rng=None,
        message_id: str | None = None,
    ) -> MessageEnvelope:
        mid = message_id or new_message_id(clock=clock, rng=rng)
        message_dir = self.root / "messages" / mid
        if message_dir.exists() or mid in self._ordered_ids:
            raise DuplicateMessageIdError(mid)
        encoded = body_md.encode("utf-8")
        sha = hashlib.sha256(encoded).hexdigest()
        body_rel = f"messages/{mid}/body.md"
        envelope = MessageEnvelope(
            message_id=mid,
            correlation_id=correlation_id,
            from_role=from_role,
            to_role=to_role,
            kind=kind,
            refs=refs,
            body_path=body_rel,
            body_sha256=sha,
            body_bytes=len(encoded),
            created_at=now_iso(clock),
            turn_index=turn_index,
            cost_hint=cost_hint,
        )
        _atomic_write_text(self.root / body_rel, body_md)
        _atomic_write_text(message_dir / "envelope.json", envelope_to_json(envelope))
        _atomic_append_line(
            self._index_path,
            json.dumps(
                {
                    "id": mid,
                    "kind": kind,
                    "from": from_role,
                    "to": to_role,
                    "correlation_id": correlation_id,
                },
                sort_keys=True,
            ),
        )
        self._ordered_ids.append(mid)
        self._ordered_ids = sorted(dict.fromkeys(self._ordered_ids))
        return envelope

    def get(self, message_id: str) -> InboundMessage:
        if message_id not in self._ordered_ids and not (self.root / "messages" / message_id / "envelope.json").exists():
            raise UnknownMessageIdError(message_id)
        envelope = self._load_envelope(message_id)
        body_path = self.root / envelope.body_path
        if not body_path.exists():
            raise MailboxIntegrityError(f"missing body for {message_id}")
        body = body_path.read_text(encoding="utf-8")
        actual_sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
        if actual_sha != envelope.body_sha256:
            raise MailboxIntegrityError(f"body sha mismatch for {message_id}")
        return InboundMessage(envelope=envelope, body_md=body)

    def list_all(self) -> list[MessageEnvelope]:
        return [self._load_envelope(mid) for mid in self._ordered_ids]

    def inbox_for(self, role: RoleName) -> list[InboundMessage]:
        consumed = self.consumed_set(role)
        out: list[InboundMessage] = []
        for mid in self._ordered_ids:
            envelope = self._load_envelope(mid)
            if envelope.to_role != role or mid in consumed:
                continue
            out.append(self.get(mid))
        return out

    def thread(self, correlation_id: str) -> list[InboundMessage]:
        return [self.get(mid) for mid in self._ordered_ids if self._load_envelope(mid).correlation_id == correlation_id]

    def mark_consumed(self, message_id: str, role: RoleName) -> bool:
        if not (self.root / "messages" / message_id / "envelope.json").exists():
            raise UnknownMessageIdError(message_id)
        marker = self.root / "consumed" / role / message_id
        if marker.exists():
            return False
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_bytes(b"")
        _atomic_append_line(self._consumed_index_path, json.dumps({"id": message_id, "role": role, "at": now_iso()}))
        return True

    def consumed_set(self, role: RoleName) -> set[str]:
        root = self.root / "consumed" / role
        if not root.exists():
            return set()
        return {p.name for p in root.iterdir() if p.is_file()}

    def is_consumed(self, message_id: str, role: RoleName) -> bool:
        return (self.root / "consumed" / role / message_id).exists()

    def stats(self) -> MailboxStats:
        envelopes = self.list_all()
        consumed_by_role: dict[str, int] = {}
        for role_dir in (self.root / "consumed").glob("*"):
            if role_dir.is_dir():
                consumed_by_role[role_dir.name] = len([p for p in role_dir.iterdir() if p.is_file()])
        return MailboxStats(
            total_messages=len(envelopes),
            by_kind=dict(Counter(env.kind for env in envelopes)),
            by_to_role=dict(Counter(env.to_role for env in envelopes)),
            by_correlation_id=dict(Counter(env.correlation_id for env in envelopes)),
            consumed_by_role=consumed_by_role,
        )

    def _load_index_ids(self) -> list[str]:
        ids: list[str] = []
        if self._index_path.exists():
            for line in self._index_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    mid = json.loads(line)["id"]
                except Exception as exc:  # pragma: no cover - corrupt file path
                    raise MailboxIntegrityError(f"invalid index line: {line!r}") from exc
                if (self.root / "messages" / mid / "envelope.json").exists() and mid not in ids:
                    ids.append(mid)
        discovered = []
        for envelope_path in (self.root / "messages").glob("*/envelope.json"):
            mid = envelope_path.parent.name
            if mid not in ids:
                discovered.append(mid)
        if discovered:
            for mid in sorted(discovered):
                env = self._load_envelope(mid)
                _atomic_append_line(
                    self._index_path,
                    json.dumps(
                        {"id": mid, "kind": env.kind, "from": env.from_role, "to": env.to_role, "correlation_id": env.correlation_id},
                        sort_keys=True,
                    ),
                )
                ids.append(mid)
        return sorted(dict.fromkeys(ids))

    def _load_envelope(self, message_id: str) -> MessageEnvelope:
        path = self.root / "messages" / message_id / "envelope.json"
        if not path.exists():
            raise UnknownMessageIdError(message_id)
        return envelope_from_json(path.read_text(encoding="utf-8"))
