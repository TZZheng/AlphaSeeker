from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from src.research_platform.team.envelope import RoleName
from src.research_platform.team.ids import iso_utc, utc_now

TerminalReason = Literal[
    "accepted",
    "give_up_budget_exhausted",
    "give_up_total_messages_cap",
    "give_up_wall_clock",
    "give_up_role_failed",
    "stuck_escalated",
    "failed_unrecoverable",
]
ReviewStatus = Literal[
    "accepted",
    "accepted_with_caveats",
    "revise_exhausted",
    "failed_unrecoverable",
    "stuck_escalated",
]

TEAM_ROLES: tuple[RoleName, ...] = ("writer", "reviewer", "source_maintainer")


@dataclass(frozen=True)
class RoleBudget:
    max_turns: int


@dataclass(frozen=True)
class TeamConfig:
    role_budgets: dict[RoleName, RoleBudget]
    wall_clock_budget_s: int = 2400
    total_messages_cap: int = 60
    max_dissent_depth: int = 2
    max_issue_retry_depth: int = 3
    max_source_retry_depth: int = 2
    max_no_progress_streak: int = 3
    per_message_body_max_chars: int = 16_384
    pad_max_chars: int = 32_768
    deterministic_clock: object | None = None
    random_seed: int | None = None
    human_mailbox_path: Path | None = None

    @classmethod
    def default(cls) -> "TeamConfig":
        return cls(
            role_budgets={
                "writer": RoleBudget(max_turns=5),
                "reviewer": RoleBudget(max_turns=6),
                "source_maintainer": RoleBudget(max_turns=8),
            }
        )


@dataclass(frozen=True)
class TeamStateView:
    turn_index_global: int
    turns_taken: dict[RoleName, int]
    remaining_turns: dict[RoleName, int]
    correlation_depth: dict[str, int]
    issue_retry_counts: dict[str, int]
    source_request_attempts: dict[str, int]
    total_messages: int
    started_at: str
    now: str


@dataclass(frozen=True)
class TerminalResult:
    reason: TerminalReason
    review_status: ReviewStatus
    final_state_snapshot_path: Path
    accept_message_id: str | None
    open_issue_refs: tuple[str, ...]
    notes: tuple[str, ...] = ()

    def to_jsonable(self) -> dict[str, object]:
        return {
            "reason": self.reason,
            "review_status": self.review_status,
            "final_state_snapshot_path": str(self.final_state_snapshot_path),
            "accept_message_id": self.accept_message_id,
            "open_issue_refs": list(self.open_issue_refs),
            "notes": list(self.notes),
        }


@dataclass
class TeamState:
    started_at: datetime
    turns_taken: dict[RoleName, int]
    no_progress_streak: dict[RoleName, int]
    correlation_depth: dict[str, int]
    correlation_alternations: dict[str, int]
    correlation_last_sender: dict[str, RoleName]
    issue_retry_counts: dict[str, int]
    source_request_attempts: dict[str, int]
    total_messages: int
    role_failed: dict[RoleName, str | None]
    terminal: TerminalResult | None = None
    deferred_issue_refs: set[str] = field(default_factory=set)

    @classmethod
    def fresh(cls, *, clock=None) -> "TeamState":
        now = (clock or utc_now)()
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return cls(
            started_at=now,
            turns_taken={role: 0 for role in TEAM_ROLES},
            no_progress_streak={role: 0 for role in TEAM_ROLES},
            correlation_depth={},
            correlation_alternations={},
            correlation_last_sender={},
            issue_retry_counts={},
            source_request_attempts={},
            total_messages=0,
            role_failed={role: None for role in TEAM_ROLES},
        )

    def view(self, *, now: datetime, role_budgets: dict[RoleName, RoleBudget]) -> TeamStateView:
        return TeamStateView(
            turn_index_global=sum(self.turns_taken.values()),
            turns_taken=dict(self.turns_taken),
            remaining_turns={
                role: max(0, role_budgets.get(role, RoleBudget(0)).max_turns - self.turns_taken.get(role, 0))
                for role in TEAM_ROLES
            },
            correlation_depth=dict(self.correlation_depth),
            issue_retry_counts=dict(self.issue_retry_counts),
            source_request_attempts=dict(self.source_request_attempts),
            total_messages=self.total_messages,
            started_at=iso_utc(self.started_at),
            now=iso_utc(now),
        )

    def snapshot_dict(self, *, now: datetime, role_budgets: dict[RoleName, RoleBudget]) -> dict[str, object]:
        view = self.view(now=now, role_budgets=role_budgets)
        return {
            "started_at": view.started_at,
            "now": view.now,
            "turns_taken": view.turns_taken,
            "remaining_turns": view.remaining_turns,
            "total_messages": self.total_messages,
            "correlation_depth": dict(self.correlation_depth),
            "correlation_alternations": dict(self.correlation_alternations),
            "issue_retry_counts": dict(self.issue_retry_counts),
            "source_request_attempts": dict(self.source_request_attempts),
            "no_progress_streak": dict(self.no_progress_streak),
            "role_failed": dict(self.role_failed),
            "deferred_issue_refs": sorted(self.deferred_issue_refs),
            "terminal": None if self.terminal is None else self.terminal.to_jsonable(),
        }

    def write_snapshot(self, path: Path, *, now: datetime, role_budgets: dict[RoleName, RoleBudget]) -> Path:
        from src.research_platform.team.mailbox import _atomic_write_text

        _atomic_write_text(path, json.dumps(self.snapshot_dict(now=now, role_budgets=role_budgets), indent=2, sort_keys=True))
        return path
