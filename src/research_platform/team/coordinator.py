from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from random import Random

from src.research_platform.roles.base import NoProgressNote, Role, RoleConfig, RoleContext, RoleTurnResult, TurnLogger
from src.research_platform.team.envelope import MessageEnvelope, RoleName
from src.research_platform.team.ids import new_correlation_id, seeded_rng, utc_now
from src.research_platform.team.mailbox import InboundMessage, Mailbox
from src.research_platform.team.pad import Pad
from src.research_platform.team.routing import URGENT_KINDS, is_allowed_route
from src.research_platform.team.state import TEAM_ROLES, ReviewStatus, RoleBudget, TeamConfig, TeamState, TeamStateView, TerminalReason, TerminalResult


@dataclass(frozen=True)
class CoordinatorStepResult:
    role: RoleName | None
    turn_index: int
    outbound_ids: tuple[str, ...]
    consumed_ids: tuple[str, ...]
    terminal: TerminalResult | None
    notes: tuple[str, ...] = ()


class NullTurnLogger(TurnLogger):
    pass


class Coordinator:
    def __init__(
        self,
        *,
        mailbox: Mailbox,
        roles: dict[RoleName, Role],
        config: TeamConfig,
        memo_dir: Path,
        role_docs: dict[RoleName, str],
        pads: dict[RoleName, Pad],
        logger: TurnLogger | None = None,
        clock: Callable[[], datetime] | None = None,
        rng: Random | None = None,
    ) -> None:
        self.mailbox = mailbox
        self.roles = roles
        self.config = config
        self.memo_dir = Path(memo_dir)
        self.role_docs = role_docs
        self.pads = pads
        self.logger = logger or NullTurnLogger()
        self._clock = clock or utc_now
        self._rng = rng or (seeded_rng(config.random_seed) if config.random_seed is not None else None)
        self.state = TeamState.fresh(clock=self._clock)
        self._round_robin_cursor = 0
        (self.memo_dir / "team").mkdir(parents=True, exist_ok=True)

    def kick_off(
        self,
        *,
        initial_to: RoleName = "writer",
        body_md: str,
        correlation_id: str | None = None,
        also_to_source_maintainer: bool = True,
    ) -> tuple[str, ...]:
        cid = correlation_id or new_correlation_id("root", clock=self._clock, rng=self._rng)
        sent: list[str] = []
        env = self.mailbox.send(
            from_role="coordinator",
            to_role=initial_to,
            kind="kick_off",
            body_md=body_md,
            correlation_id=cid,
            turn_index=0,
            clock=self._clock,
            rng=self._rng,
        )
        sent.append(env.message_id)
        self._record_envelope(env, body_md)
        if also_to_source_maintainer and initial_to != "source_maintainer":
            env2 = self.mailbox.send(
                from_role="coordinator",
                to_role="source_maintainer",
                kind="kick_off",
                body_md=body_md,
                correlation_id=cid,
                turn_index=0,
                clock=self._clock,
                rng=self._rng,
            )
            sent.append(env2.message_id)
            self._record_envelope(env2, body_md)
        self.snapshot()
        return tuple(sent)

    def step(self) -> CoordinatorStepResult:
        if self.state.terminal is not None:
            return CoordinatorStepResult(
                role=None,
                turn_index=sum(self.state.turns_taken.values()),
                outbound_ids=(),
                consumed_ids=(),
                terminal=self.state.terminal,
                notes=("already_terminal",),
            )
        pre_terminal = self._pre_step_terminal()
        if pre_terminal is not None:
            return pre_terminal

        role_name = self._pick_next()
        if role_name is None:
            return self._terminate("give_up_budget_exhausted", note="no eligible role with inbox")

        role = self.roles[role_name]
        inbox = tuple(self.mailbox.inbox_for(role_name))
        ctx = self._build_role_context(role_name)
        try:
            result = role.take_turn(ctx, inbox)
        except Exception as exc:
            self.state.role_failed[role_name] = repr(exc)
            return self._terminate("give_up_role_failed", note=f"{role_name}: {exc}")

        err = self._validate_result(role_name, result)
        if err is not None:
            self.state.role_failed[role_name] = err
            return self._terminate("give_up_role_failed", note=err)

        for append in result.pad_appends:
            self.pads[role_name].append(append.title, append.body)
        if result.no_progress and result.no_progress.pad_note:
            self.pads[role_name].append("no-progress", result.no_progress.pad_note)

        outbound_ids = self._send_outbound(role_name, result, inbox)
        consumed_ids = tuple(message.envelope.message_id for message in inbox)
        for message_id in consumed_ids:
            self.mailbox.mark_consumed(message_id, role_name)

        self.state.turns_taken[role_name] = self.state.turns_taken.get(role_name, 0) + 1
        if result.is_no_progress:
            self.state.no_progress_streak[role_name] = self.state.no_progress_streak.get(role_name, 0) + 1
        else:
            self.state.no_progress_streak[role_name] = 0

        self._check_post_step_terminals(outbound_ids)
        snap_path = self.snapshot()
        return CoordinatorStepResult(
            role=role_name,
            turn_index=self.state.turns_taken[role_name],
            outbound_ids=outbound_ids,
            consumed_ids=consumed_ids,
            terminal=self.state.terminal,
            notes=(f"snapshot={snap_path.name}",),
        )

    def run_until_terminal(self, *, max_steps: int | None = None) -> TerminalResult:
        steps = 0
        while self.state.terminal is None:
            if max_steps is not None and steps >= max_steps:
                self._terminate("give_up_budget_exhausted", note="max_steps reached")
                break
            self.step()
            steps += 1
        assert self.state.terminal is not None
        return self.state.terminal

    def terminal(self) -> TerminalResult | None:
        return self.state.terminal

    def state_view(self) -> TeamStateView:
        return self.state.view(now=self._clock(), role_budgets=self.config.role_budgets)

    def snapshot(self) -> Path:
        return self.state.write_snapshot(
            self.memo_dir / "team" / "team_state.json",
            now=self._clock(),
            role_budgets=self.config.role_budgets,
        )

    def _pre_step_terminal(self) -> CoordinatorStepResult | None:
        now = self._clock()
        started = self.state.started_at
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        if (now - started).total_seconds() > self.config.wall_clock_budget_s:
            return self._terminate("give_up_wall_clock")
        if self.state.total_messages >= self.config.total_messages_cap:
            return self._terminate("give_up_total_messages_cap")
        return None

    def _eligible_roles(self) -> list[RoleName]:
        eligible: list[RoleName] = []
        for name in self.roles:
            if name not in TEAM_ROLES:
                continue
            if self.state.role_failed.get(name):
                continue
            budget = self.config.role_budgets.get(name, RoleBudget(0))
            if self.state.turns_taken.get(name, 0) >= budget.max_turns:
                continue
            if not self.mailbox.inbox_for(name):
                continue
            eligible.append(name)
        return eligible

    def _pick_next(self) -> RoleName | None:
        eligible = self._eligible_roles()
        if not eligible:
            return None

        ordered_roles = list(TEAM_ROLES)

        def sort_key(role: RoleName) -> tuple[int, int, str, int]:
            inbox = self.mailbox.inbox_for(role)
            urgent_count = sum(1 for message in inbox if message.envelope.kind in URGENT_KINDS)
            blocked = self._count_correlations_blocked_on(role)
            oldest = min(message.envelope.message_id for message in inbox)
            rr = (ordered_roles.index(role) - self._round_robin_cursor) % len(ordered_roles)
            return (-urgent_count, -blocked, oldest, rr)

        eligible.sort(key=sort_key)
        picked = eligible[0]
        self._round_robin_cursor = (ordered_roles.index(picked) + 1) % len(ordered_roles)
        return picked

    def _count_correlations_blocked_on(self, role: RoleName) -> int:
        return len({message.envelope.correlation_id for message in self.mailbox.inbox_for(role)})

    def _build_role_context(self, role_name: RoleName) -> RoleContext:
        return RoleContext(
            role_name=role_name,
            role_doc=self.role_docs.get(role_name, ""),
            pad=self.pads[role_name],
            mailbox=self.mailbox,
            memo_dir=self.memo_dir,
            team_state=self.state_view(),
            config=RoleConfig(role_name=role_name),
            logger=self.logger,
            turn_index=self.state.turns_taken.get(role_name, 0),
        )

    def _validate_result(self, role_name: RoleName, result: RoleTurnResult) -> str | None:
        if not result.outbound and result.no_progress is None:
            return f"{role_name} returned a silent turn"
        if len(result.outbound) > 4:
            return f"{role_name} emitted too many outbound messages"
        for outbound in result.outbound:
            if len(outbound.body_md) > self.config.per_message_body_max_chars:
                return f"{role_name} emitted oversized message body"
            if not is_allowed_route(role_name, outbound.to_role, outbound.kind):
                return f"disallowed route: {role_name}->{outbound.to_role} for kind={outbound.kind}"
        if result.no_progress and len(result.no_progress.reason) > 240:
            return f"{role_name} no-progress reason too long"
        return None

    def _send_outbound(self, role_name: RoleName, result: RoleTurnResult, inbox: tuple[InboundMessage, ...]) -> tuple[str, ...]:
        ids: list[str] = []
        default_cid = max(inbox, key=lambda m: m.envelope.message_id).envelope.correlation_id if inbox else None
        for outbound in result.outbound:
            cid = outbound.correlation_id or default_cid or new_correlation_id("c", clock=self._clock, rng=self._rng)
            env = self.mailbox.send(
                from_role=role_name,
                to_role=outbound.to_role,
                kind=outbound.kind,
                body_md=outbound.body_md,
                correlation_id=cid,
                refs=outbound.refs,
                turn_index=self.state.turns_taken.get(role_name, 0),
                cost_hint=outbound.cost_hint,
                clock=self._clock,
                rng=self._rng,
            )
            ids.append(env.message_id)
            self._record_envelope(env, outbound.body_md)
        return tuple(ids)

    def _record_envelope(self, env: MessageEnvelope, body_md: str) -> None:
        self.state.total_messages += 1
        cid = env.correlation_id
        self.state.correlation_depth[cid] = self.state.correlation_depth.get(cid, 0) + 1
        prev = self.state.correlation_last_sender.get(cid)
        if prev and prev != env.from_role and {prev, env.from_role} == {"writer", "reviewer"}:
            self.state.correlation_alternations[cid] = self.state.correlation_alternations.get(cid, 0) + 1
        self.state.correlation_last_sender[cid] = env.from_role
        for ref in env.refs:
            if ref.startswith("I-"):
                self.state.issue_retry_counts[ref] = self.state.issue_retry_counts.get(ref, 0) + 1
        if env.kind == "source_request":
            sig = self._source_signature(env, body_md)
            self.state.source_request_attempts[sig] = self.state.source_request_attempts.get(sig, 0) + 1
        if env.kind == "defer":
            for ref in env.refs:
                if ref.startswith("I-"):
                    self.state.deferred_issue_refs.add(ref)

    def _source_signature(self, env: MessageEnvelope, body_md: str) -> str:
        seed = "|".join(env.refs) if env.refs else " ".join(body_md.lower().split())
        return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]

    def _check_post_step_terminals(self, outbound_ids: tuple[str, ...]) -> None:
        envelopes = [self.mailbox.get(mid).envelope for mid in outbound_ids]
        accept = next((env for env in envelopes if env.kind == "accept"), None)
        if accept is not None:
            self._terminate(
                "accepted",
                review_status="accepted_with_caveats" if self.state.deferred_issue_refs else "accepted",
                accept_message_id=accept.message_id,
                open_issue_refs=tuple(sorted(self.state.deferred_issue_refs)),
            )
            return
        escalate = next((env for env in envelopes if env.kind == "escalate"), None)
        if escalate is not None:
            if self.config.human_mailbox_path is None:
                self._terminate("failed_unrecoverable", note="escalate without human path")
                return
            human_mailbox = Mailbox(self.config.human_mailbox_path)
            human_mailbox.send(
                from_role="coordinator",
                to_role="human",
                kind="escalate",
                body_md=self.mailbox.get(escalate.message_id).body_md,
                correlation_id=escalate.correlation_id,
                refs=escalate.refs,
                turn_index=0,
                clock=self._clock,
                rng=self._rng,
            )
        for cid, alternations in list(self.state.correlation_alternations.items()):
            if alternations > self.config.max_dissent_depth:
                self._inject_defer(cid, note=f"correlation exceeded max dissent depth {self.config.max_dissent_depth}")
        for issue, count in list(self.state.issue_retry_counts.items()):
            if count > self.config.max_issue_retry_depth:
                self._inject_defer_for_issue(issue)
        for sig, count in list(self.state.source_request_attempts.items()):
            if count > self.config.max_source_retry_depth:
                self._inject_source_unavailable(sig)
        for role, streak in self.state.no_progress_streak.items():
            if streak >= self.config.max_no_progress_streak:
                self._terminate("stuck_escalated", note=f"{role} no-progress streak")
                return

    def _inject_defer(self, correlation_id: str, *, note: str) -> None:
        for to_role in ("writer", "reviewer"):
            env = self.mailbox.send(
                from_role="coordinator",
                to_role=to_role,
                kind="defer",
                body_md=f"Coordinator-forced defer: {note}",
                correlation_id=correlation_id,
                turn_index=0,
                clock=self._clock,
                rng=self._rng,
            )
            self._record_envelope(env, f"Coordinator-forced defer: {note}")

    def _inject_defer_for_issue(self, issue: str) -> None:
        cid = new_correlation_id("issue", clock=self._clock, rng=self._rng)
        self.state.deferred_issue_refs.add(issue)
        for to_role in ("writer", "reviewer"):
            body = f"Coordinator-forced defer: issue {issue} exceeded retry depth {self.config.max_issue_retry_depth}."
            env = self.mailbox.send(
                from_role="coordinator",
                to_role=to_role,
                kind="defer",
                body_md=body,
                correlation_id=cid,
                refs=(issue,),
                turn_index=0,
                clock=self._clock,
                rng=self._rng,
            )
            self._record_envelope(env, body)

    def _inject_source_unavailable(self, sig: str) -> None:
        cid = new_correlation_id("source", clock=self._clock, rng=self._rng)
        body = f"Coordinator-forced source_unavailable: source request signature {sig} exceeded retry depth."
        for to_role in ("writer", "reviewer"):
            env = self.mailbox.send(
                from_role="coordinator",
                to_role=to_role,
                kind="source_unavailable",
                body_md=body,
                correlation_id=cid,
                refs=(sig,),
                turn_index=0,
                clock=self._clock,
                rng=self._rng,
            )
            self._record_envelope(env, body)

    def _terminate(
        self,
        reason: TerminalReason,
        *,
        review_status: ReviewStatus | None = None,
        accept_message_id: str | None = None,
        open_issue_refs: tuple[str, ...] = (),
        note: str | None = None,
    ) -> CoordinatorStepResult:
        if review_status is None:
            if reason == "accepted":
                review_status = "accepted_with_caveats" if open_issue_refs else "accepted"
            elif reason == "failed_unrecoverable":
                review_status = "failed_unrecoverable"
            elif reason == "stuck_escalated":
                review_status = "stuck_escalated"
            else:
                review_status = "revise_exhausted"
        snap_path = self.memo_dir / "team" / "team_state.json"
        terminal = TerminalResult(
            reason=reason,
            review_status=review_status,
            final_state_snapshot_path=snap_path,
            accept_message_id=accept_message_id,
            open_issue_refs=open_issue_refs,
            notes=(() if note is None else (note,)),
        )
        self.state.terminal = terminal
        self.snapshot()
        return CoordinatorStepResult(
            role=None,
            turn_index=sum(self.state.turns_taken.values()),
            outbound_ids=(),
            consumed_ids=(),
            terminal=terminal,
            notes=terminal.notes,
        )
