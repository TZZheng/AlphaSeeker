"""Rule-based StateOwner policy for v3.0 proposals."""

from __future__ import annotations

import re

from src.research_platform.state.contracts import EvidenceIndex, ProposalRecord, StateUpdateDecision

DERIVED_SECTIONS = {"open_questions", "conflicts", "conflicts_uncertainty"}
NUMBERISH_RE = re.compile(r"(?<![A-Za-z])(?:\$\s*)?\d+(?:\.\d+)?\s*(?:%|x|billion|million|bn|mm|mb/d|boe/d)?", re.IGNORECASE)
CITED_SENTENCE_RE = re.compile(r"\[(S[1-9][0-9]*)\]")


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+", text) if part.strip()]


def quantitative_uncited_warnings(markdown: str) -> list[str]:
    warnings: list[str] = []
    for sentence in _sentences(markdown):
        if NUMBERISH_RE.search(sentence) and not CITED_SENTENCE_RE.search(sentence):
            warnings.append(f"quantitative-looking sentence lacks same-sentence cite: {sentence[:160]}")
    return warnings


def decide_proposal(
    proposal: ProposalRecord,
    *,
    evidence_index: EvidenceIndex,
    proposal_number: int = 1,
    soft_cap: int = 20,
    hard_cap: int = 50,
) -> StateUpdateDecision:
    """Return StateOwner decision for one proposal without mutating state."""

    warnings: list[str] = []
    if proposal_number > hard_cap:
        return StateUpdateDecision(
            proposal_id=proposal.proposal_id,
            decision="rejected",
            reason=f"proposal hard cap exceeded ({hard_cap})",
        )
    if proposal_number > soft_cap:
        warnings.append(f"proposal soft cap exceeded ({soft_cap})")

    if proposal.type in {"propose_section_update", "propose_close_question", "propose_conflict"} and not proposal.rationale.strip():
        return StateUpdateDecision(
            proposal_id=proposal.proposal_id,
            decision="rejected",
            reason="proposal rationale is required",
            warnings=warnings,
        )

    keys_to_check = list(proposal.evidence_keys) + list(proposal.source_keys)
    if proposal.left_evidence_key:
        keys_to_check.append(proposal.left_evidence_key)
    if proposal.right_evidence_key:
        keys_to_check.append(proposal.right_evidence_key)
    missing = [key for key in keys_to_check if key not in evidence_index.entries]
    if missing:
        return StateUpdateDecision(
            proposal_id=proposal.proposal_id,
            decision="rejected",
            reason="unresolved evidence keys: " + ", ".join(sorted(set(missing))),
            warnings=warnings,
        )

    if proposal.type == "propose_section_update":
        if proposal.section_key in DERIVED_SECTIONS:
            return StateUpdateDecision(
                proposal_id=proposal.proposal_id,
                decision="rejected",
                reason=f"derived section must use dedicated proposal type: {proposal.section_key}",
                warnings=warnings,
            )
        body = proposal.body_markdown or ""
        if proposal.action != "remove" and re.search(r"^##\s+", body, flags=re.MULTILINE):
            return StateUpdateDecision(
                proposal_id=proposal.proposal_id,
                decision="rejected",
                reason="section update body must not contain H2 headings",
                warnings=warnings,
            )
        warnings.extend(quantitative_uncited_warnings(body))
        if warnings:
            return StateUpdateDecision(
                proposal_id=proposal.proposal_id,
                decision="revised",
                reason="accepted with policy warnings",
                warnings=warnings,
                revised_proposal=proposal,
            )
        return StateUpdateDecision(
            proposal_id=proposal.proposal_id,
            decision="accepted",
            reason="section update passed policy",
            warnings=warnings,
        )

    if proposal.type == "propose_no_op":
        return StateUpdateDecision(
            proposal_id=proposal.proposal_id,
            decision="accepted",
            reason="no state change proposed",
            warnings=warnings,
        )

    return StateUpdateDecision(
        proposal_id=proposal.proposal_id,
        decision="accepted",
        reason=f"{proposal.type} passed policy",
        warnings=warnings,
    )
