"""Markdown-first durable company research state helpers."""

from src.research_platform.state.apply import ApplyResult, apply_proposals
from src.research_platform.state.contracts import (
    ConflictsFile,
    EvidenceIndex,
    OpenQuestionsFile,
    ProposalRecord,
    StateIndex,
    StateUpdateDecision,
    ValuationSnapshot,
)

__all__ = [
    "ApplyResult",
    "ConflictsFile",
    "EvidenceIndex",
    "OpenQuestionsFile",
    "ProposalRecord",
    "StateIndex",
    "StateUpdateDecision",
    "ValuationSnapshot",
    "apply_proposals",
]
