"""Skill-pack selection for the harness."""

from __future__ import annotations

from typing import Callable

from src.supervisor.router import ClassificationResult, classify_user_prompt


PACK_ORDER = ["core", "equity", "macro", "commodity"]
_DOMAIN_PACKS = {"equity", "macro", "commodity"}


def select_packs(
    prompt: str,
    classify_fn: Callable[[str], ClassificationResult] | None = None,
) -> list[str]:
    """Select enabled packs from the existing supervisor classifier."""

    packs = ["core"]
    fn = classify_fn or classify_user_prompt

    try:
        classification = fn(prompt)
        seen = set(packs)
        for task in classification.tasks:
            agent_type = task.agent_type.strip().lower()
            if agent_type in _DOMAIN_PACKS and agent_type not in seen:
                packs.append(agent_type)
                seen.add(agent_type)
        if len(packs) == 1 and classification.primary_intent in _DOMAIN_PACKS:
            packs.append(classification.primary_intent)
        return [pack for pack in PACK_ORDER if pack in packs]
    except Exception as exc:
        print(f"Harness selector fallback triggered: {exc}")
        return PACK_ORDER.copy()
