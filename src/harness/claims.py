"""Claim-map construction for harness reports."""

from __future__ import annotations

import re

from src.harness.types import ClaimRecord, HarnessState


_INFERENCE_MARKERS = ("likely", "may", "could", "suggests", "appears", "we infer", "inference")


def _parse_sections(draft: str) -> list[tuple[str, str]]:
    current_section = "Summary"
    buffer: list[str] = []
    sections: list[tuple[str, str]] = []
    for line in draft.splitlines():
        if line.startswith("## "):
            if buffer:
                sections.append((current_section, "\n".join(buffer).strip()))
                buffer = []
            current_section = line[3:].strip()
            continue
        buffer.append(line)
    if buffer:
        sections.append((current_section, "\n".join(buffer).strip()))
    return sections


def build_claim_map(state: HarnessState, draft: str) -> list[ClaimRecord]:
    """Convert report sentences into normalized fact/inference claims."""

    supporting_pool = [item.id for item in state.evidence_ledger if item.id]
    complicating_pool = [
        item.id
        for item in state.evidence_ledger
        if item.id
        and (
            "counter" in item.summary.lower()
            or "risk" in item.summary.lower()
            or "peer" in item.summary.lower()
            or "competitor" in item.summary.lower()
            or item.metadata.get("counterevidence") is True
        )
    ]

    claims: list[ClaimRecord] = []
    claim_index = 1
    for section_label, section_text in _parse_sections(draft):
        sentences = re.split(r"(?<=[.!?])\s+", section_text)
        for sentence in sentences:
            text = sentence.strip()
            if not text or len(text) < 20:
                continue
            cited_ids = re.findall(r"\[(E\d+)\]", text)
            if not cited_ids:
                continue
            lowered = text.lower()
            claim_type = "inference" if any(marker in lowered for marker in _INFERENCE_MARKERS) else "fact"
            freshness_date = None
            match = re.search(r"\b(20\d{2})\b", text)
            if match:
                freshness_date = match.group(1)
            elif state.evidence_ledger:
                freshness_date = state.evidence_ledger[-1].created_at[:10]

            if section_label.lower() in {"risks and counterevidence", "peer and competitive pressure"}:
                complicating_ids = cited_ids
            else:
                complicating_ids = complicating_pool[:2]

            claims.append(
                ClaimRecord(
                    id=f"C{claim_index}",
                    claim_text=text,
                    claim_type=claim_type,
                    section_label=section_label,
                    supporting_evidence_ids=cited_ids or supporting_pool[:2],
                    complicating_evidence_ids=complicating_ids,
                    freshness_date=freshness_date,
                    appears_in_final_report=True,
                )
            )
            claim_index += 1

    return claims
