"""Render derived markdown sections from structured state sidecars."""

from __future__ import annotations

from src.research_platform.state.contracts import ConflictsFile, OpenQuestionsFile


def render_open_questions(open_questions: OpenQuestionsFile) -> str:
    if not open_questions.questions:
        return "No open questions recorded."
    lines: list[str] = []
    for question in open_questions.questions:
        marker = "x" if question.status == "answered" else " "
        suffix = f" — {question.status}"
        if question.priority != "normal":
            suffix += f" / {question.priority}"
        if question.evidence_keys:
            suffix += " / evidence " + ", ".join(f"[{key}]" for key in question.evidence_keys)
        lines.append(f"- [{marker}] {question.question_id} — {question.text}{suffix}")
        if question.proposed_answer:
            lines.append(f"  - Proposed answer: {question.proposed_answer}")
    return "\n".join(lines)


def render_conflicts(conflicts: ConflictsFile) -> str:
    if not conflicts.conflicts:
        return "No conflicts recorded."
    lines: list[str] = []
    for conflict in conflicts.conflicts:
        lines.append(
            f"- {conflict.conflict_id} — {conflict.summary} "
            f"([{conflict.left_evidence_key}] vs [{conflict.right_evidence_key}]) "
            f"— {conflict.status} / {conflict.severity}"
        )
        if conflict.rationale:
            lines.append(f"  - Rationale: {conflict.rationale}")
    return "\n".join(lines)
