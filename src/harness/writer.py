"""Draft writer for harness responses."""

from __future__ import annotations

from typing import cast

from langchain_core.messages import HumanMessage, SystemMessage

from src.harness.types import HarnessState
from src.shared.llm_manager import get_llm
from src.shared.model_config import get_model


def _content_to_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text_val = item.get("text")
                if isinstance(text_val, str):
                    parts.append(text_val)
        return "\n".join(parts)
    return str(content)


def _format_evidence(state: HarnessState, limit: int = 14) -> str:
    if not state.evidence_ledger:
        return "No evidence collected."

    chunks = []
    for item in state.evidence_ledger[-limit:]:
        source_text = ", ".join(item.sources or item.artifact_paths) or item.source_type
        content = (item.content or "").strip()
        if len(content) > 900:
            content = content[:900] + "..."
        chunks.append(f"[{item.id}] {item.summary}\nSource: {source_text}\n{content}")
    return "\n\n".join(chunks)


def _fallback_draft(state: HarnessState) -> str:
    lines = [
        f"# Response to: {state.request.user_prompt}",
        "",
        "## Answer",
    ]

    if state.revision_notes:
        lines.append(
            "The draft below was revised with verifier feedback in mind, but it may still need manual review."
        )
    elif not state.evidence_ledger:
        lines.append("Insufficient evidence was collected to produce a grounded answer.")
    else:
        lines.append("Grounded summary based on the evidence collected so far:")
        for item in state.evidence_ledger[-5:]:
            lines.append(f"- [{item.id}] {item.summary}")

    lines.extend(["", "## Sources"])
    if state.evidence_ledger:
        for item in state.evidence_ledger[-8:]:
            source_text = ", ".join(item.sources or item.artifact_paths) or item.source_type
            lines.append(f"- [{item.id}] {source_text}")
    else:
        lines.append("- No sources collected.")
    return "\n".join(lines)


def write_draft(state: HarnessState) -> str:
    """Write a markdown draft grounded in the current evidence ledger."""

    system_prompt = """
You are writing the final user-facing answer for a research harness.

Rules:
- Answer the user's actual question directly.
- Use only the supplied evidence.
- If a statement is an inference, label it clearly as an inference.
- Cite evidence IDs inline like [E1], [E2].
- Produce clean Markdown with concise sections.
- End with a short Sources section listing the evidence IDs you relied on.
"""

    user_prompt = f"""User request:
{state.request.user_prompt}

Revision notes:
{chr(10).join(state.revision_notes[-8:]) or "None"}

Recent working memory:
{chr(10).join(state.working_memory[-8:]) or "None"}

Evidence:
{_format_evidence(state)}
"""

    try:
        model_name = get_model("harness", "writer")
        llm = get_llm(model_name)
        response = llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        return _content_to_text(cast(object, response.content)).strip()
    except Exception as exc:
        print(f"Harness writer fallback triggered: {exc}")
        return _fallback_draft(state)
