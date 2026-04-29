"""Preset tool allowlists and visible-skill policies for the harness kernel."""

from __future__ import annotations

from src.harness.types import AGENT_PRESETS, HarnessRequest, SkillSpec


PRESET_TOOL_ALLOWLIST: dict[str, list[str]] = {
    "orchestrator": [
        "delegate",
        "agents",
        "bash",
        "write",
        "edit",
        "patch",
        "status",
    ],
    "research": [
        "delegate",
        "agents",
        "bash",
        "write",
        "edit",
        "patch",
        "status",
    ],
    "writer": [
        "delegate",
        "agents",
        "bash",
        "write",
        "edit",
        "patch",
        "status",
    ],
    "synthesizer": [
        "delegate",
        "agents",
        "bash",
        "write",
        "edit",
        "patch",
        "status",
    ],
    "evaluator": [
        "delegate",
        "agents",
        "bash",
        "write",
        "edit",
        "patch",
        "status",
    ],
}


PRESET_EXPLANATIONS: dict[str, str] = {
    "orchestrator": "delegate focused work, wait for child results, and synthesize the final answer",
    "research": "gather evidence, call deterministic research tools, and finish the assigned research task yourself",
    "writer": "draft or revise polished prose from published/context files",
    "synthesizer": "combine child outputs into a coherent published deliverable",
    "evaluator": "critique evidence quality, find contradictions, and publish concrete revision feedback",
}

AGENT_HIDDEN_SKILLS = {
    "retrieve_sources",
    "search_and_read",
    "read_artifact",
    "research_earnings_call",
    "analyze_peers",
}


def default_tool_allowlist(preset: str) -> list[str]:
    return list(PRESET_TOOL_ALLOWLIST.get(preset, PRESET_TOOL_ALLOWLIST["research"]))


def visible_skills_for_preset(
    *,
    preset: str,
    available_skills: list[SkillSpec],
) -> list[SkillSpec]:
    visible = [spec for spec in available_skills if spec.name not in AGENT_HIDDEN_SKILLS]
    primitive_core = [
        spec
        for spec in visible
        if spec.name
        in {"read", "grep", "get_current_datetime", "search_web", "search_news", "read_web_pages", "condense_context"}
    ]
    if preset == "research":
        return visible
    return primitive_core


def render_budget_snapshot(*, request: HarnessRequest, snapshot: dict[str, int]) -> str:
    lines = [
        "# Runtime Capacity Snapshot",
        "",
        f"- total agents created: {snapshot['created_agents']}/{request.max_agents_per_run}",
        f"- live agents: {snapshot['live_agents']}/{request.max_live_agents}",
        f"- remaining agent slots: {snapshot['remaining_agent_slots']}",
        f"- live children per parent cap: {request.max_live_children_per_parent}",
        f"- remaining live child slots: {snapshot['remaining_live_child_slots']}",
    ]
    remaining_run = snapshot.get("remaining_run_seconds")
    if remaining_run is not None:
        lines.append(f"- remaining run time: ~{remaining_run}s")
    remaining_agent = snapshot.get("remaining_agent_seconds")
    if remaining_agent is not None:
        lines.append(f"- remaining agent time: ~{remaining_agent}s")
    return "\n".join(lines)
