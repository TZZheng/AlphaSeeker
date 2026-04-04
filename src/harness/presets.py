"""Prompt presets and tool allowlists for the file-based harness kernel."""

from __future__ import annotations

from pathlib import Path

from src.harness.types import AGENT_PRESETS, HarnessRequest, SkillSpec


PRESET_TOOL_ALLOWLIST: dict[str, list[str]] = {
    "orchestrator": [
        "spawn_subagent",
        "list_children",
        "wait_children",
        "list_publish_files",
        "bash",
        "write_file",
        "edit_file",
        "set_status",
    ],
    "research": [
        "spawn_subagent",
        "list_children",
        "wait_children",
        "list_publish_files",
        "promote_artifact",
        "bash",
        "write_file",
        "edit_file",
        "set_status",
    ],
    "source_triage": [
        "spawn_subagent",
        "list_publish_files",
        "promote_artifact",
        "bash",
        "write_file",
        "edit_file",
        "set_status",
    ],
    "writer": [
        "spawn_subagent",
        "list_children",
        "wait_children",
        "list_publish_files",
        "bash",
        "write_file",
        "edit_file",
        "set_status",
    ],
    "synthesizer": [
        "spawn_subagent",
        "list_children",
        "wait_children",
        "list_publish_files",
        "bash",
        "write_file",
        "edit_file",
        "set_status",
    ],
    "evaluator": [
        "spawn_subagent",
        "list_children",
        "wait_children",
        "list_publish_files",
        "bash",
        "write_file",
        "edit_file",
        "set_status",
    ],
}


PRESET_EXPLANATIONS: dict[str, str] = {
    "orchestrator": "delegate focused work, wait for child results, and synthesize the final answer",
    "research": "gather evidence, call deterministic research tools, and finish the assigned research task yourself",
    "source_triage": "review noisy candidate material and separate useful sources from noise",
    "writer": "draft or revise polished prose from published/context files",
    "synthesizer": "combine child outputs into a coherent published deliverable",
    "evaluator": "critique evidence quality, find contradictions, and publish concrete revision feedback",
}

AGENT_HIDDEN_SKILLS = {
    "retrieve_sources",
    "search_web_resources",
    "search_and_read",
    "read_artifact",
    "research_earnings_call",
    "analyze_peers",
}
PROMPTS_ROOT = Path(__file__).with_name("prompts")


def _preset_catalog_lines() -> list[str]:
    return [f"- `{preset}`: {PRESET_EXPLANATIONS[preset]}" for preset in AGENT_PRESETS]


def _read_prompt_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Prompt file is empty: {path}")
    return text


def _render_prompt_template(template: str, replacements: dict[str, str]) -> str:
    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value.strip())
    return rendered


def _response_mode_prompt_text(response_mode: str) -> str:
    return _read_prompt_file(PROMPTS_ROOT / "response_modes" / f"{response_mode}.md")


def _render_prompt_list(items: list[str]) -> str:
    return "\n".join(items) if items else "- None"


def _system_prompt_text() -> str:
    return _read_prompt_file(PROMPTS_ROOT / "system.md")


def _role_prompt_text(preset: str) -> str:
    return _read_prompt_file(PROMPTS_ROOT / "roles" / f"{preset}.md")


def _environment_prompt_text(
    *,
    preset: str,
    response_mode: str,
    available_tools: list[str],
    available_skills: list[SkillSpec],
) -> str:
    template = _read_prompt_file(PROMPTS_ROOT / "environment.md")
    tool_lines = [f"- `{tool}`" for tool in available_tools]
    skill_lines = [f"- `{spec.name}` ({spec.pack}): {spec.description}" for spec in available_skills]
    replacements = {
        "preset": preset,
        "agent_tools": _render_prompt_list(tool_lines),
        "deterministic_skills": _render_prompt_list(skill_lines),
        "child_presets": _render_prompt_list(_preset_catalog_lines()),
        "response_mode": _response_mode_prompt_text(response_mode),
    }
    return _render_prompt_template(template, replacements)


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
        in {"read_file", "search_in_files", "get_current_datetime", "search_web", "search_news", "read_web_pages", "condense_context"}
    ]
    if preset == "research":
        return visible
    if preset == "source_triage":
        return [spec for spec in visible if spec.pack == "core"]
    return primitive_core


def render_tools_markdown(
    *,
    preset: str,
    available_tools: list[str],
    available_skills: list[SkillSpec],
) -> str:
    template = _read_prompt_file(PROMPTS_ROOT / "tools.md")
    skill_lines = [f"- `{spec.name}` ({spec.pack}): {spec.description}" for spec in available_skills]
    tool_lines = [f"- `{tool}`" for tool in available_tools]
    return _render_prompt_template(
        template,
        {
            "agent_tools": _render_prompt_list(tool_lines),
            "child_presets": _render_prompt_list(_preset_catalog_lines()),
            "deterministic_skills": _render_prompt_list(skill_lines),
        },
    )


def render_root_task_markdown(user_prompt: str) -> str:
    template = _read_prompt_file(PROMPTS_ROOT / "root_task.md")
    return _render_prompt_template(
        template,
        {
            "user_prompt": user_prompt,
            "child_presets": _render_prompt_list(_preset_catalog_lines()),
        },
    )


def render_budget_snapshot(*, request: HarnessRequest, snapshot: dict[str, int]) -> str:
    return "\n".join(
        [
            "# Runtime Capacity Snapshot",
            "",
            f"- total agents created: {snapshot['created_agents']}/{request.max_agents_per_run}",
            f"- live agents: {snapshot['live_agents']}/{request.max_live_agents}",
            f"- remaining agent slots: {snapshot['remaining_agent_slots']}",
            f"- live children per parent cap: {request.max_live_children_per_parent}",
            f"- remaining live child slots: {snapshot['remaining_live_child_slots']}",
        ]
    )


def preset_system_prompt(preset: str, *, response_mode: str = "native_tools") -> str:
    return preset_system_prompt_with_context(
        preset,
        response_mode=response_mode,
        available_tools=default_tool_allowlist(preset),
        available_skills=[],
    )


def preset_system_prompt_with_context(
    preset: str,
    *,
    response_mode: str = "native_tools",
    available_tools: list[str] | None = None,
    available_skills: list[SkillSpec] | None = None,
) -> str:
    tools = available_tools or default_tool_allowlist(preset)
    skills = available_skills or []
    return "\n\n".join(
        [
            _system_prompt_text(),
            _role_prompt_text(preset),
            _environment_prompt_text(
                preset=preset,
                response_mode=response_mode,
                available_tools=tools,
                available_skills=skills,
            ),
        ]
    )
