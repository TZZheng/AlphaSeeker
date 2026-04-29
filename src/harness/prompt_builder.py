"""Prompt bundle construction for harness agents and commenters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.harness.artifacts import (
    agent_workspace_paths,
    latest_agent_records,
    read_status,
    read_text,
    remaining_agent_seconds,
    remaining_run_seconds,
    write_text_atomic,
)
from src.harness.presets import PRESET_EXPLANATIONS, render_budget_snapshot
from src.harness.types import AGENT_PRESETS, HarnessRequest, SkillSpec


PROMPTS_ROOT = Path(__file__).with_name("prompts")
COMMENTER_TOOL_NAMES = ("read", "grep")


@dataclass(frozen=True)
class PromptSection:
    name: str
    scope: Literal["system", "user"]
    content: str
    protected: bool = True


@dataclass(frozen=True)
class PromptBundle:
    system_sections: list[PromptSection]
    user_sections: list[PromptSection]

    @property
    def system_prompt(self) -> str:
        return _join_sections(self.system_sections)

    @property
    def user_prompt(self) -> str:
        return _join_sections(self.user_sections)


def _join_sections(sections: list[PromptSection]) -> str:
    return "\n\n".join(section.content.strip() for section in sections if section.content.strip())


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


def _render_prompt_list(items: list[str]) -> str:
    return "\n".join(items) if items else "- None"


def _response_mode_prompt_text(response_mode: str) -> str:
    return _read_prompt_file(PROMPTS_ROOT / "response_modes" / f"{response_mode}.md")


def _preset_catalog_lines() -> list[str]:
    return [f"- `{preset}`: {PRESET_EXPLANATIONS[preset]}" for preset in AGENT_PRESETS]


def render_tools_markdown(
    *,
    preset: str,
    available_tools: list[str],
    available_skills: list[SkillSpec],
    response_mode: str = "native_tools",
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
            "response_mode": _response_mode_prompt_text(response_mode),
        },
    )


def render_task_markdown(user_prompt: str) -> str:
    template = _read_prompt_file(PROMPTS_ROOT / "task.md")
    return _render_prompt_template(
        template,
        {
            "user_prompt": user_prompt,
            "child_presets": _render_prompt_list(_preset_catalog_lines()),
        },
    )


def _render_commenter_interface_markdown(tool_names: list[str]) -> str:
    template = _read_prompt_file(PROMPTS_ROOT / "commenter_interface.md")
    return _render_prompt_template(
        template,
        {
            "commenter_tools": _render_prompt_list([f"- `{name}`" for name in tool_names]),
        },
    )


def _ensure_agent_state_prompt_files(run_root: str, agent_id: str) -> None:
    paths = agent_workspace_paths(run_root, agent_id)
    for key in ("prompt_memory", "history_summary"):
        path = paths[key]
        if not path.exists():
            write_text_atomic(path, "")


def sync_agent_tools_markdown(
    *,
    run_root: str,
    agent_id: str,
    preset: str,
    response_mode: str,
    available_tools: list[str],
    available_skills: list[SkillSpec],
) -> str:
    paths = agent_workspace_paths(run_root, agent_id)
    expected = render_tools_markdown(
        preset=preset,
        available_tools=available_tools,
        available_skills=available_skills,
        response_mode=response_mode,
    ).strip() + "\n"
    current = read_text(paths["tools"])
    if current != expected:
        write_text_atomic(paths["tools"], expected)
    _ensure_agent_state_prompt_files(run_root, agent_id)
    return expected.rstrip()


def _list_context_files(run_root: str, agent_id: str) -> str:
    context_root = agent_workspace_paths(run_root, agent_id)["context_root"]
    if not context_root.exists():
        return "None"
    files = [str(path) for path in sorted(context_root.rglob("*")) if path.is_file()]
    return "\n".join(f"- {name}" for name in files) or "None"


def _list_publish_files(run_root: str, agent_id: str) -> str:
    publish_root = agent_workspace_paths(run_root, agent_id)["publish_root"]
    if not publish_root.exists():
        return "None"
    rows: list[str] = []
    for path in sorted(publish_root.rglob("*")):
        if not path.is_file():
            continue
        first_line = ""
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line:
                first_line = line[:120]
                break
        rows.append(f"- {path.relative_to(publish_root).as_posix()}: {first_line}")
    return "\n".join(rows) or "None"


def _children_overview(run_root: str, agent_id: str) -> str:
    rows: list[str] = []
    for record in latest_agent_records(run_root).values():
        if record.parent_id != agent_id:
            continue
        child_paths = agent_workspace_paths(run_root, record.agent_id)
        has_summary = "yes" if child_paths["publish_summary"].exists() else "no"
        has_final = "yes" if child_paths["publish_final"].exists() else "no"
        rows.append(
            f"- {record.agent_id} [{record.status}] preset={record.preset} "
            f"desc={record.description} has_summary={has_summary} has_final={has_final} "
            f"summary={child_paths['publish_summary']} index={child_paths['publish_index']}"
        )
    return "\n".join(sorted(rows)) or "None"


def _budget_snapshot(request: HarnessRequest, run_root: str, agent_id: str) -> dict[str, int]:
    records = latest_agent_records(run_root)
    live_agents = sum(1 for record in records.values() if record.status in {"running", "waiting"})
    live_children = sum(
        1
        for record in records.values()
        if record.parent_id == agent_id and record.status in {"running", "waiting"}
    )
    return {
        "created_agents": len(records),
        "live_agents": live_agents,
        "remaining_agent_slots": max(0, request.max_agents_per_run - len(records)),
        "remaining_live_child_slots": max(0, request.max_live_children_per_parent - live_children),
        "remaining_run_seconds": remaining_run_seconds(request, run_root),
        "remaining_agent_seconds": remaining_agent_seconds(request, run_root, agent_id),
    }


def _agent_lineage(run_root: str, agent_id: str) -> tuple[int, str]:
    records = latest_agent_records(run_root)
    depth = 0
    chain: list[str] = []
    current = records.get(agent_id)
    seen: set[str] = set()
    while current is not None and current.agent_id not in seen:
        seen.add(current.agent_id)
        chain.append(f"{current.agent_id}:{current.preset}")
        if not current.parent_id:
            break
        depth += 1
        current = records.get(current.parent_id)
    chain.reverse()
    return depth, " -> ".join(chain) if chain else "None"


def _render_runtime_snapshot(
    *, request: HarnessRequest, run_root: str, agent_id: str, show_budget_time: bool = False,
) -> str:
    records = latest_agent_records(run_root)
    current = records.get(agent_id)
    depth, lineage = _agent_lineage(run_root, agent_id)
    budget_snapshot = _budget_snapshot(request, run_root, agent_id)
    sections = [
        "# Runtime Snapshot",
        "",
        "## Agent Identity",
        f"- agent_id: {agent_id}",
        f"- preset: {current.preset if current else 'unknown'}",
        f"- parent_id: {current.parent_id if current and current.parent_id else '-'}",
        f"- root_orchestrator: {'yes' if agent_id == 'agent_root' else 'no'}",
        "",
        render_budget_snapshot(request=request, snapshot=budget_snapshot, show_budget_time=show_budget_time),
        "",
        "## Agent Lineage",
        f"- depth: {depth}",
        f"- lineage: {lineage}",
        "",
        "## Current Status",
        read_status(run_root, agent_id),
        "",
        "## Context Files",
        _list_context_files(run_root, agent_id),
        "- Read listed context file paths with `read(path=...)`.",
        "",
        "## Published Files",
        _list_publish_files(run_root, agent_id),
        "",
        "## Children",
        _children_overview(run_root, agent_id),
    ]
    return "\n".join(sections)


def _render_runtime_history(
    *,
    request: HarnessRequest,
    run_root: str,
    agent_id: str,
    previous_error: str | None,
    comment_feed: str | None,
    soft_stop_active: bool,
) -> str:
    sections = ["# Runtime History / Delta"]
    has_content = False
    if previous_error:
        has_content = True
        sections.extend(["", "## Previous Error To Fix", previous_error])
    if comment_feed:
        has_content = True
        sections.extend(["", comment_feed])
    if not has_content:
        sections.extend(["", "- None"])
    if soft_stop_active:
        record = latest_agent_records(run_root).get(agent_id)
        is_root = record is None or not record.parent_id
        remaining_run = remaining_run_seconds(request, run_root)
        remaining_agent = remaining_agent_seconds(request, run_root, agent_id)
        sections.extend(
            [
                "",
                "## Remaining Time",
                f"- remaining run time: ~{remaining_run}s",
                f"- remaining agent time: ~{remaining_agent}s",
                "",
                "## Soft-Stop Mode",
                *_soft_stop_guidance_lines(is_root=is_root),
            ]
        )
    return "\n".join(sections)


def build_agent_runtime_delta_prompt(
    *,
    request: HarnessRequest,
    run_root: str,
    agent_id: str,
    previous_error: str | None,
    comment_feed: str | None,
    soft_stop_active: bool,
) -> str:
    sections = ["# Runtime Delta"]
    has_content = False
    if previous_error:
        has_content = True
        sections.extend(["", "## Previous Error To Fix", previous_error])
    if comment_feed:
        has_content = True
        sections.extend(["", comment_feed])
    if soft_stop_active:
        has_content = True
        record = latest_agent_records(run_root).get(agent_id)
        is_root = record is None or not record.parent_id
        remaining_run = remaining_run_seconds(request, run_root)
        remaining_agent = remaining_agent_seconds(request, run_root, agent_id)
        sections.extend(
            [
                "",
                "## Remaining Time",
                f"- remaining run time: ~{remaining_run}s",
                f"- remaining agent time: ~{remaining_agent}s",
                "",
                "## Soft-Stop Mode",
                *_soft_stop_guidance_lines(is_root=is_root),
            ]
        )
    return "\n".join(sections) if has_content else ""


def _soft_stop_guidance_lines(*, is_root: bool) -> list[str]:
    shared = [
        "Soft-stop mode is active. Treat this as your final model turn before shutdown.",
        "Classify any remaining gap before using another tool; broad new research, delegation, web search, and polish are no longer appropriate in soft-stop.",
        "Commenter suggestions are advisory; in soft-stop, extra depth, risk detail, and prose improvements are material or optional unless the deliverable is unusable without them.",
    ]
    completion_rules = [
        "Classify remaining gaps:",
        "  - blocking: the deliverable or handoff is unusable without this.",
        "  - material but caveatable: the deliverable is usable, but the gap should be documented as a limitation.",
        "  - optional: nice-to-have polish, extra checks, or depth that will not change the core answer.",
        "Use current materials to produce the required publish files; if publish/final.md already exists, do not inspect or patch it for polish.",
        "Only a blocking gap justifies more tool work in soft-stop; take the minimal write or patch action needed for required publish files.",
        "If a tool call for a material or optional improvement fails, do not retry or read more context; document the gap if possible, then call status(\"done\").",
        "Use status(\"blocked\") only when no usable deliverable or handoff can be written.",
    ]
    if is_root:
        return [
            *shared,
            "Use current materials to write or patch publish/final.md, publish/summary.md, and publish/artifact_index.md now.",
            *completion_rules,
        ]
    return [
        *shared,
        "Use current materials to write or patch your publish/summary.md and publish/artifact_index.md now so your parent can use them.",
        *completion_rules,
    ]


def build_agent_prompt_bundle(
    *,
    request: HarnessRequest,
    run_root: str,
    agent_id: str,
    preset: str,
    response_mode: str,
    available_tools: list[str],
    available_skills: list[SkillSpec],
    previous_error: str | None = None,
    comment_feed: str | None = None,
    soft_stop_active: bool = False,
    show_budget_time: bool = False,
) -> PromptBundle:
    tools_text = sync_agent_tools_markdown(
        run_root=run_root,
        agent_id=agent_id,
        preset=preset,
        response_mode=response_mode,
        available_tools=available_tools,
        available_skills=available_skills,
    )
    paths = agent_workspace_paths(run_root, agent_id)
    task_text = read_text(paths["task"]).strip() or "# Task Assignment\n\nNone"
    prompt_memory = read_text(paths["prompt_memory"]).strip()
    history_summary = read_text(paths["history_summary"]).strip()

    system_sections = [
        PromptSection("call_core", "system", _read_prompt_file(PROMPTS_ROOT / "system.md")),
        PromptSection("actor_base", "system", _read_prompt_file(PROMPTS_ROOT / "actors" / "agent_base.md")),
        PromptSection("role_contract", "system", _read_prompt_file(PROMPTS_ROOT / "roles" / f"{preset}.md")),
        PromptSection("runtime_interface", "system", tools_text),
    ]
    if prompt_memory:
        system_sections.append(
            PromptSection("prompt_memory", "system", f"# Prompt Memory\n\n{prompt_memory}")
        )
    if history_summary:
        system_sections.append(
            PromptSection("history_summary", "system", f"# Compacted History\n\n{history_summary}")
        )

    user_sections = [
        PromptSection("task_assignment", "user", task_text, protected=False),
        PromptSection(
            "runtime_snapshot",
            "user",
            _render_runtime_snapshot(request=request, run_root=run_root, agent_id=agent_id, show_budget_time=show_budget_time),
            protected=False,
        ),
        PromptSection(
            "runtime_history",
            "user",
            _render_runtime_history(
                request=request,
                run_root=run_root,
                agent_id=agent_id,
                previous_error=previous_error,
                comment_feed=comment_feed,
                soft_stop_active=soft_stop_active,
            ),
            protected=False,
        ),
    ]
    return PromptBundle(system_sections=system_sections, user_sections=user_sections)


def build_commenter_prompt_bundle(
    *,
    tools_available: bool,
    user_prompt: str,
) -> PromptBundle:
    tool_names = list(COMMENTER_TOOL_NAMES if tools_available else ())
    system_sections = [
        PromptSection("actor_base", "system", _read_prompt_file(PROMPTS_ROOT / "actors" / "commenter_base.md")),
        PromptSection("role_contract", "system", _read_prompt_file(PROMPTS_ROOT / "internal" / "commenter_review.md")),
        PromptSection("runtime_interface", "system", _render_commenter_interface_markdown(tool_names)),
    ]
    user_sections = [PromptSection("commenter_input", "user", user_prompt, protected=False)]
    return PromptBundle(system_sections=system_sections, user_sections=user_sections)
