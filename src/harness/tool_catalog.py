"""Shared model-facing tool schemas for harness agents."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable

from src.harness.types import AGENT_PRESETS, SkillSpec


HARNESS_TOOL_NAMES = [
    "delegate",
    "agents",
    "files",
    "bash",
    "write",
    "edit",
    "patch",
    "status",
]

LEGAL_PRESET_LIST = ", ".join(f"'{preset}'" for preset in AGENT_PRESETS)

_TYPE_MAP = {
    "string": {"type": "string"},
    "integer": {"type": "integer"},
    "number": {"type": "number"},
    "boolean": {"type": "boolean"},
}


def tool_schema_properties(input_schema: dict[str, Any]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    for name, raw_type in input_schema.items():
        if isinstance(raw_type, dict) and "type" in raw_type:
            properties[name] = deepcopy(raw_type)
            continue
        type_name = str(raw_type).strip()
        if type_name.endswith("[]"):
            item_type = type_name[:-2]
            properties[name] = {
                "type": "array",
                "items": deepcopy(_TYPE_MAP.get(item_type, {"type": "string"})),
            }
            continue
        properties[name] = deepcopy(_TYPE_MAP.get(type_name, {"type": "string"}))
    return properties


def harness_tool_definitions() -> dict[str, dict[str, Any]]:
    return {
        "delegate": {
            "description": f"Launch one child agent for a narrower task. Legal preset values: {LEGAL_PRESET_LIST}.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "task_name": {"type": "string"},
                    "description": {"type": "string"},
                    "preset": {
                        "type": "string",
                        "enum": list(AGENT_PRESETS),
                        "description": f"One of: {LEGAL_PRESET_LIST}.",
                    },
                    "instructions": {"type": "string"},
                    "context_files": {"type": "array", "items": {"type": "string"}},
                    "expected_publish_files": {"type": "array", "items": {"type": "string"}},
                    "task_markdown": {"type": "string"},
                },
            },
        },
        "agents": {
            "description": "List all child agents with status. Drains the events queue so callers know which children just finished. Do not poll in a tight loop when nothing new has appeared.",
            "input_schema": {"type": "object", "properties": {}},
        },
        "files": {
            "description": "List published files for an agent.",
            "input_schema": {
                "type": "object",
                "properties": {"agent_id": {"type": "string"}},
            },
        },
        "bash": {
            "description": "Run one repo-scoped bash command from the allowlist for filesystem inspection or file movement.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "argv": {"type": "array", "items": {"type": "string"}},
                    "cwd": {"type": "string"},
                    "timeout_seconds": {"type": "integer"},
                    "max_output_chars": {"type": "integer"},
                },
            },
        },
        "write": {
            "description": "Write one file under this agent's publish/ or scratch/ tree.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
        "edit": {
            "description": "Apply one anchored text edit to a file under this agent's publish/ or scratch/ tree.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "operation": {
                        "type": "string",
                        "enum": ["replace", "insert_before", "insert_after", "append", "prepend"],
                    },
                    "target_text": {"type": "string"},
                    "content": {"type": "string"},
                    "occurrence": {"type": "integer"},
                    "replace_all": {"type": "boolean"},
                },
            },
        },
        "patch": {
            "description": "Apply one Codex-style single-file patch to an existing publish/ or scratch/ file. The patch string must use the exact markers '*** Begin Patch', one '*** Update File: ...' block, one or more '@@' hunks, and '*** End Patch'. In hunk lines the first character is the patch prefix; write '-### Heading' to remove '### Heading', not '- ### Heading' unless the target line really starts with a space.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "patch": {"type": "string"},
                },
                "required": ["patch"],
            },
        },
        "status": {
            "description": "Set this agent's status when it is ready to stop or block.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "error": {"type": "string"},
                },
            },
        },
    }


def _skill_tool_schema(spec: SkillSpec) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": tool_schema_properties(spec.input_schema),
    }


def tool_specs_for_names(
    names: Iterable[str],
    *,
    available_skills: Iterable[SkillSpec] | None = None,
    description_overrides: dict[str, str] | None = None,
    schema_overrides: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    base_tools = harness_tool_definitions()
    skills_by_name = {spec.name: spec for spec in available_skills or []}
    description_overrides = description_overrides or {}
    schema_overrides = schema_overrides or {}

    tools: list[dict[str, Any]] = []
    for name in names:
        if name in base_tools:
            description = str(base_tools[name]["description"])
            input_schema = deepcopy(base_tools[name]["input_schema"])
        elif name in skills_by_name:
            spec = skills_by_name[name]
            description = spec.description
            input_schema = _skill_tool_schema(spec)
        else:
            continue

        tools.append(
            {
                "name": name,
                "description": description_overrides.get(name, description),
                "input_schema": deepcopy(schema_overrides.get(name, input_schema)),
            }
        )
    return tools
