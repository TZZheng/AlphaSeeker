from __future__ import annotations

from src.harness.skills.core import CORE_SKILLS
from src.harness.tool_catalog import (
    harness_tool_definitions,
    tool_schema_properties,
    tool_specs_for_names,
)


def test_harness_base_tool_schema_is_available_by_name() -> None:
    specs = tool_specs_for_names(["write_file"])

    assert specs == [
        {
            "name": "write_file",
            "description": "Write one file under this agent's publish/ or scratch/ tree.",
            "input_schema": harness_tool_definitions()["write_file"]["input_schema"],
        }
    ]


def test_skill_specs_convert_compact_schema_to_json_schema() -> None:
    specs = tool_specs_for_names(["read_file", "search_in_files"], available_skills=CORE_SKILLS)
    by_name = {spec["name"]: spec for spec in specs}

    assert by_name["read_file"]["input_schema"]["properties"]["path"] == {"type": "string"}
    assert by_name["read_file"]["input_schema"]["properties"]["max_chars"] == {"type": "integer"}
    assert by_name["search_in_files"]["input_schema"]["properties"]["paths"] == {
        "type": "array",
        "items": {"type": "string"},
    }


def test_description_and_schema_overrides_apply_to_named_tools() -> None:
    specs = tool_specs_for_names(
        ["read_file"],
        available_skills=CORE_SKILLS,
        description_overrides={"read_file": "Read only files surfaced to the commenter."},
        schema_overrides={
            "read_file": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            }
        },
    )

    assert specs[0]["description"] == "Read only files surfaced to the commenter."
    assert specs[0]["input_schema"]["properties"] == {"path": {"type": "string"}}


def test_unknown_tool_names_are_skipped() -> None:
    specs = tool_specs_for_names(["write_file", "missing_tool"], available_skills=CORE_SKILLS)

    assert [spec["name"] for spec in specs] == ["write_file"]


def test_tool_schema_properties_accepts_nested_json_schema_fragments() -> None:
    properties = tool_schema_properties(
        {
            "path": "string",
            "tags": "string[]",
            "mode": {"type": "string", "enum": ["a", "b"]},
        }
    )

    assert properties == {
        "path": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "mode": {"type": "string", "enum": ["a", "b"]},
    }
