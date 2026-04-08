from __future__ import annotations

from src.cli.screens.dashboard import _assistant_response_lines


def test_assistant_response_lines_render_text_and_tool_call_for_same_turn() -> None:
    entry = {
        "kind": "assistant_response",
        "turn_index": 6,
        "created_at": "2026-04-08T01:52:29Z",
        "message": {
            "content": [
                {"type": "text", "text": "Still running.\n"},
                {"type": "tool_use", "name": "list_children", "input": {}},
            ]
        },
        "decision": {"tool_calls": [{"name": "list_children", "arguments": {}}]},
    }

    llm_lines, thinking_lines = _assistant_response_lines("agent_root", entry)

    assert len(llm_lines) == 2
    assert "Still running." in llm_lines[0]
    assert "→ list_children" in llm_lines[1]
    assert thinking_lines == []


def test_assistant_response_lines_render_tool_only_turn() -> None:
    entry = {
        "kind": "assistant_response",
        "turn_index": 3,
        "created_at": "2026-04-08T01:52:29Z",
        "message": {
            "content": [
                {"type": "tool_use", "name": "list_children", "input": {}},
            ]
        },
        "decision": {"tool_calls": [{"name": "list_children", "arguments": {}}]},
    }

    llm_lines, _thinking_lines = _assistant_response_lines("agent_root", entry)

    assert llm_lines == [
        "[dim cyan]agent_root[/dim cyan] [dim]turn 3  2026-04-08T01:52:29[/dim]\n  [dim]→ list_children[/dim]"
    ]


def test_assistant_response_lines_show_bash_command_preview() -> None:
    entry = {
        "kind": "assistant_response",
        "turn_index": 3,
        "created_at": "2026-04-08T01:52:29Z",
        "message": {"content": []},
        "decision": {"tool_calls": [{"name": "bash", "arguments": {"argv": ["sleep", "20"]}}]},
    }

    llm_lines, _thinking_lines = _assistant_response_lines("agent_root", entry)

    assert "→ bash sleep 20" in llm_lines[0]
