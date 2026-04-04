from __future__ import annotations

import pytest

from pydantic import ValidationError

from src.harness.types import AgentCommand, HarnessResponse


def test_agent_command_requires_known_shape() -> None:
    command = AgentCommand(tool="set_status", arguments={"status": "done"}, note="stop")

    assert command.tool == "set_status"
    assert command.arguments["status"] == "done"


def test_agent_command_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        AgentCommand.model_validate({"tool": "set_status", "arguments": {}, "extra": True})


def test_harness_response_uses_kernel_fields() -> None:
    response = HarnessResponse(
        status="completed",
        stop_reason="done",
        run_root="/tmp/run",
        root_agent_path="/tmp/run/agents/agent_root",
        final_report_path="/tmp/run/agents/agent_root/publish/final.md",
    )

    assert response.status == "completed"
    assert response.final_report_path.endswith("publish/final.md")
