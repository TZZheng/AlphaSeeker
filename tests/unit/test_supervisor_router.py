from __future__ import annotations

import pytest

from src.supervisor.router import (
    AgentTask,
    ClassificationResult,
    get_agent_nodes,
    validate_classification,
)

pytestmark = pytest.mark.unit


def test_validate_classification_accepts_valid_multi_task_result() -> None:
    result = ClassificationResult(
        primary_intent="equity",
        tasks=[
            AgentTask(agent_type="equity", ticker="AAPL"),
            AgentTask(agent_type="macro", topic="US rates"),
        ],
        reasoning="cross-domain",
    )

    validate_classification(result)


def test_validate_classification_rejects_invalid_primary_intent() -> None:
    result = ClassificationResult(
        primary_intent="crypto",
        tasks=[AgentTask(agent_type="equity", ticker="AAPL")],
        reasoning="bad",
    )

    with pytest.raises(ValueError, match="Invalid primary_intent"):
        validate_classification(result)


def test_validate_classification_rejects_missing_required_task_field() -> None:
    result = ClassificationResult(
        primary_intent="equity",
        tasks=[AgentTask(agent_type="equity", ticker="")],
        reasoning="missing ticker",
    )

    with pytest.raises(ValueError, match="Equity task is missing a ticker"):
        validate_classification(result)


def test_get_agent_nodes_returns_ordered_node_list() -> None:
    tasks = [
        AgentTask(agent_type="commodity", asset="crude oil"),
        AgentTask(agent_type="macro", topic="US rates"),
    ]

    nodes = get_agent_nodes(tasks)

    assert nodes == ["run_commodity_agent", "run_macro_agent"]


def test_get_agent_nodes_rejects_unknown_agent_type() -> None:
    tasks = [AgentTask(agent_type="unknown", topic="x")]

    with pytest.raises(ValueError, match="Unrecognized agent_type"):
        get_agent_nodes(tasks)
