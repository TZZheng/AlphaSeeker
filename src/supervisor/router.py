"""
Supervisor Router — Intent Classification & Agent Dispatch Logic.

This module contains the core routing logic for the Supervisor Agent.
It is intentionally decoupled from graph.py (which handles wiring) so that
the classification and dispatch logic can be tested independently.

Responsibilities:
  1. Classify a natural language prompt into EntityType(s) using a structured LLM call.
  2. Extract the parameters each sub-agent needs (ticker, macro topic, commodity asset).
  3. Map the list of required agents to concrete LangGraph node names.
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field
from src.supervisor.types import EntityType


# ---------------------------------------------------------------------------
# Classification Schema — structured output from the classification LLM
# ---------------------------------------------------------------------------

class AgentTask(BaseModel):
    """
    Parameters extracted for a single sub-agent.
    Each sub-agent type uses a different subset of these fields.
    """
    agent_type: str = Field(
        ...,
        description="Which agent handles this task. One of: 'equity', 'macro', 'commodity'."
    )
    ticker: str = Field(
        default="",
        description="Stock ticker symbol if agent_type is 'equity'. E.g. 'AAPL'."
    )
    topic: str = Field(
        default="",
        description="Free-form topic description if agent_type is 'macro'. E.g. 'US interest rates'."
    )
    asset: str = Field(
        default="",
        description="Commodity asset name if agent_type is 'commodity'. E.g. 'crude oil'."
    )


class ClassificationResult(BaseModel):
    """
    Structured output of the intent classification LLM call.
    Produced by classify_user_prompt() and consumed by the supervisor graph nodes.
    """
    primary_intent: str = Field(
        ...,
        description=(
            "The dominant topic type of the user prompt. "
            "One of: 'equity', 'macro', 'commodity'."
        )
    )
    tasks: List[AgentTask] = Field(
        ...,
        description=(
            "One AgentTask per sub-agent that needs to run. "
            "Single-topic prompts have one task; cross-domain prompts have multiple."
        )
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of why this classification was chosen (for debugging)."
    )


# ---------------------------------------------------------------------------
# Core Classification Function
# ---------------------------------------------------------------------------

def classify_user_prompt(prompt: str) -> ClassificationResult:
    """
    Calls the classification LLM with structured output to determine which
    sub-agents are needed and what parameters to pass them.

    This is the single LLM call that drives all routing decisions.
    Called by the `classify_intent` node in supervisor/graph.py.

    Args:
        prompt: Raw natural language user input.

    Returns:
        A ClassificationResult with primary_intent, tasks, and reasoning.

    Raises:
        ValueError: If the LLM returns an unrecognized agent_type.
    """
    ...


# ---------------------------------------------------------------------------
# Agent Dispatch Mapping
# ---------------------------------------------------------------------------

# Maps EntityType string → supervisor graph node name
AGENT_NODE_MAP: Dict[str, str] = {
    EntityType.EQUITY:    "run_equity_agent",
    EntityType.MACRO:     "run_macro_agent",
    EntityType.COMMODITY: "run_commodity_agent",
}


def get_agent_nodes(tasks: List[AgentTask]) -> List[str]:
    """
    Translates a list of AgentTask objects into the list of LangGraph node names
    that the supervisor graph's conditional edge should route to.

    This function is the bridge between the LLM's classification output and
    LangGraph's routing mechanism.

    Args:
        tasks: List of AgentTask objects from ClassificationResult.

    Returns:
        List of node name strings, e.g. ["run_equity_agent", "run_macro_agent"].
        LangGraph executes all of them in parallel if more than one is returned.

    Raises:
        ValueError: If a task contains an unrecognized agent_type.
    """
    ...


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_classification(result: ClassificationResult) -> None:
    """
    Validates that the classification result is internally consistent and
    all agent_type values are recognized.

    Raises:
        ValueError: With a descriptive message if any field is invalid.
    """
    ...
