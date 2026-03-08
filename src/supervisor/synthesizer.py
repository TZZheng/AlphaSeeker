"""
Supervisor Synthesizer — Final Response Composition.

This module is responsible for the last mile of the Supervisor pipeline:
taking the Markdown output(s) from one or more sub-agents and producing
a single, coherent final response that directly answers the user's original question.

Two modes of operation:
  1. Single-agent passthrough: If only one sub-agent ran, the synthesizer
     acts as a lightweight formatter (adds framing, no LLM call needed).
  2. Multi-agent synthesis: If multiple sub-agents ran, an LLM integrates
     all agent outputs into a unified response that directly addresses the
     user's cross-domain question.

Integration contract:
  - INPUT:  agent_results: Dict[str, str]  — each value is a full Markdown string
            from a sub-agent (equity report, macro brief, commodity brief, etc.)
  - OUTPUT: final_response: str            — a single Markdown string for the CLI
"""

from typing import Dict, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Model Assignment
# ---------------------------------------------------------------------------

MODEL_SYNTHESIZE = "kimi-k2.5"   # High-quality writing for multi-agent integration


# ---------------------------------------------------------------------------
# Data Schema
# ---------------------------------------------------------------------------

class SynthesisInput(BaseModel):
    """
    Encapsulates everything the synthesizer needs. Passed to both
    format_single_result() and synthesize_multi_agent().
    """
    user_prompt: str = Field(
        ...,
        description="The original natural language question from the user. "
                    "The final response should directly answer this."
    )
    agent_results: Dict[str, str] = Field(
        ...,
        description="Keyed by agent name ('equity', 'macro', 'commodity'). "
                    "Each value is a full Markdown string from that sub-agent."
    )
    primary_intent: Optional[str] = Field(
        default=None,
        description="The dominant EntityType (e.g. 'equity'). Used to decide "
                    "which agent's output to lead with in multi-agent synthesis."
    )


class SynthesisOutput(BaseModel):
    """
    Wraps the final synthesized response and any metadata about how it was produced.
    """
    final_response: str = Field(
        ...,
        description="The complete Markdown response to deliver to the user."
    )
    mode: str = Field(
        ...,
        description="'passthrough' if only one agent ran, 'synthesis' if multiple agents ran."
    )
    agents_used: list[str] = Field(
        ...,
        description="List of agent names whose outputs contributed to this response."
    )


# ---------------------------------------------------------------------------
# Core Functions
# ---------------------------------------------------------------------------

def format_single_result(
    agent_type: str,
    report_md: str,
    user_prompt: str,
) -> str:
    """
    Passthrough formatter for the single-agent case.
    No LLM call — simply adds a brief header acknowledging the user's question
    and returns the sub-agent's Markdown report as-is.

    Called when len(agent_results) == 1.

    Args:
        agent_type:  Which agent produced the output (e.g. "equity").
        report_md:   The full Markdown string from the sub-agent.
        user_prompt: The original user question (used to build the header).

    Returns:
        A Markdown string ready for CLI output.
    """
    ...


def synthesize_multi_agent(synthesis_input: SynthesisInput) -> str:
    """
    LLM-based synthesis for the multi-agent case.
    Combines outputs from two or more sub-agents into a unified Markdown
    response that directly addresses the user's cross-domain question.

    The LLM is given:
      - The original user_prompt (so it knows what question to answer)
      - Each agent's Markdown output, labelled by agent type
      - The primary_intent (so it knows which dimension to lead with)

    Called when len(agent_results) >= 2.

    Args:
        synthesis_input: A SynthesisInput containing user_prompt, agent_results,
                         and primary_intent.

    Returns:
        A unified Markdown string directly answering the user's question.

    Raises:
        SynthesisError: If the LLM call fails.
    """
    ...


def run_synthesis(synthesis_input: SynthesisInput) -> SynthesisOutput:
    """
    Entry point for the synthesize_results node in supervisor/graph.py.
    Dispatches to format_single_result() or synthesize_multi_agent()
    based on how many agents produced output.

    Args:
        synthesis_input: A SynthesisInput assembled by the synthesize_results node.

    Returns:
        A SynthesisOutput with the final_response, mode, and agents_used.
    """
    ...


# ---------------------------------------------------------------------------
# Error Type
# ---------------------------------------------------------------------------

class SynthesisError(Exception):
    """Raised when LLM synthesis fails after retries."""
    pass
