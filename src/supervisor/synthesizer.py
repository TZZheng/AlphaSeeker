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
    return f"# Response to: {user_prompt}\n\n{report_md}"


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
    from src.shared.llm_manager import get_llm
    from src.shared.model_config import get_model
    from langchain_core.messages import SystemMessage, HumanMessage

    model_name = get_model("supervisor", "synthesize")
    llm = get_llm(model_name)

    primary_directive = ""
    if synthesis_input.primary_intent:
        primary_directive = f"The primary focus of this query is {synthesis_input.primary_intent.upper()}. Lead the report with insights from this domain."

    system_prompt = f"""You are the AlphaSeeker Supervisor Compiler.
The user asked: "{synthesis_input.user_prompt}"

You have received research reports from multiple specialized sub-agents.
Your job is to synthesize these into a single, cohesive, professional Markdown report that directly answers the user's question.

{primary_directive}

RULES:
1. Synthesize and integrate the insights. Do NOT just append the reports back-to-back.
2. Form a clear narrative that addresses the cross-domain nature of the prompt.
3. Preserve all important data points, numbers, and references from the sub-agents.
4. Keep the formatting clean with Headers, bullet points, and bold text for emphasis.
"""

    user_content = "Here are the sub-agent reports:\n\n"
    for agent_type, report in synthesis_input.agent_results.items():
        user_content += f"=== {agent_type.upper()} AGENT REPORT ===\n{report}\n\n"

    try:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content)
        ]
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        raise SynthesisError(f"Multi-agent synthesis failed: {e}")


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
    agents_used = list(synthesis_input.agent_results.keys())
    
    if not agents_used:
        return SynthesisOutput(
            final_response="No agents produced any results.",
            mode="error",
            agents_used=[]
        )
        
    if len(agents_used) == 1:
        agent_type = agents_used[0]
        report_md = synthesis_input.agent_results[agent_type]
        final_response = format_single_result(agent_type, report_md, synthesis_input.user_prompt)
        return SynthesisOutput(
            final_response=final_response,
            mode="passthrough",
            agents_used=agents_used
        )
    else:
        final_response = synthesize_multi_agent(synthesis_input)
        return SynthesisOutput(
            final_response=final_response,
            mode="synthesis",
            agents_used=agents_used
        )


# ---------------------------------------------------------------------------
# Error Type
# ---------------------------------------------------------------------------

class SynthesisError(Exception):
    """Raised when LLM synthesis fails after retries."""
    pass
