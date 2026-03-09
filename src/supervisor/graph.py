"""
Supervisor LangGraph workflow — Multi-Agent Orchestrator.

Flow:
  classify_intent → route_to_agents (Send-based parallel fan-out)
    → [run_equity_agent | run_macro_agent | run_commodity_agent] (run in parallel via Send)
    → synthesize_results → END

The Supervisor does NOT fetch data itself. It delegates entirely to Sub-Agents,
then merges their outputs into a single coherent response.

Parallel fan-out uses the LangGraph `Send` API (available since langgraph >=0.2).
Each Send dispatches an independent copy of the state to a sub-agent node;
all active sub-agents run concurrently and converge at synthesize_results.
"""

from typing import TypedDict, Optional, List, Literal, Dict, Any, Annotated
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from src.supervisor.types import EntityType
from src.supervisor.router import (
    ClassificationResult,
    AgentTask,
    classify_user_prompt,
    get_agent_nodes,
    validate_classification,
    AGENT_NODE_MAP,
)
from src.shared.schemas import SubAgentRequest, SubAgentResponse
from src.supervisor.synthesizer import SynthesisInput, run_synthesis


# ---------------------------------------------------------------------------
# Supervisor State — carries data between all nodes in this graph
# ---------------------------------------------------------------------------

class SupervisorState(TypedDict):
    # Input
    user_prompt: str                              # Raw natural language input from the user

    # Intent Classification (populated by classify_intent node)
    intent: Optional[str]                                    # Dominant EntityType, e.g. "equity"
    sub_agents_needed: Optional[List[str]]                   # e.g. ["equity", "macro"]
    classified_entities: Optional[Dict[str, SubAgentRequest]] # One SubAgentRequest per agent

    # Sub-Agent Outputs — keyed by agent name, each value is a Markdown string.
    # Uses operator.ior (|=) as the merge reducer so parallel Send branches
    # merge their dicts safely: {"equity": "..."} | {"macro": "..."} = both keys.
    agent_results: Annotated[Dict[str, str], operator.ior]

    # Final Output
    final_response: Optional[str]                 # Synthesized, integrated response for the user
    error: Optional[str]


# ---------------------------------------------------------------------------
# Model Assignments
# ---------------------------------------------------------------------------

from src.shared.model_config import get_model

MODEL_CLASSIFY   = get_model("supervisor", "classify")
MODEL_SYNTHESIZE = get_model("supervisor", "synthesize")


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def classify_intent(state: SupervisorState) -> dict:
    """
    Classifies the user prompt into one or more EntityTypes and builds a
    typed SubAgentRequest for each sub-agent that needs to run.

    Delegates to `router.classify_user_prompt()` for the LLM call, then
    `router.validate_classification()` for safety.

    For example:
      - "Analyze AAPL"
            → intent=EQUITY, sub_agents_needed=["equity"],
              classified_entities={
                  "equity": SubAgentRequest(user_prompt="Analyze AAPL", ticker="AAPL")
              }
      - "How will rising rates affect JPMorgan?"
            → intent=EQUITY, sub_agents_needed=["equity", "macro"],
              classified_entities={
                  "equity": SubAgentRequest(user_prompt=..., ticker="JPM"),
                  "macro":  SubAgentRequest(user_prompt=..., topic="interest rates")
              }

    Returns:
        Updates: intent, sub_agents_needed, classified_entities (Dict[str, SubAgentRequest])
    """
    from src.supervisor.router import classify_user_prompt
    from src.shared.schemas import SubAgentRequest
    
    prompt = state["user_prompt"]
    try:
        classification = classify_user_prompt(prompt)
    except Exception as e:
        return {"error": f"Classification failed: {e}"}
        
    sub_agents_needed = []
    classified_entities = {}
    
    for task in classification.tasks:
        sub_agents_needed.append(task.agent_type)
        request = SubAgentRequest(
            user_prompt=prompt,
        )
        if task.agent_type == "equity":
            request.ticker = task.ticker
        elif task.agent_type == "macro":
            request.topic = task.topic
        elif task.agent_type == "commodity":
            request.asset = task.asset
            
        classified_entities[task.agent_type] = request
        
    return {
        "intent": classification.primary_intent,
        "sub_agents_needed": sub_agents_needed,
        "classified_entities": classified_entities,
    }


def run_equity_agent(state: SupervisorState) -> dict:
    """
    Invokes the Equity Research Sub-Agent using the SubAgentRequest stored in
    classified_entities["equity"]. Returns a SubAgentResponse.

    Data flow:
      1. Read:    request = state["classified_entities"]["equity"]  # SubAgentRequest
      2. Translate into AgentState input for the equity LangGraph app:
              equity_input = {"messages": [HumanMessage(content=f"Analyze {request.ticker}")]}
      3. Invoke: result = equity_app.invoke(equity_input)
      4. Load:   report_md = open(result["report_path"]).read()
      5. Wrap:   response = SubAgentResponse(
                     agent_type="equity",
                     report_md=report_md,
                     success=True,
                     metadata={"report_path": result["report_path"]}
                 )
      6. Store:  agent_results["equity"] = response.report_md

    Returns:
        Updates: agent_results["equity"] (str — the Markdown from SubAgentResponse.report_md)
    """
    try:
        from src.agents.equity.graph import app as equity_app
        from langchain_core.messages import HumanMessage
        request = state["classified_entities"]["equity"]

        equity_input = {
            "messages": [HumanMessage(content=f"{request.user_prompt}")],
        }
        result = equity_app.invoke(equity_input)

        report_md = _extract_report(result, agent_type="equity")
        return {"agent_results": {"equity": report_md}}

    except Exception as e:
        error_msg = f"**Equity Agent Error:** {e}"
        return {"agent_results": {"equity": error_msg}}


def run_macro_agent(state: SupervisorState) -> dict:
    """
    Invokes the Macro Sub-Agent using the SubAgentRequest stored in
    classified_entities["macro"]. Returns a SubAgentResponse.

    Data flow:
      1. Read:    request = state["classified_entities"]["macro"]  # SubAgentRequest
      2. Translate into MacroState input for the macro LangGraph app:
              macro_input = {"messages": [HumanMessage(content=request.topic)]}
      3. Invoke: result = macro_app.invoke(macro_input)
      4. Load:   report_md = open(result["report_path"]).read()
      5. Wrap:   response = SubAgentResponse(
                     agent_type="macro",
                     report_md=report_md,
                     success=True,
                     metadata={"report_path": result["report_path"]}
                 )
      6. Store:  agent_results["macro"] = response.report_md

    Returns:
        Updates: agent_results["macro"] (str — the Markdown from SubAgentResponse.report_md)
    """
    try:
        from src.agents.macro.graph import app as macro_app
        from langchain_core.messages import HumanMessage
        request = state["classified_entities"]["macro"]

        macro_input = {
            "messages": [HumanMessage(content=f"{request.user_prompt}")],
        }
        result = macro_app.invoke(macro_input)

        report_md = _extract_report(result, agent_type="macro")
        return {"agent_results": {"macro": report_md}}

    except Exception as e:
        error_msg = f"**Macro Agent Error:** {e}"
        return {"agent_results": {"macro": error_msg}}


def run_commodity_agent(state: SupervisorState) -> dict:
    """
    Invokes the Commodity Sub-Agent using the SubAgentRequest stored in
    classified_entities["commodity"]. Returns a SubAgentResponse.

    Data flow:
      1. Read:    request = state["classified_entities"]["commodity"]  # SubAgentRequest
      2. Translate into CommodityState input for the commodity LangGraph app:
              commodity_input = {"messages": [HumanMessage(content=request.asset)]}
      3. Invoke: result = commodity_app.invoke(commodity_input)
      4. Load:   report_md = open(result["report_path"]).read()
      5. Wrap:   response = SubAgentResponse(
                     agent_type="commodity",
                     report_md=report_md,
                     success=True,
                     metadata={"report_path": result["report_path"]}
                 )
      6. Store:  agent_results["commodity"] = response.report_md

    Returns:
        Updates: agent_results["commodity"] (str — the Markdown from SubAgentResponse.report_md)
    """
    try:
        from src.agents.commodity.graph import app as commodity_app
        from langchain_core.messages import HumanMessage
        request = state["classified_entities"]["commodity"]

        commodity_input = {
            "messages": [HumanMessage(content=f"{request.user_prompt}")],
        }
        result = commodity_app.invoke(commodity_input)

        report_md = _extract_report(result, agent_type="commodity")
        return {"agent_results": {"commodity": report_md}}

    except Exception as e:
        error_msg = f"**Commodity Agent Error:** {e}"
        return {"agent_results": {"commodity": error_msg}}

def _extract_report(final_state: dict, agent_type: str) -> str:
    """
    Extracts the best available output from a sub-agent's final state.

    Priority order:
      1. Saved report file (report_path) — full pipeline completed
      2. report_content (Pydantic model) — sections generated but not saved
      3. Individual sections — partial pipeline completion
      4. Error message — pipeline failed early
    """
    import os

    # 1. Full report file exists
    report_path = final_state.get("report_path")
    if report_path and os.path.exists(report_path):
        try:
            with open(report_path, "r") as f:
                return f.read()
        except Exception:
            pass

    # 2. Report content model exists (sections generated but not saved)
    report_content = final_state.get("report_content")
    if report_content:
        try:
            parts = []
            for field_name, field_value in report_content:
                if hasattr(field_value, "title") and hasattr(field_value, "content"):
                    parts.append(f"## {field_value.title}\n\n{field_value.content}")
                elif isinstance(field_value, str) and field_value:
                    parts.append(f"**{field_name}:** {field_value}")
            if parts:
                return f"# Partial {agent_type.title()} Report\n\n" + "\n\n".join(parts)
        except Exception:
            pass

    # 3. Individual sections exist
    sections = final_state.get("sections", {})
    if sections:
        parts = []
        for key, section in sections.items():
            if hasattr(section, "title") and hasattr(section, "content"):
                parts.append(f"## {section.title}\n\n{section.content}")
            elif isinstance(section, str):
                parts.append(f"## {key}\n\n{section}")
        if parts:
            return f"# Partial {agent_type.title()} Report (incomplete)\n\n" + "\n\n".join(parts)

    # 4. Error fallback
    error = final_state.get("error", "Unknown error")
    return f"**{agent_type.title()} Agent:** Pipeline failed — {error}"


def synthesize_results(state: SupervisorState) -> dict:
    """
    Convergence node — merges outputs from all sub-agents that ran into a single
    coherent response directed at the user's original question.

    Delegates to synthesizer.run_synthesis() which handles:
      - Single-agent passthrough (no LLM call)
      - Multi-agent LLM synthesis

    Returns:
        Updates: final_response
    """
    agent_results = state.get("agent_results", {})
    if not agent_results:
        return {"final_response": "No agent produced results.", "error": "Empty agent_results"}

    synthesis_input = SynthesisInput(
        user_prompt=state["user_prompt"],
        agent_results=agent_results,
        primary_intent=state.get("intent"),
    )

    try:
        output = run_synthesis(synthesis_input)
        return {"final_response": output.final_response}
    except Exception as e:
        return {"final_response": f"Synthesis failed: {e}", "error": str(e)}


def handle_error(state: SupervisorState) -> dict:
    """
    Catches errors propagated from any upstream node via state["error"].
    Always produces a user-friendly message in final_response so the CLI
    never silently fails.

    Returns:
        Updates: final_response
    """
    error_msg = state.get("error", "Unknown Error")
    return {"final_response": f"**Supervisor Error:** {error_msg}"}


# ---------------------------------------------------------------------------
# Edge routing functions
# ---------------------------------------------------------------------------

def route_to_agents(state: SupervisorState):
    """
    Conditional edge using the LangGraph Send API for true parallel fan-out.

    Each sub-agent needed gets its own independent `Send(node_name, state)` object.
    LangGraph executes all Sends concurrently; their outputs are merged before
    synthesize_results runs.

    Returns:
        List of Send objects, one per sub-agent. E.g.:
          [Send("run_equity_agent", state), Send("run_macro_agent", state)]

    Routes to handle_error if state["error"] is set.
    """
    if state.get("error"):
        return "handle_error"

    from src.supervisor.router import AGENT_NODE_MAP
    from langgraph.types import Send
    sends = []
    for agent in state.get("sub_agents_needed", []):
        node = AGENT_NODE_MAP.get(agent)
        if node:
            sends.append(Send(node, state))
    return sends


# ---------------------------------------------------------------------------
# Graph Assembly
# ---------------------------------------------------------------------------

workflow = StateGraph(SupervisorState)

# Register nodes
workflow.add_node("classify_intent",     classify_intent)
workflow.add_node("run_equity_agent",    run_equity_agent)
workflow.add_node("run_macro_agent",     run_macro_agent)
workflow.add_node("run_commodity_agent", run_commodity_agent)
workflow.add_node("synthesize_results",  synthesize_results)
workflow.add_node("handle_error",        handle_error)

# Entry: always classify intent first
workflow.add_edge(START, "classify_intent")

# After classification: guard for errors, then fan out via Send
workflow.add_conditional_edges(
    "classify_intent",
    route_to_agents,
    ["run_equity_agent", "run_macro_agent", "run_commodity_agent", "handle_error"]
)

# Each sub-agent node converges to synthesize_results
# (LangGraph automatically waits for all parallel Send branches to complete)
workflow.add_edge("run_equity_agent",    "synthesize_results")
workflow.add_edge("run_macro_agent",     "synthesize_results")
workflow.add_edge("run_commodity_agent", "synthesize_results")

# Final edges
workflow.add_edge("synthesize_results", END)
workflow.add_edge("handle_error",       END)

app = workflow.compile()
