"""
Macro & Nation Sub-Agent — LangGraph Graph Wiring.

This file contains ONLY graph structure (nodes, edges, compile).
All node/edge logic lives in nodes.py.

Flow:
  planner → fetch_indicators → research_qualitative → synthesize_research
    → generate_section (loop until all sections done) → generate_report
    → save_report → END
"""

from langgraph.graph import StateGraph, START, END

from src.agents.macro.schemas import MacroState
from src.agents.macro.nodes import (
    planner,
    fetch_indicators,
    research_qualitative,
    synthesize_research,
    generate_section,
    generate_report,
    verify_content,
    save_report,
    check_error,
    check_loop,
)

# ---------------------------------------------------------------------------
# Graph Assembly
# ---------------------------------------------------------------------------

workflow = StateGraph(MacroState)

workflow.add_node("planner",              planner)
workflow.add_node("fetch_indicators",     fetch_indicators)
workflow.add_node("research_qualitative", research_qualitative)
workflow.add_node("synthesize_research",  synthesize_research)
workflow.add_node("generate_section",     generate_section)
workflow.add_node("generate_report",      generate_report)
workflow.add_node("verify_content",       verify_content)
workflow.add_node("save_report",          save_report)

# Entry
workflow.add_edge(START, "planner")

# Linear pipeline with error guards
workflow.add_conditional_edges("planner",              check_error, {"continue": "fetch_indicators",     "end": END})
workflow.add_conditional_edges("fetch_indicators",     check_error, {"continue": "research_qualitative", "end": END})
workflow.add_conditional_edges("research_qualitative", check_error, {"continue": "synthesize_research",  "end": END})
workflow.add_conditional_edges("synthesize_research",  check_error, {"continue": "generate_section",    "end": END})

# Section generation loop
workflow.add_conditional_edges(
    "generate_section",
    check_loop,
    {
        "generate_section": "generate_section",
        "generate_report":  "generate_report",
    }
)

workflow.add_edge("generate_report", "verify_content")
workflow.add_edge("verify_content",  "save_report")
workflow.add_edge("save_report", END)

app = workflow.compile()
