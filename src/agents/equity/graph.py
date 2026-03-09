"""
Equity Research Sub-Agent — LangGraph Graph Wiring.

This file contains ONLY graph structure (nodes, edges, compile).
All node/edge logic, constants, and helpers live in nodes.py.

Flow:
  planner → fetch_data → generate_chart → fetch_company_profile
    → fetch_financials → research_qualitative → synthesize_research
    → review_and_expand_peers → generate_section (loop 7 times)
    → generate_summary → verify_content → save_report → END
"""

from langgraph.graph import StateGraph, START, END

from src.agents.equity.schemas import AgentState
from src.agents.equity.nodes import (
    planner,
    fetch_data,
    generate_chart,
    fetch_company_profile_node,
    fetch_financials,
    research_qualitative,
    synthesize_research,
    review_and_expand_peers,
    generate_section,
    generate_summary,
    verify_content,
    save_report,
    check_error,
)

# ---------------------------------------------------------------------------
# Graph Assembly
# ---------------------------------------------------------------------------

workflow = StateGraph(AgentState)

workflow.add_node("planner", planner)
workflow.add_node("fetch_data", fetch_data)
workflow.add_node("generate_chart", generate_chart)
workflow.add_node("fetch_company_profile", fetch_company_profile_node)
workflow.add_node("fetch_financials", fetch_financials)
workflow.add_node("research_qualitative", research_qualitative)
workflow.add_node("synthesize_research", synthesize_research)
workflow.add_node("review_and_expand_peers", review_and_expand_peers)
workflow.add_node("generate_section", generate_section)
workflow.add_node("generate_summary", generate_summary)
workflow.add_node("verify_content", verify_content)
workflow.add_node("save_report", save_report)

# Entry
workflow.add_edge(START, "planner")

# Error-guarded transitions
workflow.add_conditional_edges("planner", check_error, {"continue": "fetch_data", "end": END})
workflow.add_conditional_edges("fetch_data", check_error, {"continue": "generate_chart", "end": END})
workflow.add_conditional_edges("generate_chart", check_error, {"continue": "fetch_company_profile", "end": END})

# Linear pipeline (company profile onward does not use error guard — uses soft fallback)
workflow.add_edge("fetch_company_profile", "fetch_financials")
workflow.add_edge("fetch_financials", "research_qualitative")
workflow.add_edge("research_qualitative", "synthesize_research")
workflow.add_edge("synthesize_research", "review_and_expand_peers")
workflow.add_edge("review_and_expand_peers", "generate_section")

# Sections generated in parallel internally
workflow.add_edge("generate_section", "generate_summary")

workflow.add_edge("generate_summary", "verify_content")
workflow.add_edge("verify_content", "save_report")
workflow.add_edge("save_report", END)

app = workflow.compile()
