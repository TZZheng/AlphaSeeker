"""
Commodity Sub-Agent — Node Functions.

All node and edge functions for the commodity research pipeline.
graph.py imports from here and handles only the LangGraph wiring.

Pipeline order:
  planner → fetch_eia_data → fetch_cot_data → fetch_futures_curve
    → research_qualitative → synthesize_research
    → generate_section (loop) → generate_report → save_report → END
"""

from typing import Literal
from src.agents.commodity.schemas import CommodityState
from src.shared.llm_manager import get_llm
from src.shared.model_config import get_model
from src.shared.text_utils import condense_context, read_file_safe


# --- Model Assignments (from config/models.yaml, overridable via env vars) ---
MODEL_PLAN     = get_model("commodity", "plan")
MODEL_CONDENSE = get_model("commodity", "condense")
MODEL_FOLLOWUP = get_model("commodity", "followup")
MODEL_MAP      = get_model("commodity", "map")
MODEL_REDUCE   = get_model("commodity", "reduce")
MODEL_SECTION  = get_model("commodity", "section")
MODEL_SUMMARY  = get_model("commodity", "summary")

# --- Constants ---
# Must match the CommodityReport section field names exactly.
SECTION_ORDER = [
    "supply_demand",
    "futures_curve",
    "positioning",
    "macro_linkages",
]

# TODO: Polish these prompts with domain-specific guidance after initial implementation.
SECTION_PROMPTS = {
    "supply_demand": (
        "Analyze the current and projected supply/demand balance. Cover production levels, "
        "inventory draws/builds (EIA data for energy), OPEC+ quotas and compliance, "
        "seasonal patterns, and demand trends from major consumers (China, US, EU). "
        "Include specific inventory figures and week-over-week changes."
    ),
    "futures_curve": (
        "Analyze the shape of the futures curve. Determine whether the market is in contango "
        "or backwardation, and what that implies for storage economics and producer hedging. "
        "Calculate annualised roll yield. Compare current curve shape to historical norms. "
        "Reference specific contract months and prices."
    ),
    "positioning": (
        "Analyze CFTC Commitments of Traders (COT) positioning data. Cover speculative "
        "net long/short levels, weekly changes, and historical context. Identify whether "
        "positioning is crowded (extreme net longs/shorts) and what that implies for "
        "potential squeezes or unwinds. Reference specific contract and positioning figures."
    ),
    "macro_linkages": (
        "Analyze how macro factors drive this commodity. Cover USD correlation (DXY impact), "
        "interest rate sensitivity, emerging market demand linkages, and geopolitical drivers "
        "(sanctions, trade wars, regional conflicts). Include specific examples and quantify "
        "correlations where possible."
    ),
}


# ---------------------------------------------------------------------------
# Data Nodes
# ---------------------------------------------------------------------------

def planner(state: CommodityState) -> dict:
    """
    Decomposes the user's commodity request into a CommodityPlan via structured LLM output.
    Identifies the asset name, asset_class, primary_exchange, and time_horizon.
    Initialises all state containers to empty.

    Returns:
        Updates: plan, asset, source_metadata, research_data, research_brief,
                 sections, error
    """
    ...


def fetch_eia_data(state: CommodityState) -> dict:
    """
    Fetches EIA inventory and production data for energy commodities.
    Saves a Markdown summary to eia_data_path.
    Skipped gracefully (warning only) for non-energy commodities.

    Returns:
        Updates: eia_data_path, source_metadata, error
    """
    ...


def fetch_cot_data(state: CommodityState) -> dict:
    """
    Fetches CFTC Commitments of Traders (COT) report for the commodity's
    primary futures market. Shows speculative long/short positioning.
    Saves a Markdown summary to cot_data_path.

    Returns:
        Updates: cot_data_path, source_metadata, error
    """
    ...


def fetch_futures_curve(state: CommodityState) -> dict:
    """
    Fetches the current futures curve (spot through 12-month contracts).
    Identifies contango or backwardation structure and annualised roll yield.
    Saves a Markdown summary to futures_data_path.

    Returns:
        Updates: futures_data_path, source_metadata, error
    """
    ...


# ---------------------------------------------------------------------------
# Research Nodes
# ---------------------------------------------------------------------------

def research_qualitative(state: CommodityState) -> dict:
    """
    Runs multi-phase web research on the commodity.

    Phase 1: Generic commodity queries (supply/demand, geopolitics, seasonal
             patterns, demand outlook from major consumers).
    Phase 2: LLM-generated follow-up queries based on initial findings
             (e.g. for crude oil: "OPEC+ production quota compliance 2025").

    Results stored in research_data keyed by query string.

    Returns:
        Updates: research_data, source_metadata, error
    """
    ...


def synthesize_research(state: CommodityState) -> dict:
    """
    Map-Reduce LLM synthesis of raw research_data into per-section briefs.
    Identical pattern to equity/nodes.synthesize_research.

    Returns:
        Updates: research_brief, extracted_facts, error
    """
    ...


# ---------------------------------------------------------------------------
# Report Generation Nodes
# ---------------------------------------------------------------------------

def generate_section(state: CommodityState) -> dict:
    """
    Generates one CommodityReport section per call using the per-section research brief.
    The check_loop edge drives repeated calls until all sections are generated.

    Returns:
        Updates: sections (adds one new CommoditySection), error
    """
    ...


def generate_report(state: CommodityState) -> dict:
    """
    Synthesizes all sections into the final CommodityReport:
    price_outlook, key_risks, and assembled section objects.

    Returns:
        Updates: report_content, error
    """
    ...


def verify_content(state: CommodityState) -> dict:
    """
    Quality-control pass: checks that all required CommodityReport sections are
    non-empty, that price_outlook and key_risks are present, and that data source
    references (EIA, CFTC URLs) are non-empty.
    Logs warnings but does not block the pipeline.

    Returns:
        Updates: report_content (finalized CommodityReport), error
    """
    ...


def save_report(state: CommodityState) -> dict:
    """
    Serializes the CommodityReport to a Markdown file under reports/.
    Stores the file path in report_path.

    Returns:
        Updates: report_path
    """
    ...


# ---------------------------------------------------------------------------
# Edge Functions
# ---------------------------------------------------------------------------

def check_error(state: CommodityState) -> Literal["continue", "end"]:
    """Guards every transition. Routes to END if state["error"] is set."""
    ...


def check_loop(state: CommodityState) -> Literal["generate_section", "generate_report"]:
    """
    Drives the section generation loop. Returns "generate_section" until all
    CommodityReport sections have been written, then returns "generate_report".
    """
    ...
