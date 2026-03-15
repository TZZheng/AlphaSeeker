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
from src.shared.report_filename import build_prompt_report_filename, extract_prompt_text
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
    print("\n--- [Commodity] Phase: Planner ---")
    recent_messages = state["messages"][-10:]
    
    from langchain_core.messages import SystemMessage
    from src.agents.commodity.schemas import CommodityPlan
    
    system_prompt = SystemMessage(content="""
    You are a commodity market research planner.
    Analyze the user's request and create a detailed execution plan.
    Identify the specific physical asset, its broad class, primary exchange, and time horizon.

    You must respond in valid JSON matching this exact schema:
    {
        "asset": "Crude Oil",
        "asset_class": "energy",
        "primary_exchange": "NYMEX",
        "time_horizon": "12m"
    }
    """)
    try:
        structured_llm = get_llm(MODEL_PLAN).with_structured_output(CommodityPlan, method="json_mode")
        plan = structured_llm.invoke([system_prompt] + recent_messages)
        return {
            "plan": plan,
            "asset": plan.asset,
            "source_metadata": {},
            "sections": {},
            "research_data": {},
            "research_brief": {},
            "error": None
        }
    except Exception as e:
        return {"error": f"Failed to generate plan: {str(e)}"}


def fetch_eia_data(state: CommodityState) -> dict:
    """
    Fetches EIA inventory and production data for energy commodities.
    Saves a Markdown summary to eia_data_path.
    Skipped gracefully (warning only) for non-energy commodities.

    Returns:
        Updates: eia_data_path, source_metadata, error
    """
    print("\n--- [Commodity] Phase: Fetching EIA Data ---")
    if state.get("error"): return state
    asset = state["plan"].asset
    
    from src.agents.commodity.tools.eia import fetch_eia_inventory
    path, meta = fetch_eia_inventory(asset)
    
    metadata = state.get("source_metadata", {})
    if meta:
        metadata["eia"] = meta
        
    return {"eia_data_path": path, "source_metadata": metadata, "error": None}


def fetch_cot_data(state: CommodityState) -> dict:
    """
    Fetches CFTC Commitments of Traders (COT) report for the commodity's
    primary futures market. Shows speculative long/short positioning.
    Saves a Markdown summary to cot_data_path.

    Returns:
        Updates: cot_data_path, source_metadata, error
    """
    print("\n--- [Commodity] Phase: Fetching CFTC COT Data ---")
    if state.get("error"): return state
    asset = state["plan"].asset
    
    from src.agents.commodity.tools.cftc import fetch_cot_report
    path, meta = fetch_cot_report(asset)
    
    metadata = state.get("source_metadata", {})
    if meta:
        metadata["cot"] = meta
        
    return {"cot_data_path": path, "source_metadata": metadata, "error": None}


def fetch_futures_curve(state: CommodityState) -> dict:
    """
    Fetches the current futures curve (spot through 12-month contracts).
    Identifies contango or backwardation structure and annualised roll yield.
    Saves a Markdown summary to futures_data_path.

    Returns:
        Updates: futures_data_path, source_metadata, error
    """
    print("\n--- [Commodity] Phase: Fetching Futures Curve ---")
    if state.get("error"): return state
    asset = state["plan"].asset
    
    from src.agents.commodity.tools.futures import fetch_futures_curve as fetch_curve
    path, meta = fetch_curve(asset)
    
    metadata = state.get("source_metadata", {})
    if meta:
        metadata["futures"] = meta
        
    return {"futures_data_path": path, "source_metadata": metadata, "error": None}


# ---------------------------------------------------------------------------
# Research Nodes
# ---------------------------------------------------------------------------

def research_qualitative(state: CommodityState) -> dict:
    """
    Runs multi-phase web research on the commodity.
    """
    print("\n--- [Commodity] Phase: Deep Qualitative Research ---")
    if state.get("error"): return state
    plan = state["plan"]
    asset = plan.asset
    
    from src.shared.web_search import deep_search
    from langchain_core.messages import HumanMessage
    
    # 1. Base queries
    base_queries = [
        f"{asset} supply demand fundamentals",
        f"{asset} geopolitical risks supply chain",
        f"{asset} demand outlook major consumers",
        f"{asset} market structure contango backwardation",
        f"{asset} macro linkages interest rates USD",
    ]
    
    print(f"Phase 1: Running {len(base_queries)} general queries...")
    text_results = deep_search(queries=base_queries, urls_per_query=3, max_workers=5, use_news=False)
    news_results = deep_search(queries=base_queries, urls_per_query=2, max_workers=5, use_news=True)
    
    initial_snippets = ""
    for r in text_results + news_results:
        initial_snippets += f"- {r['title']}: {r.get('snippet', '')}\n"
        
    prompt = f"""
    You are a commodity researcher investigating: {asset}
    Here are initial search findings:
    {initial_snippets[:10000]}

    Generate 10 highly specific follow-up search queries to dive deeper.
    Output ONLY the queries, one per line.
    """
    try:
        response = get_llm(MODEL_FOLLOWUP).invoke([HumanMessage(content=prompt)])
        lines = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
        followup_queries = [q for q in lines if not q.startswith("#") and not q.startswith("*")][:10]
    except Exception as e:
        print(f"Follow-up generation failed: {e}")
        followup_queries = []
        
    followup_results = []
    if followup_queries:
        print(f"Phase 2: Running {len(followup_queries)} targeted queries...")
        followup_results = deep_search(followup_queries, urls_per_query=2, max_workers=5, use_news=False)

    research_data = {}
    for r in text_results + news_results + followup_results:
        query = r["query"]
        content = f"### {r['title']}\nSource: {r['url']}\n\n"
        if r.get("full_text"): content += r["full_text"] + "\n"
        else: content += f"Snippet: {r.get('snippet', '')}\n"
            
        if query in research_data: research_data[query] += "\n---\n" + content
        else: research_data[query] = content
            
    print(f"Collected research across {len(research_data)} query buckets.")
    return {"research_data": research_data, "error": None}


def synthesize_research(state: CommodityState) -> dict:
    """
    Map-Reduce LLM synthesis of raw research_data into per-section briefs.
    """
    print("\n--- [Commodity] Phase: Synthesizing Research (Map/Reduce) ---")
    if state.get("error"): return state
    asset = state["plan"].asset
    research_data = state.get("research_data", {})
    
    import concurrent.futures
    from langchain_core.messages import HumanMessage
    
    if not research_data:
        return {"research_brief": {}, "error": None}

    # MAP
    all_text_items = list(research_data.values())
    chunks = []
    current_chunk = ""
    for text in all_text_items:
        if len(current_chunk) + len(text) > 50000:
            if current_chunk: chunks.append(current_chunk)
            current_chunk = text
        else:
            current_chunk += "\n\n" + text
    if current_chunk: chunks.append(current_chunk)

    def _map_chunk(args):
        i, chunk = args
        map_prompt = f"""
        Extract EVERY important fact, number, and quote about {asset} supply, demand, positioning, and macro factors.
        RAW TEXT:
        {chunk}
        """
        try:
            resp = get_llm(MODEL_MAP).invoke([HumanMessage(content=map_prompt)])
            return resp.content
        except Exception as e:
            print(f"Commodity MAP chunk {i+1} failed: {e}")
            return ""

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        extracted = list(executor.map(_map_chunk, enumerate(chunks)))
        
    all_facts = "\n\n".join([f for f in extracted if f])

    # REDUCE
    research_brief = {}
    
    def _reduce_section(section_key):
        reduce_prompt = f"""
        Prepare a research brief for the **{section_key}** section analyzing {asset}.
        SECTION GUIDANCE:
        {SECTION_PROMPTS[section_key]}
        EXTRACTED FACTS:
        {all_facts}
        """
        try:
            resp = get_llm(MODEL_REDUCE).invoke([HumanMessage(content=reduce_prompt)])
            return section_key, resp.content
        except Exception as e:
            print(f"Commodity REDUCE section '{section_key}' failed: {e}")
            return section_key, ""

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(SECTION_ORDER)) as executor:
        futures = {executor.submit(_reduce_section, key): key for key in SECTION_ORDER}
        for future in concurrent.futures.as_completed(futures):
            key, content = future.result()
            research_brief[key] = content

    return {"research_brief": research_brief, "extracted_facts": all_facts, "error": None}


# ---------------------------------------------------------------------------
# Report Generation Nodes
# ---------------------------------------------------------------------------

def generate_section(state: CommodityState) -> dict:
    """
    Generates one CommodityReport section per call using the per-section research brief.
    The check_loop edge drives repeated calls until all sections are generated.
    """
    print("\n--- [Commodity] Phase: Generating Report Sections ---")
    if state.get("error"): return state
    asset = state["plan"].asset
    
    research_brief = state.get("research_brief", {})
    new_sections = state.get("sections", {}).copy()
    
    import concurrent.futures
    from langchain_core.messages import SystemMessage
    from src.agents.commodity.schemas import CommoditySection

    # Build quantitative context
    quant_content = ""
    p1, p2, p3 = state.get("eia_data_path"), state.get("cot_data_path"), state.get("futures_data_path")
    if p1: quant_content += read_file_safe(p1) + "\n\n"
    if p2: quant_content += read_file_safe(p2) + "\n\n"
    if p3: quant_content += read_file_safe(p3) + "\n\n"

    def _generate_single_section(section_key):
        if section_key in new_sections:
            return section_key, new_sections[section_key]

        section_brief = research_brief.get(section_key, "")
        
        prompt = f"""
        You are a Commodity Market Analyst for AlphaSeeker. Generate the **{section_key}** section for {asset}.
        Quantitative Data: {quant_content[:15000]}
        Deep Research: {section_brief}
        Guidance: {SECTION_PROMPTS[section_key]}
        
        Output MUST be valid JSON matching the CommoditySection schema: {{"title": "Title", "content": "Markdown text"}}
        """
        try:
            structured_llm = get_llm(MODEL_SECTION).with_structured_output(CommoditySection, method="json_mode")
            section = structured_llm.invoke([SystemMessage(content=prompt)])
            return section_key, section
        except Exception as e:
            print(f"Failed to generate {section_key}: {e}")
            return section_key, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(SECTION_ORDER)) as executor:
        futures = {executor.submit(_generate_single_section, key): key for key in SECTION_ORDER}
        for future in concurrent.futures.as_completed(futures):
            key, section = future.result()
            if section:
                new_sections[key] = section

    return {"sections": new_sections, "error": None}


def generate_report(state: CommodityState) -> dict:
    """
    Synthesizes all sections into the final CommodityReport.
    """
    print("\n--- [Commodity] Phase: Generating Final Summary ---")
    if state.get("error"): return state
    asset = state["plan"].asset
    sections = state.get("sections", {})
    
    from src.agents.commodity.schemas import CommodityReport, CommoditySection
    from langchain_core.messages import SystemMessage
    from pydantic import BaseModel
    from typing import List

    for key in SECTION_ORDER:
        if key not in sections:
            sections[key] = CommoditySection(title=key.replace("_", " ").title(), content="Section failed to generate.")
            
    full_text = "\n\n".join([f"## {s.title}\n{s.content}" for s in sections.values()])
    
    prompt = f"""
    Write the final price outlook and key risks summary for {asset}.
    FULL ANALYSIS:
    {full_text}
    
    1. **Outlook**: Bullish/Bearish/Neutral with brief rationale.
    2. **Key Risks**: 2-3 most critical risks.
    3. Output must be valid JSON matching SummaryOutput: {{"price_outlook": "...", "key_risks": "...", "references": ["urls"]}}
    """
    
    class SummaryOutput(BaseModel):
        price_outlook: str
        key_risks: str
        references: List[str]
        
    structured_llm = get_llm(MODEL_SUMMARY).with_structured_output(SummaryOutput, method="json_mode")
    
    try:
        summary = structured_llm.invoke([SystemMessage(content=prompt)])
        
        final_report = CommodityReport(
            asset=asset,
            price_outlook=summary.price_outlook,
            key_risks=summary.key_risks,
            supply_demand=sections["supply_demand"],
            futures_curve=sections["futures_curve"],
            positioning=sections["positioning"],
            macro_linkages=sections["macro_linkages"],
            references=summary.references
        )
        return {"report_content": final_report, "error": None}
    except Exception as e:
        return {"error": f"Failed to generate report: {e}"}


def verify_content(state: CommodityState) -> dict:
    """
    Quality-control pass.
    """
    print("\n--- [Commodity] Phase: Verifying Output ---")
    if state.get("error"): return state
    if not state.get("report_content"):
        return {"error": "No report generated."}
    return state


def save_report(state: CommodityState) -> dict:
    """
    Serializes the CommodityReport to a Markdown file under reports/.
    """
    print("\n--- [Commodity] Phase: Saving Final Report ---")
    if state.get("error"): return state
    
    import os
    import re
    
    def _clean_section_content(title: str, content: str) -> str:
        """Post-process LLM section content to fix common formatting bugs:
        1. Strip duplicate title if the LLM started its content with the same title.
        2. Demote all headings by one level (H1->H2, H2->H3, ...) so they nest
           correctly under the H2 section header added by save_report.
        3. Remove debug "Word Count:" paragraphs the LLM occasionally hallucinates.
        """
        lines = content.splitlines()

        # 1. Strip leading title duplicate (exact or close match, ignoring leading #s)
        if lines:
            first_heading = lines[0].lstrip("#").strip()
            if first_heading.lower() == title.lower():
                # Drop that line plus any immediately following blank line
                lines = lines[1:]
                if lines and lines[0].strip() == "":
                    lines = lines[1:]
            content = "\n".join(lines)

        # 2. Demote headings by one level so sub-headings nest under the ## section header.
        def _demote(m: re.Match) -> str:
            hashes = m.group(1)
            # Cap at H5 (##### -> ######) to avoid going too deep
            return ("#" * min(len(hashes) + 1, 6)) + m.group(2)

        content = re.sub(r"^(#{1,5})( .+)$", _demote, content, flags=re.MULTILINE)

        # 3. Remove "Word Count: ..." debug lines/paragraphs
        content = re.sub(
            r"\*?\*?Word Count[:\*\s][^\n]*(\n[^\n]+)*",
            "",
            content,
            flags=re.IGNORECASE,
        ).strip()

        return content
        
    report = state.get("report_content")
    
    md = f"# Commodity Research: {report.asset}\n\n"
    md += f"## Price Outlook\n{report.price_outlook}\n\n"
    md += f"### Key Risks\n{report.key_risks}\n\n"
    md += "---\n\n"
    
    sections = [
        report.supply_demand,
        report.futures_curve,
        report.positioning,
        report.macro_linkages,
    ]
    
    for s in sections:
        cleaned_content = _clean_section_content(s.title, s.content)
        md += f"## {s.title}\n{cleaned_content}\n\n"
        
    md += "## References\n"
    for r in report.references:
        md += f"- {r}\n"
        
    report_dir = os.path.join(os.getcwd(), "reports")
    os.makedirs(report_dir, exist_ok=True)
    prompt_text = extract_prompt_text(state.get("messages"))
    filename = build_prompt_report_filename(
        prompt_text=prompt_text,
        fallback_stem=f"Commodity_{report.asset}",
        suffix="commodity",
    )
    report_path = os.path.join(report_dir, filename)
    
    with open(report_path, "w") as f:
        f.write(md)
        
    return {"report_path": report_path}


# ---------------------------------------------------------------------------
# Edge Functions
# ---------------------------------------------------------------------------

def check_error(state: CommodityState) -> Literal["continue", "end"]:
    """Guards every transition. Routes to END if state["error"] is set."""
    return "end" if state.get("error") else "continue"


def check_loop(state: CommodityState) -> Literal["generate_section", "generate_report"]:
    """
    Drives the section generation loop. Returns "generate_section" until all
    CommodityReport sections have been written, then returns "generate_report".
    """
    return "generate_report"
