"""
Macro & Nation Sub-Agent — Node Functions.

All node and edge functions for the macro research pipeline.
graph.py imports from here and handles only the LangGraph wiring.

Pipeline order:
  planner → fetch_indicators → research_qualitative → synthesize_research
    → generate_section (loop) → generate_report → save_report → END
"""

import os
import datetime
import concurrent.futures
from typing import Literal, Dict, List
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from src.agents.macro.schemas import MacroState, MacroPlan, MacroSection, MacroReport
from src.shared.llm_manager import get_llm
from src.shared.model_config import get_model
from src.shared.text_utils import condense_context, read_file_safe
from src.shared.web_search import deep_search

from src.agents.macro.tools.fred import fetch_macro_indicators
from src.agents.macro.tools.world_bank import fetch_world_bank_indicators

# --- Model Assignments (from config/models.yaml, overridable via env vars) ---
MODEL_PLAN     = get_model("macro", "plan")
MODEL_CONDENSE = get_model("macro", "condense")
MODEL_FOLLOWUP = get_model("macro", "followup")
MODEL_MAP      = get_model("macro", "map")
MODEL_REDUCE   = get_model("macro", "reduce")
MODEL_SECTION  = get_model("macro", "section")
MODEL_SUMMARY  = get_model("macro", "summary")

# --- Constants ---
SECTION_ORDER = [
    "current_conditions",
    "policy_analysis",
    "global_linkages",
    "scenario_analysis",
]

SECTION_PROMPTS = {
    "current_conditions": (
        "Analyze the current state of key economic indicators. Include recent data points "
        "for GDP growth, inflation (CPI/PCE), unemployment, and any other relevant metrics. "
        "Reference specific FRED series IDs or data sources. Compare current readings to "
        "historical averages and recent trends."
    ),
    "policy_analysis": (
        "Analyze central bank policy stance and forward guidance. Cover the current rate "
        "environment, dot plot expectations, quantitative tightening/easing status, and "
        "market-implied rate probabilities. Include specific Fed/ECB/BOJ statements and dates."
    ),
    "global_linkages": (
        "Analyze cross-country economic spillovers. Cover trade flows, currency dynamics "
        "(DXY, major pairs), capital flow patterns, and contagion risks. Identify which "
        "economies are most exposed to the macro topic and through what transmission channels."
    ),
    "scenario_analysis": (
        "Construct bull/base/bear macro scenarios with probability weights. Each scenario "
        "should specify: key assumptions, GDP/inflation/rate trajectory, market implications, "
        "and trigger events that would confirm or invalidate the scenario."
    ),
}

# ---------------------------------------------------------------------------
# Data Nodes
# ---------------------------------------------------------------------------

def planner(state: MacroState) -> dict:
    print("\n--- [Macro] Phase: Planner ---")
    recent_messages = state["messages"][-10:]
    system_prompt = SystemMessage(content="""
    You are a macroeconomic research planner.
    Analyze the user's request and create a detailed execution plan.
    Identify the main topic, countries involved, and target economic indicators.

    You must respond in valid JSON matching this exact schema:
    {
        "topic": "US Inflation and Interest Rates",
        "countries": ["US"],
        "indicators": ["CPI", "PCE", "Fed Funds Rate"],
        "time_horizon": "12m"
    }
    """)
    structured_llm = get_llm(MODEL_PLAN).with_structured_output(MacroPlan, method="json_mode")
    try:
        plan = structured_llm.invoke([system_prompt] + recent_messages)
        return {
            "plan": plan,
            "topic": plan.topic,
            "source_metadata": {},
            "sections": {},
            "research_data": {},
            "research_brief": {},
            "error": None
        }
    except Exception as e:
        return {"error": f"Failed to generate plan: {str(e)}"}


def fetch_indicators(state: MacroState) -> dict:
    print("\n--- [Macro] Phase: Fetching Indicators (FRED & World Bank) ---")
    if state.get("error"): return state
    plan = state["plan"]
    topic = plan.topic
    countries = plan.countries
    
    metadata_store = state.get("source_metadata", {})
    indicators_content = "# Macroeconomic Indicators\n\n"
    
    # FRED is best for US data and general topical curves
    try:
        fred_path, fred_meta = fetch_macro_indicators(topic=topic, countries=countries)
        if fred_path:
            indicators_content += f"## FRED Data\nFetched from: {fred_path}\n"
            metadata_store["fred"] = fred_meta
            indicators_content += read_file_safe(fred_path, max_chars=10000) + "\n\n"
    except Exception as e:
        print(f"Warning: FRED fetch failed: {e}")

    # World Bank if non-US countries are mentioned
    non_us = [c for c in countries if c.lower() not in ["us", "usa", "united states"]]
    if non_us:
        try:
            wb_path, wb_meta = fetch_world_bank_indicators(countries=non_us)
            if wb_path:
                indicators_content += f"## World Bank Data\nFetched from: {wb_path}\n"
                metadata_store["world_bank"] = wb_meta
                indicators_content += read_file_safe(wb_path, max_chars=10000) + "\n\n"
        except Exception as e:
            print(f"Warning: World Bank fetch failed: {e}")

    # Save aggregated indicators
    save_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    indicators_path = os.path.join(save_dir, f"macro_indicators_{timestamp}.md")
    
    with open(indicators_path, "w") as f:
        f.write(indicators_content)

    return {"indicators_path": indicators_path, "source_metadata": metadata_store, "error": None}


# ---------------------------------------------------------------------------
# Research Nodes
# ---------------------------------------------------------------------------

def research_qualitative(state: MacroState) -> dict:
    print("\n--- [Macro] Phase: Deep Qualitative Research ---")
    if state.get("error"): return state
    plan = state["plan"]
    
    # 1. Base queries
    base_queries = [
        f"{plan.topic} macroeconomic forecast",
        f"{plan.topic} central bank statements policy",
        f"{plan.topic} economic indicators analysis",
        f"{plan.topic} Morgan Stanley Goldman Sachs research",
        f"{plan.topic} geopolitical risks impact",
    ]
    
    print(f"Phase 1: Running {len(base_queries)} general queries...")
    text_results = deep_search(queries=base_queries, urls_per_query=3, max_workers=5, use_news=False)
    news_results = deep_search(queries=base_queries, urls_per_query=2, max_workers=5, use_news=True)
    
    # 2. Extract snippets to seed the follow-up generator
    initial_snippets = ""
    for r in text_results + news_results:
        initial_snippets += f"- {r['title']}: {r.get('snippet', '')}\n"
        
    prompt = f"""
    You are a macroeconomic researcher investigating: {plan.topic}

    Here are initial search findings:
    {initial_snippets[:10000]}

    Generate 15 highly specific follow-up search queries to dive deeper into the economics.
    Include queries for specific central bank speeches, recent data release reactions, 
    and systemic risk factors mentioned above.
    
    Output ONLY the queries, one per line.
    """
    try:
        response = get_llm(MODEL_FOLLOWUP).invoke([HumanMessage(content=prompt)])
        lines = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
        followup_queries = [q for q in lines if not q.startswith("#") and not q.startswith("*")][:15]
    except Exception as e:
        print(f"Follow-up generation failed: {e}")
        followup_queries = []
        
    followup_results = []
    if followup_queries:
        print(f"Phase 2: Running {len(followup_queries)} targeted queries...")
        f_text_results = deep_search(followup_queries, urls_per_query=2, max_workers=5, use_news=False)
        followup_results = f_text_results

    # 3. Consolidate
    research_data: Dict[str, str] = {}
    for r in text_results + news_results + followup_results:
        query = r["query"]
        content = f"### {r['title']}\nSource: {r['url']}\n\n"
        if r.get("full_text"):
            content += r["full_text"] + "\n"
        else:
            content += f"Snippet: {r.get('snippet', '')}\n"
            
        if query in research_data:
            research_data[query] += "\n---\n" + content
        else:
            research_data[query] = content
            
    print(f"Collected research across {len(research_data)} query buckets.")
    return {"research_data": research_data, "error": None}


def synthesize_research(state: MacroState) -> dict:
    print("\n--- [Macro] Phase: Synthesizing Research (Map/Reduce) ---")
    if state.get("error"): return state
    topic = state["plan"].topic
    research_data = state.get("research_data", {})
    
    if not research_data:
        return {"research_brief": {}, "error": None}

    # MAP STEP
    all_text_items = list(research_data.values())
    CHUNK_SIZE = 50000
    chunks: List[str] = []
    current_chunk = ""
    for text in all_text_items:
        if len(current_chunk) + len(text) > CHUNK_SIZE:
            if current_chunk: chunks.append(current_chunk)
            current_chunk = text
        else:
            current_chunk += "\n\n" + text
    if current_chunk: chunks.append(current_chunk)

    def _map_chunk(args):
        i, chunk = args
        map_prompt = f"""
        You are a macro analyst extracting facts about: {topic}.
        Extract EVERY important macroeconomic fact, number, date, quote, and policy stance.
        Focus on rates, inflation, GDP, employment, trade, and central bank commentary.
        Output a bullet-point list of the critical data points.
        RAW TEXT:
        {chunk}
        """
        try:
            resp = get_llm(MODEL_MAP).invoke([HumanMessage(content=map_prompt)])
            return resp.content
        except:
            return ""

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        extracted = list(executor.map(_map_chunk, enumerate(chunks)))
        
    all_facts = "\n\n".join([f for f in extracted if f])

    # REDUCE STEP
    research_brief: Dict[str, str] = {}
    
    def _reduce_section(section_key):
        section_label = section_key.replace("_", " ").title()
        reduce_prompt = f"""
        You are preparing a research brief for the **{section_label}** section analyzing {topic}.
        Select and organize ONLY the facts relevant to {section_label}.
        
        SECTION GUIDANCE:
        {SECTION_PROMPTS[section_key]}
        
        EXTRACTED FACTS:
        {all_facts}
        """
        try:
            resp = get_llm(MODEL_REDUCE).invoke([HumanMessage(content=reduce_prompt)])
            return section_key, resp.content
        except:
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

def generate_section(state: MacroState) -> dict:
    print("\n--- [Macro] Phase: Generating Report Sections ---")
    if state.get("error"): return state
    plan = state["plan"]
    topic = plan.topic
    
    indicators_path = state.get("indicators_path")
    indicators_content = read_file_safe(indicators_path, max_chars=15000)
    research_brief = state.get("research_brief", {})
    
    new_sections = state.get("sections", {}).copy()

    def _generate_single_section(section_key):
        if section_key in new_sections:
            return section_key, new_sections[section_key]

        section_brief = research_brief.get(section_key, "No research brief available.")
        
        prompt = f"""
        You are a Senior Macroeconomist for AlphaSeeker.
        Generate the **{section_key.replace('_', ' ').upper()}** section for the report on {topic}.
        
        ### Quantitative Indicators Database
        {indicators_content}
        
        ### Deep Research Brief
        {section_brief}
        
        ### Section Guidance
        {SECTION_PROMPTS[section_key]}
        
        CRITICAL INSTRUCTIONS:
        - Output must be a valid 'MacroSection' object.
        - You must respond in valid JSON matching this exact schema:
          {{
              "title": "Section Title",
              "content": "Full markdown content of the section, at least 800 words..."
          }}
        - Do NOT include 'subsections' or any other JSON fields. Put all your formatted text inside the 'content' string.
        - Use the Deep Research Brief facts natively.
        - Cite specific quantitative points from the indicators database.
        - Target 800+ words. Professional tone.
        """
        try:
            structured_llm = get_llm(MODEL_SECTION).with_structured_output(MacroSection)
            section = structured_llm.invoke([SystemMessage(content=prompt)])
            return section_key, section
        except Exception as e:
            print(f"Failed to generate section {section_key}: {e}")
            return section_key, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(SECTION_ORDER)) as executor:
        futures = {executor.submit(_generate_single_section, key): key for key in SECTION_ORDER}
        for future in concurrent.futures.as_completed(futures):
            key, section = future.result()
            if section:
                new_sections[key] = section

    return {"sections": new_sections, "error": None}


def generate_report(state: MacroState) -> dict:
    print("\n--- [Macro] Phase: Generating Final Summary ---")
    if state.get("error"): return state
    topic = state["plan"].topic
    sections = state.get("sections", {})
    
    # Check if we have missing sections and provide a fallback
    for key in SECTION_ORDER:
        if key not in sections:
            sections[key] = MacroSection(title=key.replace("_", " ").title(), content="Section failed to generate.")
            
    full_text = "\n\n".join([f"## {s.title}\n{s.content}" for s in sections.values()])
    
    prompt = f"""
    You are a Chief Economist.
    Write the final outlook and key risks summary for our research on {topic}.
    
    FULL ANALYSIS:
    {full_text}
    
    INSTRUCTIONS:
    1. **Outlook**: Write a single, highly dense paragraph summarizing the net impact on the economy and broad asset classes.
    2. **Key Risks**: Summarize the 2-3 most critical risks that could derail this base case.
    3. Output must be valid JSON matching the SummaryOutput schema exactly.
    """
    
    class SummaryOutput(BaseModel):
        outlook: str
        key_risks: str
        references: List[str]
        
    structured_llm = get_llm(MODEL_SUMMARY).with_structured_output(SummaryOutput, method="json_mode")
    
    try:
        summary = structured_llm.invoke([SystemMessage(content=prompt)])
        
        final_report = MacroReport(
            topic=topic,
            outlook=summary.outlook,
            key_risks=summary.key_risks,
            current_conditions=sections["current_conditions"],
            policy_analysis=sections["policy_analysis"],
            global_linkages=sections["global_linkages"],
            scenario_analysis=sections["scenario_analysis"],
            references=summary.references
        )
        return {"report_content": final_report, "error": None}
    except Exception as e:
        return {"error": f"Failed to generate report: {e}"}


def verify_content(state: MacroState) -> dict:
    print("\n--- [Macro] Phase: Verifying Output ---")
    if state.get("error"): return state
    if not state.get("report_content"):
        return {"error": "No report generated."}
    return state


def save_report(state: MacroState) -> dict:
    print("\n--- [Macro] Phase: Saving Final Report ---")
    if state.get("error"): return state
    
    report = state.get("report_content")
    
    md = f"# Macroeconomic Research: {report.topic}\n\n"
    md += f"## Executive Outlook\n{report.outlook}\n\n"
    md += f"### Key Risks\n{report.key_risks}\n\n"
    md += "---\n\n"
    
    sections = [
        report.current_conditions,
        report.policy_analysis,
        report.global_linkages,
        report.scenario_analysis,
    ]
    
    def _clean_section_content(title: str, content: str) -> str:
        import re
        lines = content.splitlines()
        if lines:
            first = lines[0].lstrip("#").strip()
            if first.lower() == title.lower():
                lines = lines[1:]
                if lines and lines[0].strip() == "":
                    lines = lines[1:]
            content = "\n".join(lines)

        def _demote(m):
            return ("#" * min(len(m.group(1)) + 1, 6)) + m.group(2)
        content = re.sub(r"^(#{1,5})( .+)$", _demote, content, flags=re.MULTILINE)

        content = re.sub(r"\*?\*?Word Count[:\*\s][^\n]*(\n[^\n]+)*", "", content, flags=re.IGNORECASE).strip()
        return content

    for s in sections:
        cleaned = _clean_section_content(s.title, s.content)
        md += f"## {s.title}\n{cleaned}\n\n"
        
    md += "## References\n"
    for r in report.references:
        md += f"- {r}\n"
        
    # Create a safe filename from the topic
    import re
    safe_topic = re.sub(r'[^a-zA-Z0-9]+', '_', report.topic).strip('_')
    
    report_dir = os.path.join(os.getcwd(), "reports")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, f"Macro_{safe_topic}.md")
    
    with open(report_path, "w") as f:
        f.write(md)
        
    return {"report_path": report_path}


# ---------------------------------------------------------------------------
# Edge Functions
# ---------------------------------------------------------------------------

def check_error(state: MacroState) -> Literal["continue", "end"]:
    return "end" if state.get("error") else "continue"

def check_loop(state: MacroState) -> Literal["generate_section", "generate_report"]:
    # Loop removed since we parallelized section generation. Just pass through to report.
    return "generate_report"
