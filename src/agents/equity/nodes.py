"""
Equity Research Sub-Agent — Node Functions & Helpers.

This module owns ALL node logic, edge functions, constants, and helpers for the
equity research pipeline. graph.py imports from here and handles only wiring.

Pipeline order:
  planner → fetch_data → generate_chart → fetch_company_profile_node
    → fetch_financials → research_qualitative → synthesize_research
    → review_and_expand_peers → generate_section (loop) → generate_summary
    → verify_content → save_report → END
"""

from typing import Literal, Dict, List
import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.agents.equity.schemas import AgentState, AnalysisPlan, ResearchReport, ResearchSection
from src.agents.equity.tools.market_data import fetch_historical_data
from src.agents.equity.tools.visualization import plot_price_history
from src.agents.equity.tools.financials import fetch_financial_metrics
from src.agents.equity.tools.peers import fetch_peer_metrics, evaluate_candidates, extract_peers_from_text
from src.shared.web_search import search_web, search_news, deep_search
from src.agents.equity.tools.sec_filings import search_and_read_filings, extract_supply_chain_data
from src.agents.equity.tools.insider_trading import fetch_insider_activity
from src.agents.equity.tools.earnings_calls import research_earnings_call
from src.agents.equity.tools.company_profile import fetch_company_profile
from src.shared.llm_manager import get_llm
from src.shared.model_config import get_model
from src.shared.text_utils import condense_context, read_file_safe
from src.shared.report_filename import build_prompt_report_filename, extract_prompt_text


# --- Model Assignments (from config/models.yaml, overridable via env vars) ---
MODEL_PLAN     = get_model("equity", "plan")
MODEL_CONDENSE = get_model("equity", "condense")
MODEL_FOLLOWUP = get_model("equity", "followup")
MODEL_MAP      = get_model("equity", "map")
MODEL_REDUCE   = get_model("equity", "reduce")
MODEL_SECTION  = get_model("equity", "section")
MODEL_SUMMARY  = get_model("equity", "summary")

# --- Constants ---
SECTION_ORDER = [
    "business_description",
    "industry_analysis",
    "financial_analysis",
    "valuation_analysis",
    "investment_risks",
    "competitor_analysis",
    "esg_analysis"
]

SECTION_PROMPTS = {
    "business_description": (
        "Focus on Company Economics (inputs/outputs), Revenue Models, and Key Drivers. "
        "Describe the product mix. Use the Company Profile data for the official business description "
        "and ownership structure. IMPORTANT: mention key strategic investors/shareholders if present. "
        "Include technology products, proprietary software, and hardware infrastructure details. "
        "Explicitly detail Supply Chain dependencies and Major Customer/Revenue Concentration Risks extracted from 10-K filings. "
        "Include insights from the Latest Earnings Call regarding management tone and operational updates."
    ),
    "industry_analysis": (
        "Analyze Industry Dynamics. Use **Porter's 5 Forces** (Suppliers, Buyers, Substitutes, "
        "New Entrants, Rivalry). Identify the **Moat** and **Competitive Positioning**. "
        "Consider the ownership structure (strategic investors) as a potential moat source. "
        "Include specific competitive advantages (performance benchmarks, technology features)."
    ),
    "financial_analysis": (
        "Analyze the underlying financial reality vs reported numbers. Assess **Quality of Earnings**. "
        "Use industry-specific ratios if applicable. USE THE MOST RECENT quarterly data and TTM figures, "
        "not just annual data. Always reference the specific fiscal periods (e.g., Q3 FY2025, TTM). "
        "Include backlog/contract details if available from research. Analyze debt structure in detail. "
        "You MUST include specific Forward Guidance numbers and Q&A sentiment extracted from the Latest Earnings Call."
    ),
    "valuation_analysis": (
        "Perform **Relative Valuation** (vs Peers) AND **Intrinsic/Absolute Valuation** (Conceptual DCF). "
        "Be skeptical of market pricing. Use the peer comparison data for multiples benchmarking. "
        "Include bull/base/bear scenario analysis with probability weights."
    ),
    "investment_risks": (
        "Identify specific **Operational**, **Financial**, and **Regulatory** risks. "
        "Do not just extrapolate past trends. What breaks the thesis? "
        "Include recent litigation, class-action lawsuits, execution delays, and customer concentration risks. "
        "Be specific — name names, cite events, reference amounts."
    ),
    "esg_analysis": (
        "Analyze **Environmental** (Carbon/Waste/Energy consumption), **Social** (Labor/Safety), and "
        "**Governance** (Board/Alignment/Insider activity) factors. "
        "Include data center power consumption, cooling infrastructure, and sustainability initiatives. "
        "You MUST include a summary of Recent Insider Trading Activity (Form 4), quantifying aggregate buying/selling volume and highlighting any major insider moves."
    ),
    "competitor_analysis": (
        "Provide a detailed **Competitor Analysis**. "
        "Use the specific peer data provided to compare the target against 'Giants' (Review relative size/metrics) "
        "and 'Disruptors' (Technology/Innovation threats). "
        "Include a comparative table if data is available."
    ),
}


# Note: condense_context and read_file_safe are imported from src.shared.text_utils


# ---------------------------------------------------------------------------
# Research Query System — Two-Phase: Generic Templates + LLM Follow-Up
# ---------------------------------------------------------------------------

def build_research_queries(
    ticker: str,
    company_name: str,
    sector: str,
    industry: str,
) -> Dict[str, List[str]]:
    """
    Builds generalized research queries that work for ANY ticker/industry.

    Phase 1 queries use only {name}, {ticker}, {sector}, {industry} —
    no domain-specific jargon. Domain-specific depth is handled by Phase 2.

    Args:
        ticker: Stock ticker symbol.
        company_name: Full company name.
        sector: GICS sector.
        industry: GICS industry.

    Returns:
        Dict mapping category name -> list of query strings.
    """
    name = company_name or ticker
    ind = industry or sector or "industry"

    return {
        "business_strategy": [
            f"{name} {ticker} business model revenue streams how does it make money",
            f"{name} {ticker} products services offerings platform",
            f"{name} {ticker} competitive advantage moat differentiation",
            f"{name} {ticker} market position {ind} industry ranking",
            f"{name} {ticker} customer base key clients contracts",
            f"{name} {ticker} expansion plans new markets growth strategy",
            f"{name} company history founding story background",
        ],
        "financial_performance": [
            f"{name} {ticker} latest quarterly earnings results revenue",
            f"{name} {ticker} revenue growth rate trajectory acceleration",
            f"{name} {ticker} margins profitability EBITDA operating income",
            f"{name} {ticker} balance sheet debt leverage ratio capital structure",
            f"{name} {ticker} capital expenditure capex investment spending",
            f"{name} {ticker} free cash flow generation burn rate liquidity",
            f"{name} {ticker} management guidance outlook forecast",
        ],
        "ownership_governance": [
            f"{name} {ticker} major shareholders largest institutional ownership",
            f"{name} {ticker} strategic investors equity stake partnership",
            f"{name} {ticker} insider buying selling executive stock transactions",
            f"{name} {ticker} board of directors governance structure",
            f"{name} {ticker} CEO management team leadership background",
        ],
        "competitive_landscape": [
            f"{name} {ticker} competitors top rivals comparison",
            f"{name} {ticker} market share {ind} industry position",
            f"{name} {ticker} competitive comparison strengths weaknesses",
            f"{name} {ticker} pricing power competitive dynamics",
            f"{ind} industry competitive landscape market leaders",
        ],
        "risks_headwinds": [
            f"{name} {ticker} risk factors challenges problems",
            f"{name} {ticker} lawsuit litigation legal proceedings regulatory",
            f"{name} {ticker} analyst downgrade bearish concerns sell",
            f"{name} {ticker} operational risk execution delays setbacks",
            f"{name} {ticker} customer concentration supplier dependency",
        ],
        "catalysts_events": [
            f"{name} {ticker} upcoming catalysts events milestones",
            f"{name} {ticker} new product launch expansion initiative",
            f"{name} {ticker} partnerships strategic alliances deals acquisitions",
            f"{name} {ticker} upcoming earnings conference investor day",
        ],
        "analyst_sentiment": [
            f"{name} {ticker} analyst rating price target consensus",
            f"{name} {ticker} bull case bear case investment thesis",
            f"{name} {ticker} wall street analyst coverage upgrade downgrade",
            f"{name} {ticker} short interest sentiment institutional flows",
        ],
        "industry_macro": [
            f"{ind} industry market size outlook TAM growth forecast",
            f"{ind} industry trends tailwinds headwinds dynamics",
            f"{ind} {sector} regulatory environment policy impact",
            f"{ind} industry disruption innovation technology shifts",
        ],
    }


def generate_followup_queries(
    ticker: str,
    company_name: str,
    sector: str,
    industry: str,
    profile_text: str,
    initial_snippets: str,
    categorized_peers: Dict[str, List[str]] = {},
) -> List[str]:
    """
    Phase 2: Uses the LLM to generate company-specific deep-dive queries
    based on what we already know about the company.

    Args:
        ticker: Stock ticker symbol.
        company_name: Full company name.
        sector: GICS sector.
        industry: GICS industry.
        profile_text: Company profile markdown text.
        initial_snippets: Concatenated snippets from Phase 1 search results.
        categorized_peers: Dict of Giants/Peers/Disruptors.

    Returns:
        List of 15-25 targeted follow-up search queries.
    """
    name = company_name or ticker

    condensed_profile = condense_context(
        profile_text, max_chars=10000,
        purpose=f"generating follow-up research queries for {name}",
        focus_areas="products, technology, partnerships, ownership, competitive position",
    )
    condensed_snippets = condense_context(
        initial_snippets, max_chars=20000,
        purpose=f"generating follow-up research queries for {name}",
        focus_areas="specific facts, names, numbers, events, risks not yet covered",
    )

    prompt = f"""You are a senior equity research analyst preparing a deep-dive on {name} ({ticker}).
Sector: {sector} | Industry: {industry}

You have already gathered the following information:

## Company Profile
{condensed_profile}

## Initial Research Snippets
{condensed_snippets}

## Identified Peers
Giants: {', '.join(categorized_peers.get('Giants', []))}
Direct Peers: {', '.join(categorized_peers.get('Peers', []))}
Disruptors/Private: {', '.join(categorized_peers.get('Disruptors', []))}

---

Based on what you've learned, generate 15-25 SPECIFIC follow-up search queries to find
information that a professional equity research report would need. Focus on:

1. **Company-specific products/technology** — name actual products, platforms, proprietary systems
2. **Key strategic relationships** — name actual partners, investors, customers, suppliers
3. **Specific financial events** — name actual deal sizes, contract values, filing dates
4. **Known risks and controversies** — name actual lawsuits, regulatory actions, analyst downgrades
5. **Industry-specific metrics** — use actual industry terminology and KPIs
6. **Recent M&A and partnerships** — name actual acquisition targets, deal values
7. **Competitive Benchmarking** — Compare {ticker} vs SPECIFIC peers (e.g. "vs {categorized_peers.get('Giants', ['Giant'])[0]} margins", "vs {categorized_peers.get('Disruptors', ['Startup'])[0]} technology")

IMPORTANT RULES:
- Every query should contain the company name or ticker
- Use SPECIFIC names, products, and terms you learned from the profile and snippets
- Do NOT use generic queries — those were already done in Phase 1
- Target information gaps — what's missing from the snippets above?
- Include 3-4 queries specifically about BEAR CASE / risks / what could go wrong
- Include 3-4 queries comparing against specific Giants or Disruptors

Output ONLY the queries, one per line, no numbering or formatting.
"""

    try:
        response = get_llm(MODEL_FOLLOWUP).invoke([HumanMessage(content=prompt)])
        lines = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
        queries = [q for q in lines if len(q) > 15 and not q.startswith("#") and not q.startswith("*")]
        print(f"LLM generated {len(queries)} follow-up queries")
        return queries[:25]
    except Exception as e:
        print(f"Follow-up query generation failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

# _read_file is now read_file_safe from src.shared.text_utils
_read_file = read_file_safe


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def planner(state: AgentState) -> dict:
    """Decomposes the request. Initializes empty sections."""
    recent_messages = state["messages"][-10:]
    system_prompt = SystemMessage(content="""
    You are a financial research planner. 
    Analyze the user's request and create a detailed execution plan.
    Identify the main ticker symbol.
    Leave the peers list EMPTY — peers will be discovered automatically using data-driven analysis.
    
    You must respond in valid JSON matching this exact schema:
    {
        "ticker": "AAPL",
        "subtasks": ["technical", "fundamental", "peers"],
        "peers": []
    }
    """)
    structured_llm = get_llm(MODEL_PLAN).with_structured_output(AnalysisPlan, method="json_mode")
    try:
        messages = [system_prompt] + recent_messages
        plan = structured_llm.invoke(messages)
        return {
            "plan": plan,
            "ticker": plan.ticker,
            "period": "1y",
            "source_metadata": {},
            "sections": {},
            "research_data": {},
            "research_brief": {},
            "error": None
        }
    except Exception as e:
        return {"error": f"Failed to generate plan: {str(e)}"}


def fetch_data(state: AgentState) -> dict:
    """Fetches historical market data (Technical)."""
    print("\n--- [Equity] Phase: Fetching Market Data ---")
    if state.get("error"): return state
    ticker = state["ticker"]
    period = state.get("period", "1y")
    try:
        file_path = fetch_historical_data(ticker, period)
        return {"market_data_path": file_path, "error": None}
    except Exception as e:
        return {"error": f"Failed to fetch market data: {str(e)}"}


def generate_chart(state: AgentState) -> dict:
    """Generates a chart."""
    print("\n--- [Equity] Phase: Generating Price Chart ---")
    if state.get("error"): return state
    data_path = state["market_data_path"]
    ticker = state["ticker"]
    try:
        chart_path = plot_price_history(data_path, ticker)
        return {"chart_path": chart_path, "error": None}
    except Exception as e:
        return {"error": f"Failed to generate chart: {str(e)}"}


def fetch_company_profile_node(state: AgentState) -> dict:
    """Fetches company identity, ownership structure, and institutional holders."""
    print("\n--- [Equity] Phase: Fetching Company Profile ---")
    if state.get("error"): return state
    ticker = state["ticker"]
    metadata_store = state.get("source_metadata", {})
    try:
        file_path, metadata = fetch_company_profile(ticker)
        new_metadata = metadata_store.copy()
        new_metadata["company_profile"] = metadata
        return {
            "company_profile_path": file_path,
            "source_metadata": new_metadata,
            "error": None,
        }
    except Exception as e:
        print(f"Warning: Company profile fetch failed: {e}")
        return {"company_profile_path": None, "error": None}


def fetch_financials(state: AgentState) -> dict:
    """Fetches fundamental data (annual + quarterly + TTM)."""
    print("\n--- [Equity] Phase: Fetching Financials ---")
    if state.get("error"): return state
    ticker = state["ticker"]
    metadata_store = state.get("source_metadata", {})
    try:
        file_path, metadata = fetch_financial_metrics(ticker)
        new_metadata = metadata_store.copy()
        new_metadata["financials"] = metadata
        return {"financials_path": file_path, "source_metadata": new_metadata, "error": None}
    except Exception as e:
        print(f"Warning: Financials fetch failed: {e}")
        return {"financials_path": None, "error": None}


def research_qualitative(state: AgentState) -> dict:
    """
    Two-phase deep research node.

    Phase 1: Run 40+ generic queries (text + news), collect snippets + full-text reads.
    Phase 2: Feed snippets + profile to LLM → generate 15-25 company-specific follow-up
             queries → deep-search those for domain-specific depth.
    """
    print("\n--- [Equity] Phase: Deep Qualitative Research ---")
    if state.get("error"): return state
    ticker = state["ticker"]

    profile_path = state.get("company_profile_path")
    company_name = ""
    sector_hint = ""
    industry_hint = ""
    profile_text = ""
    if profile_path and os.path.exists(profile_path):
        with open(profile_path, "r") as f:
            profile_text = f.read()
        for line in profile_text.split("\n"):
            if line.startswith("# Company Profile:"):
                company_name = line.replace("# Company Profile:", "").strip()
                if "(" in company_name:
                    company_name = company_name[:company_name.index("(")].strip()
                break
        metadata = state.get("source_metadata", {}).get("company_profile", {})
        sector_hint = metadata.get("sector", "")
        industry_hint = metadata.get("industry", "")

    # PHASE 1: Generic queries (Parallelized)
    query_categories = build_research_queries(ticker, company_name, sector_hint, industry_hint)

    all_text_queries: List[str] = []
    all_news_queries: List[str] = []

    for category, queries in query_categories.items():
        all_text_queries.extend(queries)
        if category in ("financial_performance", "risks_headwinds", "catalysts_events",
                         "analyst_sentiment", "ownership_governance"):
            all_news_queries.extend(queries[:3])

    print(f"Phase 1: {len(all_text_queries)} text queries + {len(all_news_queries)} news queries")

    def run_text_search() -> list:
        return deep_search(
            queries=all_text_queries,
            urls_per_query=3, max_workers=5, max_chars_per_url=8000,
            search_delay=0.5, use_news=False,
        )

    def run_news_search() -> list:
        return deep_search(
            queries=all_news_queries,
            urls_per_query=3, max_workers=5, max_chars_per_url=6000,
            search_delay=0.5, use_news=True,
        )

    def run_sec_search() -> list:
        try:
            results = search_and_read_filings(
                company_name=company_name or ticker, ticker=ticker,
                form_types=["10-K", "10-Q", "8-K"], max_filings=5, max_chars_per_filing=12000,
            )
            print(f"SEC: Found {len(results)} filings")
            return results
        except Exception as e:
            print(f"SEC filing search failed: {e}")
            return []

    def run_insider_trading() -> tuple:
        try:
            print(f"Fetching insider trading for {ticker}...")
            return fetch_insider_activity(ticker)
        except Exception as e:
            print(f"Insider trading fetch failed: {e}")
            return "", {}

    def run_earnings_call() -> tuple:
        try:
            print(f"Fetching earnings call insights for {ticker}...")
            return research_earnings_call(ticker, company_name)
        except Exception as e:
            print(f"Earnings call fetch failed: {e}")
            return "", {}

    text_results: list = []
    news_results: list = []
    sec_results: list = []
    insider_results: tuple = ("", {})
    earnings_results: tuple = ("", {})

    with ThreadPoolExecutor(max_workers=5) as executor:
        future_text = executor.submit(run_text_search)
        future_news = executor.submit(run_news_search)
        future_sec = executor.submit(run_sec_search)
        future_insider = executor.submit(run_insider_trading)
        future_earnings = executor.submit(run_earnings_call)
        
        text_results = future_text.result()
        news_results = future_news.result()
        sec_results = future_sec.result()
        insider_results = future_insider.result()
        earnings_results = future_earnings.result()

    phase1_snippets = "\n".join([
        f"- [{r['title']}]({r['url']}): {r.get('snippet', '') or (r.get('full_text', '') or '')[:300]}"
        for r in text_results + news_results
        if r.get("title")
    ])

    # PHASE 2: LLM follow-up queries (Parallelized)
    followup_queries = generate_followup_queries(
        ticker=ticker, company_name=company_name,
        sector=sector_hint, industry=industry_hint,
        profile_text=profile_text, initial_snippets=phase1_snippets,
        categorized_peers={},
    )

    followup_results: list = []
    if followup_queries:
        print(f"Phase 2: Running {len(followup_queries)} LLM-generated follow-up queries")

        def run_followup_text() -> list:
            return deep_search(
                queries=followup_queries, urls_per_query=3, max_workers=5,
                max_chars_per_url=8000, search_delay=0.5, use_news=False,
            )

        def run_followup_news() -> list:
            return deep_search(
                queries=followup_queries[:8], urls_per_query=2, max_workers=5,
                max_chars_per_url=6000, search_delay=0.5, use_news=True,
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_f_text = executor.submit(run_followup_text)
            future_f_news = executor.submit(run_followup_news)
            f_text_results = future_f_text.result()
            f_news_results = future_f_news.result()

        followup_results = f_text_results + f_news_results

    # Consolidate all research
    research_data: Dict[str, str] = {}
    articles_read = 0
    articles_with_full_text = 0

    for r in text_results + news_results + followup_results:
        query = r["query"]
        title = r["title"]
        url = r["url"]
        full_text = r.get("full_text")
        snippet = r.get("snippet", "")

        articles_read += 1

        if full_text:
            articles_with_full_text += 1
            content = f"### {title}\nSource: {url}\n\n{full_text}\n"
        else:
            content = f"### {title}\nSource: {url}\n\nSnippet: {snippet}\n"

        if query in research_data:
            research_data[query] += "\n---\n" + content
        else:
            research_data[query] = content

    for f in sec_results:
        key = f"SEC Filing: {f['form_type']} ({f['filing_date']})"
        research_data[key] = f"### SEC {f['form_type']} — {f['filing_date']}\nURL: {f['url']}\n\n{f['text']}\n"

    # Extract 10-K Supply Chain data via LLM
    if sec_results:
        print("Extracting Supply Chain & Customer concentration from SEC filings...")
        combined_sec_text = "\n\n".join([f["text"] for f in sec_results])
        sc_insights = extract_supply_chain_data(ticker, combined_sec_text)
        research_data["Supply Chain & Customer Concentration"] = f"### Supply Chain & Major Customers (10-K/Q)\n\n{sc_insights}"

    # Inject Earnings Call insights
    er_text, er_meta = earnings_results
    if er_text:
        research_data["Latest Earnings Call Transcript Highlights"] = f"### Earnings Call Insights\n\n{er_text}"

    # We read the insider trading markdown into research_data and state source_metadata
    new_metadata = state.get("source_metadata", {}).copy()
    insider_md_path, insider_meta = insider_results
    if insider_md_path and os.path.exists(insider_md_path):
        with open(insider_md_path, "r") as f:
            insider_text = f.read()
        research_data["Recent Insider Trading Activity"] = insider_text
        new_metadata["insider_trading"] = insider_meta

    total_chars = sum(len(v) for v in research_data.values())
    print(f"Research complete: {articles_read} articles ({articles_with_full_text} full-text), "
          f"{len(sec_results)} SEC filings, {len(followup_queries)} follow-up queries, Phase 2 data sources loaded.")
    print(f"Total raw research text: {total_chars:,} characters (~{total_chars // 4:,} tokens)")

    return {
        "research_data": research_data,
        "error": None,
        "source_metadata": new_metadata
    }


def synthesize_research(state: AgentState) -> dict:
    """
    Map-Reduce LLM synthesis: converts massive raw research into focused section briefs.

    Map step: For each batch of raw text, extract key facts relevant to the company.
    Reduce step: For each report section, merge extracted facts into a focused brief.
    """
    print("\n--- [Equity] Phase: Synthesizing Research (Map/Reduce) ---")
    if state.get("error"): return state

    ticker = state["ticker"]
    research_data = state.get("research_data", {})

    if not research_data:
        print("Warning: No research data to synthesize")
        return {"research_brief": {}, "error": None}

    # --- MAP STEP ---
    all_text_items = list(research_data.values())

    CHUNK_SIZE = 50000
    chunks: List[str] = []
    current_chunk = ""
    for text in all_text_items:
        if len(current_chunk) + len(text) > CHUNK_SIZE:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = text
        else:
            current_chunk += "\n\n---\n\n" + text
    if current_chunk:
        chunks.append(current_chunk)

    print(f"Synthesis MAP: Processing {len(chunks)} chunks of research text")

    import concurrent.futures

    def _map_chunk(args):
        i, chunk = args
        map_prompt = f"""
You are a research analyst extracting facts about {ticker}.

Below is raw text from web articles, news, and SEC filings about this company.
Extract EVERY important fact, number, date, name, and quote. Be thorough.
Focus on:
- Financial metrics (revenue, backlog, margins, debt levels, specific numbers)
- Ownership and strategic investor details (who owns how much, why it matters)
- Products, technology, and competitive advantages (specific names, benchmarks)
- Risks, lawsuits, delays, and negative events (specific details)
- Strategic partnerships and key customers (names, contract values)
- Industry dynamics and competitive positioning
- Management commentary and guidance

Output a bullet-point list of the **most critical** research points. 
For financial data, focus only on the **most important data points** (e.g., Guidance, Revenue, Margins, RPO, Debt), but for the data you select, you MUST replicate the numbers, tables, and fiscal periods EXACTLY as they appear.
Do NOT approximate — choose the most important data and replicate it precisely.

RAW TEXT:
{chunk}
"""
        try:
            response = get_llm(MODEL_MAP).invoke([HumanMessage(content=map_prompt)])
            print(f"  MAP chunk {i+1}/{len(chunks)}: extracted {len(response.content)} chars of facts")
            return response.content
        except Exception as e:
            print(f"  MAP chunk {i+1} failed: {e}")
            return ""

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        extracted_facts = list(executor.map(_map_chunk, enumerate(chunks)))
        
    extracted_facts = [f for f in extracted_facts if f]
    all_facts = "\n\n".join(extracted_facts)
    print(f"Synthesis MAP complete: {len(all_facts):,} chars of extracted facts")

    # Safeguard against blowing past the LLM context window limits in the Reduce phase
    if len(all_facts) > 150000:
        print(f"Facts exceed 150k chars. Condensing down to a safe limit...")
        all_facts = condense_context(
            all_facts, 
            max_chars=120000, 
            purpose=f"consolidated and aggregated research facts for {ticker}"
        )
        print(f"Facts successfully condensed to {len(all_facts):,} chars")

    # --- REDUCE STEP ---
    research_brief: Dict[str, str] = {}

    def _reduce_section(section_key):
        section_label = section_key.replace("_", " ").title()
        reduce_prompt = f"""
You are preparing a research brief for the **{section_label}** section of an equity research report on {ticker}.

Below are extracted facts from 50+ web articles, news, and SEC filings.
Select and organize ONLY the facts relevant to {section_label}.

SECTION GUIDANCE:
{SECTION_PROMPTS[section_key]}

Include:
- Specific numbers, dates, percentages
- Named entities (people, companies, products)
- Direct quotes when available
- Contradictory viewpoints (bull vs bear)

Output a well-organized brief with headers and bullet points.
This brief will be used by another analyst to write the final section.

EXTRACTED FACTS:
{all_facts}
"""
        try:
            response = get_llm(MODEL_REDUCE).invoke([HumanMessage(content=reduce_prompt)])
            print(f"  REDUCE {section_key}: {len(response.content)} chars")
            return section_key, response.content
        except Exception as e:
            print(f"  REDUCE {section_key} failed: {e}")
            return section_key, ""

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(SECTION_ORDER)) as executor:
        futures = {executor.submit(_reduce_section, key): key for key in SECTION_ORDER}
        for future in concurrent.futures.as_completed(futures):
            key, content = future.result()
            research_brief[key] = content

    print(f"Synthesis REDUCE complete: {sum(len(v) for v in research_brief.values()):,} chars total brief")

    return {"research_brief": research_brief, "extracted_facts": all_facts, "error": None}


def review_and_expand_peers(state: AgentState) -> dict:
    """
    Iterative Peer Analysis Node.
    
    1. Infer potential peers from the 'extracted_facts' (synthesis output).
    2. Validate and categorize them via `evaluate_candidates`.
    3. Generate specific comparison queries for the valid peers.
    4. Run a targeted deep search.
    5. Generate a 'competitor_analysis' brief for the final report.
    """
    print("\n--- [Equity] Phase: Peer Analysis ---")
    if state.get("error"): return state
    
    ticker = state["ticker"]
    extracted_facts = state.get("extracted_facts", "")
    
    if not extracted_facts:
        print("Warning: No facts available for peer inference.")
        return state

    print("--- Reviewing and Expanding Peers ---")

    # 1. Infer Peers
    extracted_candidates = extract_peers_from_text(extracted_facts[:25000])
    print(f"Inferred candidates from Key Facts: {extracted_candidates}")
    
    # 2. Validate/Categorize
    categorized_peers = evaluate_candidates(extracted_candidates, ticker)
    print(f"Validated Categories: {categorized_peers}")
    
    # 3. Targeted Queries
    peer_queries: List[str] = []
    
    for p in categorized_peers.get("Giants", []):
         peer_queries.append(f"{ticker} vs {p} revenue growth margin comparison")
         peer_queries.append(f"{ticker} competitive advantage vs {p}")
         
    for p in categorized_peers.get("Peers", []):
         peer_queries.append(f"{ticker} vs {p} product feature comparison")
         peer_queries.append(f"{ticker} market share vs {p}")

    for p in categorized_peers.get("Disruptors", []):
         clean_name = p.replace(" (Private)", "")
         peer_queries.append(f"{ticker} vs {clean_name} technology comparison")
         peer_queries.append(f"{clean_name} funding valuation revenue vs {ticker}")

    print(f"Generated {len(peer_queries)} targeted peer queries.")
    
    # 4. Run Search & Extract
    peer_research_facts = ""
    
    if peer_queries:
        try:
            print(f"Running peer research with {len(peer_queries)} queries...")
            results = deep_search(
                queries=peer_queries[:12],
                urls_per_query=2, max_workers=4, use_news=False
            )
            
            raw_peer_text = ""
            for r in results:
                raw_peer_text += f"### {r['title']}\n{r.get('snippet', '')}\n\n"
            
            if raw_peer_text:
                extraction_prompt = f"""
                You are analyzing competitors for {ticker}.
                Extract key comparative facts from the search results below.
                
                Focus on:
                - Specific revenue, growth, and margin comparisons vs {ticker}.
                - Product feature differences and benchmarks.
                - Market share data.
                - Specific threats from disruptors (technology claims, funding).
                
                RAW SEARCH RESULTS:
                {raw_peer_text[:25000]}
                """
                try:
                    extract_response = get_llm(MODEL_MAP).invoke([HumanMessage(content=extraction_prompt)])
                    peer_research_facts = extract_response.content
                    print(f"Extracted {len(peer_research_facts)} chars of peer facts.")
                except Exception as e:
                    print(f"Peer fact extraction failed: {e}")
                    peer_research_facts = raw_peer_text[:5000]

        except Exception as e:
            print(f"Peer search failed: {e}")

    # 5. Summarize into Brief
    competitor_brief_prompt = f"""
    You are writing the **Competitor Analysis** brief for {ticker}.
    
    Use the following data:
    
    ## Validated Peers
    Giants: {categorized_peers.get('Giants')}
    Direct Peers: {categorized_peers.get('Peers')}
    Disruptors: {categorized_peers.get('Disruptors')}
    
    ## Extracted Peer Research (New Findings)
    {peer_research_facts}
    
    ## Context from Main Research
    {extracted_facts[:10000]}
    
    Requirements:
    - Compare {ticker} against its Giants (David vs Goliath analysis).
    - Compare {ticker} against Direct Peers (Head-to-head).
    - Analyze threats from Disruptors/Startups.
    - Output a structured brief with headers.
    - CITE specific numbers and metrics where available.
    """
    
    new_briefs = state.get("research_brief", {}).copy()
    try:
        response = get_llm(MODEL_REDUCE).invoke([HumanMessage(content=competitor_brief_prompt)])
        new_briefs["competitor_analysis"] = response.content
    except Exception as e:
        print(f"Competitor brief generation failed: {e}")
        new_briefs["competitor_analysis"] = "Analysis failed."

    # 6. Fetch Metrics (for the table in the report)
    new_metadata = state.get("source_metadata", {}).copy()
    peer_data_path = None
    try:
        peer_data_path, metadata = fetch_peer_metrics(categorized_peers, target_ticker=ticker)
        new_metadata["peers"] = metadata
    except Exception as e:
        print(f"Peer metrics fetch failed: {e}")

    return {
        "categorized_peers": categorized_peers,
        "research_brief": new_briefs,
        "source_metadata": new_metadata,
        "peer_data_path": peer_data_path
    }


def generate_section(state: AgentState) -> dict:
    """Generates all sections in parallel using synthesized research brief."""
    print("\n--- [Equity] Phase: Generating Report Sections ---")
    if state.get("error"): return state

    ticker = state["ticker"]
    fin_path = state.get("financials_path")
    peer_path = state.get("peer_data_path")
    profile_path = state.get("company_profile_path")

    financials_content = _read_file(fin_path, max_chars=12000)
    peer_content = _read_file(peer_path, max_chars=6000)
    profile_content = _read_file(profile_path, max_chars=5000)

    research_brief = state.get("research_brief", {})
    current_sections = state.get("sections", {}) or {}
    new_sections = current_sections.copy()

    import concurrent.futures

    def _generate_single_section(section_key):
        if section_key in current_sections:
            return section_key, current_sections[section_key]

        section_brief = research_brief.get(section_key, "No research brief available.")

        prompt = f"""
        You are a Senior Equity Research Analyst for AlphaSeeker.
        
        ## DATA SOURCES (Context)
        
        ### Company Profile (Ground Truth for Identity)
        {profile_content}
        
        ### Financials (Quantitative)
        {financials_content}
        
        ### Peer Comparison
        {peer_content}
        
        ---
        
        ## YOUR TASK
        Generate the **{section_key.replace('_', ' ').upper()}** section for {ticker}.
        
        ### Deep Research Brief (Specific to this section)
        {section_brief}
        
        ### Section Guidance
        {SECTION_PROMPTS[section_key]}
        
        ## CRITICAL INSTRUCTIONS
        - Output must be a valid 'ResearchSection' object.
        - Title should be professional.
        - Content must be detailed, use Markdown, and cite data carefully with [n].
        - ALWAYS use the Company Profile as the ground truth for what the company does.
        - Use the MOST RECENT financial data available (quarterly/TTM preferred over annual).
        - The Deep Research Brief contains facts extracted from real web articles — USE THEM.
          Specific numbers, names, events, and quotes in the brief are verified from sources.
        - Include specific technology products, partnerships, and events BY NAME.
        - Be LONG and DETAILED. Target 1000+ words. This is a professional research report.
        """
        try:
            structured_llm = get_llm(MODEL_SECTION).with_structured_output(ResearchSection)
            section = structured_llm.invoke([HumanMessage(content=prompt)])
            print(f"Section '{section_key}' generated ({len(section.content)} chars)")
            return section_key, section
        except Exception as e:
            print(f"Failed to generate section {section_key}: {str(e)}")
            return section_key, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(SECTION_ORDER)) as executor:
        futures = {executor.submit(_generate_single_section, key): key for key in SECTION_ORDER}
        for future in concurrent.futures.as_completed(futures):
            key, section = future.result()
            if section:
                new_sections[key] = section

    # Check if we generated everything needed
    missing = [k for k in SECTION_ORDER if k not in new_sections]
    if missing:
        return {"error": f"Failed to generate sections: {missing}"}

    return {"sections": new_sections, "error": None}


def generate_summary(state: AgentState) -> dict:
    """Generates the Investment Summary based on all completed sections."""
    print("\n--- [Equity] Phase: Generating Final Summary ---")
    if state.get("error"): return state

    ticker = state["ticker"]
    sections = state.get("sections", {})

    full_text = "\n\n".join([f"## {s.title}\n{s.content}" for s in sections.values()])

    profile_path = state.get("company_profile_path")
    profile_content = _read_file(profile_path, max_chars=3000)

    research_brief = state.get("research_brief", {})
    brief_text = "\n\n".join([f"### {k}\n{v}" for k, v in research_brief.items()])

    prompt = f"""
    You are a Senior Equity Research Analyst.
    All detailed analysis sections for {ticker} are complete.
    Now, write the **Investment Summary** (The "Top Sheet").
    
    ## COMPANY PROFILE (Ground Truth)
    {profile_content}
    
    ## FULL ANALYSIS
    {full_text}
    
    ## KEY RESEARCH FINDINGS
    {condense_context(brief_text, max_chars=10000, purpose=f'investment summary synthesis for {ticker}', focus_areas='thesis, catalysts, risks, valuation, key numbers')}
    
    ## INSTRUCTIONS
    1. **Investment Summary**: Synthesize the key insights. Start with what the company IS
       (use the Company Profile, not your imagination). Be specific about strategic moats.
    2. **Mispricing Thesis**: Explicitly explain WHY the market is wrong (Price != Value).
       Include specific bull AND bear arguments from analyst coverage.
    3. **Catalysts**: What specific events will unlock value? Include dates when possible.
    4. **Recommendation**: BUY/SELL/HOLD and Target Price with scenario analysis.
    5. **References**: List the actual data sources used.
    
    ## FORMATTING RULES
    - **Negative Numbers**: ALWAYS use a leading minus sign (e.g., "-$500M", "-$1.2B"). NEVER use parentheses like "($500M)" or "$(500M)".
    - **Currency**: Standardize on "$" for USD.
    - **Markdown**: Do not use LaTeX-style math formatting (no `$(...)`). Use standard bold/italics.
    """ 
    
    class ExecutiveSummary(BaseModel):
        target_price: str
        recommendation: str
        investment_summary: str
        mispricing_thesis: str
        key_catalysts: str
        references: List[str]
        
    structured_llm = get_llm(MODEL_SUMMARY).with_structured_output(ExecutiveSummary)
    
    try:
        summary = structured_llm.invoke([HumanMessage(content=prompt)])
        
        final_report = ResearchReport(
            ticker=ticker,
            target_price=summary.target_price,
            recommendation=summary.recommendation,
            investment_summary=summary.investment_summary,
            mispricing_thesis=summary.mispricing_thesis,
            key_catalysts=summary.key_catalysts,
            business_description=sections["business_description"],
            industry_analysis=sections["industry_analysis"],
            financial_analysis=sections["financial_analysis"],
            valuation_analysis=sections["valuation_analysis"],
            investment_risks=sections["investment_risks"],
            esg_analysis=sections["esg_analysis"],
            competitor_analysis=sections["competitor_analysis"],
            references=summary.references
        )
        
        return {"report_content": final_report, "error": None}
        
    except Exception as e:
        return {"error": f"Failed to generate summary: {str(e)}"}


def verify_content(state: AgentState) -> dict:
    """Verifies the generated report."""
    print("\n--- [Equity] Phase: Verifying Output ---")
    if state.get("error"): return state
    report = state.get("report_content")
    if not report: return {"error": "No report content found during verification"}
    return state


def save_report(state: AgentState) -> dict:
    """Saves the report to Markdown."""
    print("\n--- [Equity] Phase: Saving Final Report ---")
    if state.get("error"): return state
    
    report = state.get("report_content")
    ticker = report.ticker
    chart_path = state.get("chart_path", "")
    
    md = f"# Equity Research: {ticker}\n\n"
    md += f"**Rating**: {report.recommendation} | **Target**: {report.target_price}\n\n"
    md += f"![Price Chart]({chart_path})\n\n"
    
    md += f"## Investment Summary\n{report.investment_summary}\n\n"
    md += f"### Mispricing Thesis\n{report.mispricing_thesis}\n\n"
    md += f"### Key Catalysts\n{report.key_catalysts}\n\n"
    
    md += "---\n\n"
    
    sections = [
        report.business_description,
        report.industry_analysis,
        report.financial_analysis,
        report.valuation_analysis,
        report.competitor_analysis,
        report.esg_analysis,
        report.investment_risks
    ]
    
    def _clean_section_content(title: str, content: str) -> str:
        """Post-process LLM section content to fix common formatting bugs:
        1. Strip duplicate title if the LLM started its content with the same title.
        2. Demote all headings by one level (H1->H2, H2->H3, ...) so they nest
           correctly under the H2 section header added by save_report.
        3. Remove debug "Word Count:" paragraphs the LLM occasionally hallucinates.
        """
        import re

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

    for s in sections:
        cleaned = _clean_section_content(s.title, s.content)
        md += f"## {s.title}\n{cleaned}\n\n"
        
    md += "## References\n"
    for r in report.references:
        md += f"- {r}\n"
        
    report_dir = os.path.join(os.getcwd(), "reports")
    os.makedirs(report_dir, exist_ok=True)
    prompt_text = extract_prompt_text(state.get("messages"))
    filename = build_prompt_report_filename(
        prompt_text=prompt_text,
        fallback_stem=f"{ticker}_initiation_report",
        suffix="equity",
    )
    report_path = os.path.join(report_dir, filename)
    
    with open(report_path, "w") as f:
        f.write(md)
        
    return {"report_path": report_path}


# ---------------------------------------------------------------------------
# Edge Functions
# ---------------------------------------------------------------------------

def check_error(state: AgentState) -> Literal["continue", "end"]:
    """Guards every node transition."""
    if state.get("error"): return "end"
    return "continue"
