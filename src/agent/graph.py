"""
AlphaSeeker LangGraph workflow — Deep Research Edition.

Flow:
  planner → fetch_data → generate_chart → fetch_company_profile
    → fetch_financials → analyze_peers → research_qualitative (50+ queries, deep read)
    → synthesize_research (map-reduce LLM extraction)
    → generate_section (loop 6 times) → generate_summary → verify → save → END
"""

from typing import TypedDict, Annotated, Literal, Dict, List
import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

# Import our schema and tools
from src.schemas import AgentState, AnalysisPlan, ResearchReport, ResearchSection
from src.tools.market_data import fetch_historical_data
from src.tools.visualization import plot_price_history
from src.tools.financials import fetch_financial_metrics
from src.tools.peers import fetch_peer_metrics, discover_peers
from src.tools.web_search import search_web, search_news, deep_search
from src.tools.sec_filings import search_and_read_filings
from src.tools.company_profile import fetch_company_profile
from src.llm_manager import get_llm

# --- Model Assignments ---
# Change any of these to swap a step to a different model.
# Prefix determines provider: "gemini-*" → Google, "kimi-*" → Moonshot, "sf/*" → SiliconFlow
MODEL_PLAN     = "sf/Qwen/Qwen3-14B"   # trivial extraction
MODEL_CONDENSE = "sf/Qwen/Qwen3-14B"   # summarization / condensation
MODEL_FOLLOWUP = "sf/Qwen/Qwen3-14B"   # search query generation
MODEL_MAP      = "sf/Qwen/Qwen3-14B"   # fact extraction from chunks
MODEL_REDUCE   = "sf/Qwen/Qwen3-14B"   # organize facts into briefs
MODEL_SECTION  = "kimi-k2.5"                    # professional report writing
MODEL_SUMMARY  = "kimi-k2.5"                    # investment summary synthesis

# --- Constants ---
SECTION_ORDER = [
    "business_description",
    "industry_analysis",
    "financial_analysis",
    "valuation_analysis",
    "investment_risks",
    "esg_analysis"
]

SECTION_PROMPTS = {
    "business_description": (
        "Focus on Company Economics (inputs/outputs), Revenue Models, and Key Drivers. "
        "Describe the product mix. Use the Company Profile data for the official business description "
        "and ownership structure. IMPORTANT: mention key strategic investors/shareholders if present. "
        "Include technology products, proprietary software, and hardware infrastructure details."
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
        "Include backlog/contract details if available from research. Analyze debt structure in detail."
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
        "Include data center power consumption, cooling infrastructure, and sustainability initiatives."
    ),
}

# ---------------------------------------------------------------------------
# Intelligent Context Condensation (replaces hard truncation)
# ---------------------------------------------------------------------------

def condense_context(
    text: str,
    max_chars: int,
    purpose: str = "equity research analysis",
    focus_areas: str = "",
) -> str:
    """
    Intelligently handles long text: returns as-is if short enough,
    or calls the LLM to extract core information if too long.

    This replaces hard truncation (text[:N]) which loses information
    near the end of the text.

    Args:
        text: The input text to potentially condense.
        max_chars: Character budget. If text is shorter, return unchanged.
        purpose: Why the text is being condensed (helps LLM focus).
        focus_areas: Optional specific topics to prioritize.

    Returns:
        Original text (if short enough) or LLM-condensed version.
    """
    if not text or len(text) <= max_chars:
        return text

    # Text exceeds budget — LLM-condense it
    target_chars = int(max_chars * 0.9)  # Leave 10% buffer
    focus_instruction = ""
    if focus_areas:
        focus_instruction = f"\nPay special attention to: {focus_areas}"

    # Initial Prompt
    condense_prompt = f"""You are condensing a long document for {purpose}.
The original is {len(text):,} characters but the budget is ~{target_chars:,} characters.

RULES:
- Preserve ALL specific numbers, financial figures, dates, percentages, and dollar amounts
- Preserve ALL named entities (people, companies, products, locations)
- Preserve ALL facts about ownership stakes, contracts, partnerships, lawsuits
- Remove boilerplate, repeated information, and filler text
- Keep the most important and unique information
- Output in the same format (markdown/bullet points) as the input{focus_instruction}

CONDENSE THIS:
{text}
"""
    
    current_prompt = condense_prompt
    
    # Retry loop (max 2 attempts)
    for attempt in range(2):
        try:
            response = get_llm(MODEL_CONDENSE).invoke([HumanMessage(content=current_prompt)])
            condensed = response.content
            
            # Check length constraint
            if len(condensed) <= max_chars:
                print(f"  Condensed {len(text):,} → {len(condensed):,} chars ({purpose[:100]}...)")
                return condensed
            
            # Failed length check
            print(f"  Warning: Condensation (Attempt {attempt+1}) output {len(condensed):,} chars > limit {max_chars:,}")
            
            # Save debug info
            timestamp = int(time.time())
            debug_dir = "data/debug"
            os.makedirs(debug_dir, exist_ok=True)
            debug_file = f"{debug_dir}/condensation_fail_{timestamp}_{attempt}.txt"
            with open(debug_file, "w", encoding="utf-8") as f:
                f.write(f"--- ORIGINAL TEXT ({len(text)} chars) ---\n")
                f.write(text)
                f.write(f"\n\n--- FAILED CONDENSATION ({len(condensed)} chars) ---\n")
                f.write(condensed)
            print(f"  Saved debug context to {debug_file}")

            # Prepare retry prompt - asking to shorten the previous output
            current_prompt = f"""
Your previous summary was {len(condensed):,} characters, which exceeds the limit of {max_chars:,} characters.
Please shorten it significantly while keeping the key facts.

Previous Output:
{condensed}
"""
        except Exception as e:
            print(f"  Condensation attempt {attempt+1} failed ({e})")
            # If we hit an exception (e.g. API error), we proceed to fallback
            break

    # Fallback to strict truncation if retries failed or errored
    print(f"  Condensation failed after retries, falling back to strict truncation")
    return text[:max_chars]


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
    no domain-specific jargon (e.g., no "GPU", "hyperscaler", "drug pipeline").
    Domain-specific depth is handled by Phase 2 (LLM follow-up queries).

    Args:
        ticker: Stock ticker symbol.
        company_name: Full company name.
        sector: GICS sector (e.g., "Technology", "Healthcare").
        industry: GICS industry (e.g., "Semiconductors", "Biotechnology").

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
) -> List[str]:
    """
    Phase 2: Uses the LLM to generate company-specific deep-dive queries
    based on what we already know about the company.

    This is where domain-specific depth comes from — the LLM reads the profile
    and initial search snippets, then produces targeted follow-up queries
    that a human analyst would ask (e.g., "NVIDIA equity stake in CoreWeave"
    for CRWV, or "Ozempic patent cliff timeline" for Novo Nordisk).

    Args:
        ticker: Stock ticker symbol.
        company_name: Full company name.
        sector: GICS sector.
        industry: GICS industry.
        profile_text: Company profile markdown text.
        initial_snippets: Concatenated snippets from Phase 1 search results.

    Returns:
        List of 15-25 targeted follow-up search queries.
    """
    name = company_name or ticker

    # Condense long inputs instead of hard-clipping
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

---

Based on what you've learned, generate 15-25 SPECIFIC follow-up search queries to find
information that a professional equity research report would need. Focus on:

1. **Company-specific products/technology** — name actual products, platforms, proprietary systems
2. **Key strategic relationships** — name actual partners, investors, customers, suppliers
3. **Specific financial events** — name actual deal sizes, contract values, filing dates
4. **Known risks and controversies** — name actual lawsuits, regulatory actions, analyst downgrades
5. **Industry-specific metrics** — use actual industry terminology and KPIs
6. **Recent M&A and partnerships** — name actual acquisition targets, deal values
7. **Competitive benchmarking** — name actual competitors for head-to-head comparison

IMPORTANT RULES:
- Every query should contain the company name or ticker
- Use SPECIFIC names, products, and terms you learned from the profile and snippets
- Do NOT use generic queries — those were already done in Phase 1
- Target information gaps — what's missing from the snippets above?
- Include 3-4 queries specifically about BEAR CASE / risks / what could go wrong

Output ONLY the queries, one per line, no numbering or formatting.
"""

    try:
        response = get_llm(MODEL_FOLLOWUP).invoke([HumanMessage(content=prompt)])
        lines = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
        # Filter out empty lines and lines that look like headers
        queries = [q for q in lines if len(q) > 15 and not q.startswith("#") and not q.startswith("*")]
        print(f"LLM generated {len(queries)} follow-up queries")
        return queries[:25]  # Cap at 25
    except Exception as e:
        print(f"Follow-up query generation failed: {e}")
        return []

# --- Nodes ---

def planner(state: AgentState):
    """Decomposes the request. Initializes empty sections."""
    recent_messages = state["messages"][-10:]
    system_prompt = SystemMessage(content="""
    You are a financial research planner. 
    Analyze the user's request and create a detailed execution plan.
    Identify the main ticker symbol.
    Leave the peers list EMPTY — peers will be discovered automatically using data-driven analysis.
    """)
    structured_llm = get_llm(MODEL_PLAN).with_structured_output(AnalysisPlan)
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


def fetch_data(state: AgentState):
    """Fetches historical market data (Technical)."""
    if state.get("error"): return state
    ticker = state["ticker"]
    period = state.get("period", "1y")
    try:
        file_path = fetch_historical_data(ticker, period)
        return {"market_data_path": file_path, "error": None}
    except Exception as e:
        return {"error": f"Failed to fetch market data: {str(e)}"}


def generate_chart(state: AgentState):
    """Generates a chart."""
    if state.get("error"): return state
    data_path = state["market_data_path"]
    ticker = state["ticker"]
    try:
        chart_path = plot_price_history(data_path, ticker)
        return {"chart_path": chart_path, "error": None}
    except Exception as e:
        return {"error": f"Failed to generate chart: {str(e)}"}


def fetch_company_profile_node(state: AgentState):
    """Fetches company identity, ownership structure, and institutional holders."""
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


def fetch_financials(state: AgentState):
    """Fetches fundamental data (annual + quarterly + TTM)."""
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


def analyze_peers(state: AgentState):
    """Discovers peers using data-driven approach, then fetches comparison metrics."""
    if state.get("error"): return state

    ticker = state["ticker"]
    metadata_store = state.get("source_metadata", {})
    profile_meta = metadata_store.get("company_profile", {})

    sector = profile_meta.get("sector")
    industry = profile_meta.get("industry")
    market_cap = profile_meta.get("market_cap")

    try:
        discovered_peers = discover_peers(
            ticker=ticker,
            sector=sector,
            industry=industry,
            market_cap=market_cap,
            max_peers=5,
        )
        print(f"Discovered peers for {ticker}: {discovered_peers}")

        if not discovered_peers:
            print("Warning: No peers discovered, skipping peer analysis.")
            return {"peer_data_path": None, "error": None}

        all_tickers = [ticker] + discovered_peers
        file_path, metadata = fetch_peer_metrics(all_tickers, target_ticker=ticker)

        new_metadata = metadata_store.copy()
        new_metadata["peers"] = metadata
        return {"peer_data_path": file_path, "source_metadata": new_metadata, "error": None}

    except Exception as e:
        print(f"Warning: Peer analysis failed: {e}")
        return {"peer_data_path": None, "error": None}


def research_qualitative(state: AgentState):
    """
    Two-phase deep research node.

    Phase 1: Run 40+ generic queries (text + news), collect snippets + full-text reads.
    Phase 2: Feed snippets + profile to LLM → generate 15-25 company-specific follow-up
             queries → deep-search those for domain-specific depth.
    """
    if state.get("error"): return state
    ticker = state["ticker"]

    # Read company profile for smarter queries
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

    # ============================
    # PHASE 1: Generic queries (Parallelized)
    # ============================
    query_categories = build_research_queries(ticker, company_name, sector_hint, industry_hint)

    all_text_queries = []
    all_news_queries = []

    for category, queries in query_categories.items():
        all_text_queries.extend(queries)
        # Key categories also searched via news for recency
        if category in ("financial_performance", "risks_headwinds", "catalysts_events",
                         "analyst_sentiment", "ownership_governance"):
            all_news_queries.extend(queries[:3])

    print(f"Phase 1: {len(all_text_queries)} text queries + {len(all_news_queries)} news queries")

    # Functions for parallel execution
    def run_text_search():
        return deep_search(
            queries=all_text_queries,
            urls_per_query=3,
            max_workers=5,   # Reduced from 15 to 5 to share bandwidth
            max_chars_per_url=8000,
            search_delay=0.5, # Slightly increased delay for safety
            use_news=False,
        )

    def run_news_search():
        return deep_search(
            queries=all_news_queries,
            urls_per_query=3,
            max_workers=5,   # Reduced from 15 to 5
            max_chars_per_url=6000,
            search_delay=0.5,
            use_news=True,
        )
    
    def run_sec_search():
        try:
            results = search_and_read_filings(
                company_name=company_name or ticker,
                ticker=ticker,
                form_types=["10-K", "10-Q", "8-K"],
                max_filings=5,
                max_chars_per_filing=12000,
            )
            print(f"SEC: Found {len(results)} filings")
            return results
        except Exception as e:
            print(f"SEC filing search failed: {e}")
            return []

    # Execute Phase 1 + SEC in parallel
    text_results = []
    news_results = []
    sec_results = []

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_text = executor.submit(run_text_search)
        future_news = executor.submit(run_news_search)
        future_sec = executor.submit(run_sec_search)

        text_results = future_text.result()
        news_results = future_news.result()
        sec_results = future_sec.result()

    # Collect Phase 1 snippets for the LLM to read
    phase1_snippets = "\n".join([
        f"- [{r['title']}]({r['url']}): {r.get('snippet', '') or (r.get('full_text', '') or '')[:300]}"
        for r in text_results + news_results
        if r.get("title")
    ])

    # ============================
    # PHASE 2: LLM follow-up queries (Parallelized)
    # ============================
    followup_queries = generate_followup_queries(
        ticker=ticker,
        company_name=company_name,
        sector=sector_hint,
        industry=industry_hint,
        profile_text=profile_text,
        initial_snippets=phase1_snippets,
    )

    followup_results = []
    if followup_queries:
        print(f"Phase 2: Running {len(followup_queries)} LLM-generated follow-up queries")
        
        def run_followup_text():
            return deep_search(
                queries=followup_queries,
                urls_per_query=3,
                max_workers=5,
                max_chars_per_url=8000,
                search_delay=0.5,
                use_news=False,
            )

        def run_followup_news():
            # Also run a subset through news for recency
            return deep_search(
                queries=followup_queries[:8],
                urls_per_query=2,
                max_workers=5,
                max_chars_per_url=6000,
                search_delay=0.5,
                use_news=True,
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_f_text = executor.submit(run_followup_text)
            future_f_news = executor.submit(run_followup_news)
            
            f_text_results = future_f_text.result()
            f_news_results = future_f_news.result()
            
        followup_results = f_text_results + f_news_results

    # ============================
    # Consolidate all research
    # ============================
    research_data = {}
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

    # Add SEC filing content
    for f in sec_results:
        key = f"SEC Filing: {f['form_type']} ({f['filing_date']})"
        research_data[key] = f"### SEC {f['form_type']} — {f['filing_date']}\nURL: {f['url']}\n\n{f['text']}\n"

    total_chars = sum(len(v) for v in research_data.values())
    print(f"Research complete: {articles_read} articles ({articles_with_full_text} full-text), "
          f"{len(sec_results)} SEC filings, {len(followup_queries)} follow-up queries")
    print(f"Total raw research text: {total_chars:,} characters (~{total_chars // 4:,} tokens)")

    return {"research_data": research_data, "error": None}


def synthesize_research(state: AgentState):
    """
    Map-Reduce LLM synthesis: converts massive raw research into focused section briefs.

    Map step: For each batch of raw text, extract key facts relevant to the company.
    Reduce step: For each report section, merge extracted facts into a focused brief.
    """
    if state.get("error"): return state

    ticker = state["ticker"]
    research_data = state.get("research_data", {})

    if not research_data:
        print("Warning: No research data to synthesize")
        return {"research_brief": {}, "error": None}

    # --- MAP STEP ---
    # Chunk the raw research into batches that fit in context
    # Each batch gets summarized by the LLM to extract key facts
    all_text_items = list(research_data.values())
    
    # Combine into manageable chunks (~50KB each, ~10K tokens)
    CHUNK_SIZE = 50000
    chunks = []
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

    # Extract key facts from each chunk
    extracted_facts: List[str] = []

    for i, chunk in enumerate(chunks):
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
            extracted_facts.append(response.content)
            print(f"  MAP chunk {i+1}/{len(chunks)}: extracted {len(response.content)} chars of facts")
        except Exception as e:
            print(f"  MAP chunk {i+1} failed: {e}")

    all_facts = "\n\n".join(extracted_facts)
    print(f"Synthesis MAP complete: {len(all_facts):,} chars of extracted facts")

    # --- REDUCE STEP ---
    # For each report section, ask the LLM to select and organize the relevant facts
    research_brief = {}

    for section_key in SECTION_ORDER:
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
            research_brief[section_key] = response.content
            print(f"  REDUCE {section_key}: {len(response.content)} chars")
        except Exception as e:
            print(f"  REDUCE {section_key} failed: {e}")
            research_brief[section_key] = ""

    print(f"Synthesis REDUCE complete: {sum(len(v) for v in research_brief.values()):,} chars total brief")

    return {"research_brief": research_brief, "error": None}


def generate_section(state: AgentState):
    """Generates the next missing section using synthesized research brief."""
    if state.get("error"): return state

    current_sections = state.get("sections", {})

    # Identify next section to generate
    next_section_key = None
    for key in SECTION_ORDER:
        if key not in current_sections:
            next_section_key = key
            break

    if not next_section_key:
        return {"error": "Loop Error: No next section found but loop continued."}

    # Gather Data Context
    ticker = state["ticker"]
    fin_path = state.get("financials_path")
    peer_path = state.get("peer_data_path")
    profile_path = state.get("company_profile_path")

    # Read structured data files
    financials_content = _read_file(fin_path, max_chars=12000)
    peer_content = _read_file(peer_path, max_chars=6000)
    profile_content = _read_file(profile_path, max_chars=5000)

    # Use the synthesized research brief (not raw research)
    research_brief = state.get("research_brief", {})
    section_brief = research_brief.get(next_section_key, "No research brief available.")

    # Prompt Construction
    # Static content (Profile, Financials, Peer Data) is placed FIRST so it can be cached across the loop.
    # Variable content (Section Instructions, Research Brief) is placed LAST.
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
    Generate the **{next_section_key.replace('_', ' ').upper()}** section for {ticker}.
    
    ### Deep Research Brief (Specific to this section)
    {section_brief}
    
    ### Section Guidance
    {SECTION_PROMPTS[next_section_key]}
    
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

    structured_llm = get_llm(MODEL_SECTION).with_structured_output(ResearchSection)

    try:
        section = structured_llm.invoke([HumanMessage(content=prompt)])
        new_sections = current_sections.copy()
        new_sections[next_section_key] = section
        print(f"Section '{next_section_key}' generated ({len(section.content)} chars)")
        return {"sections": new_sections, "error": None}
    except Exception as e:
        return {"error": f"Failed to generate section {next_section_key}: {str(e)}"}


def generate_summary(state: AgentState):
    """Generates the Investment Summary based on all completed sections."""
    if state.get("error"): return state

    ticker = state["ticker"]
    sections = state.get("sections", {})

    full_text = "\n\n".join([f"## {s.title}\n{s.content}" for s in sections.values()])

    profile_path = state.get("company_profile_path")
    profile_content = _read_file(profile_path, max_chars=3000)

    # Also include the full research brief for synthesis
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
            references=summary.references
        )
        
        return {"report_content": final_report, "error": None}
        
    except Exception as e:
        return {"error": f"Failed to generate summary: {str(e)}"}


def verify_content(state: AgentState):
    """Verifies the generated report."""
    if state.get("error"): return state
    report = state.get("report_content")
    if not report: return {"error": "No report content found during verification"}
    return state


def save_report(state: AgentState):
    """Saves the report to Markdown."""
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
        report.esg_analysis,
        report.investment_risks
    ]
    
    for s in sections:
        md += f"## {s.title}\n{s.content}\n\n"
        
    md += "## References\n"
    for r in report.references:
        md += f"- {r}\n"
        
    report_dir = os.path.join(os.getcwd(), "reports")
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, f"{ticker}_initiation_report.md")
    
    with open(report_path, "w") as f:
        f.write(md)
        
    return {"report_path": report_path}


# --- Helper ---

def _read_file(
    path: str | None,
    max_chars: int = 5000,
    condense_purpose: str = "equity research section generation",
) -> str:
    """
    Safely reads a file. If content exceeds max_chars, uses LLM condensation
    instead of hard truncation to preserve critical information.
    """
    if not path or not os.path.exists(path):
        return "N/A"
    try:
        with open(path, "r") as f:
            content = f.read()
        if len(content) > max_chars:
            return condense_context(
                content, max_chars=max_chars,
                purpose=condense_purpose,
                focus_areas="financial figures, ratios, company metrics, key data points",
            )
        return content
    except Exception:
        return "N/A"


# --- Edges ---

def check_loop(state: AgentState) -> Literal["generate_section", "generate_summary"]:
    """Decides whether to continue the loop or finish."""
    current = state.get("sections", {})
    if len(current) < len(SECTION_ORDER):
        return "generate_section"
    return "generate_summary"

def check_error(state: AgentState) -> Literal["continue", "end"]:
    if state.get("error"): return "end"
    return "continue"


# --- Graph ---

workflow = StateGraph(AgentState)

workflow.add_node("planner", planner)
workflow.add_node("fetch_data", fetch_data)
workflow.add_node("generate_chart", generate_chart)
workflow.add_node("fetch_company_profile", fetch_company_profile_node)
workflow.add_node("fetch_financials", fetch_financials)
workflow.add_node("analyze_peers", analyze_peers)
workflow.add_node("research_qualitative", research_qualitative)
workflow.add_node("synthesize_research", synthesize_research)
workflow.add_node("generate_section", generate_section)
workflow.add_node("generate_summary", generate_summary)
workflow.add_node("verify_content", verify_content)
workflow.add_node("save_report", save_report)

# Flow
workflow.add_edge(START, "planner")
workflow.add_conditional_edges("planner", check_error, {"continue": "fetch_data", "end": END})
workflow.add_conditional_edges("fetch_data", check_error, {"continue": "generate_chart", "end": END})
workflow.add_conditional_edges("generate_chart", check_error, {"continue": "fetch_company_profile", "end": END})

# Company profile → financials → peers → deep research → synthesis → section loop
workflow.add_edge("fetch_company_profile", "fetch_financials")
workflow.add_edge("fetch_financials", "analyze_peers")
workflow.add_edge("analyze_peers", "research_qualitative")
workflow.add_edge("research_qualitative", "synthesize_research")
workflow.add_edge("synthesize_research", "generate_section")

# The Section Loop
workflow.add_conditional_edges(
    "generate_section",
    check_loop,
    {
        "generate_section": "generate_section",
        "generate_summary": "generate_summary"
    }
)

workflow.add_edge("generate_summary", "verify_content")
workflow.add_edge("verify_content", "save_report")
workflow.add_edge("save_report", END)

app = workflow.compile()
