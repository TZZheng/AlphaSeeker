"""
Earnings Call Transcripts Tool (Alternative free approach).

Since FMP specific transcripts are paid, this uses the existing `deep_search`
utility to locate recent earnings call transcripts or summaries on the web,
and uses an LLM to extract forward guidance, Q&A sentiment, and operations updates.
"""

from typing import Tuple, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage
from src.shared.web_search import deep_search
from src.shared.llm_manager import get_llm
from src.shared.model_config import get_model

def research_earnings_call(ticker: str, company_name: str) -> Tuple[str, Dict[str, Any]]:
    """
    Searches the web for the latest earnings call transcript or summary,
    and extracts key insights using an LLM.

    Args:
        ticker: Stock ticker.
        company_name: Company name.

    Returns:
        Tuple of (markdown summary string, metadata dict).
    """
    name = company_name or ticker
    
    # Generate targeted queries to specifically find transcripts/summaries
    queries = [
        f"{ticker} {name} latest earnings call transcript full text",
        f"{ticker} {name} earnings conference call seeking alpha motley fool transcript",
        f"{ticker} {name} management guidance earnings call Q&A highlights"
    ]

    print(f"--- [Earnings Call] Searching web for {ticker} transcripts ---")
    
    # Run deep search
    try:
        results = deep_search(
            queries=queries,
            urls_per_query=2,
            max_workers=3,
            max_chars_per_url=15000,
            use_news=False
        )
    except Exception as e:
        print(f"Search for earnings call failed: {e}")
        return f"Failed to search for earnings calls: {e}", {"error": str(e)}

    if not results:
        return f"No earnings call data found on the web for {ticker}.", {"status": "No results"}

    # Concatenate findings
    raw_text = ""
    for r in results:
        raw_text += f"\n\n### Source: {r['title']} ({r['url']})\n"
        raw_text += r.get("full_text", "") or r.get("snippet", "")

    if len(raw_text.strip()) < 200:
        return f"Insufficient text retrieved for {ticker} earnings calls.", {"status": "Insufficient text"}

    print(f"--- [Earnings Call] Analyzing {len(raw_text)} chars of text ---")

    # Use LLM to condense
    model_name = get_model("equity", "condense")
    llm = get_llm(model_name)

    sys_prompt = f"""You are an expert equity research analyst reviewing the latest earnings call transcript and summaries for {name} ({ticker}).
    
Based on the provided search results, extract the following key information:
1. **Management Tone / Sentiment**: Are they confident, cautious, or defensive?
2. **Forward Guidance**: Any specific numbers, growth rates, or margin targets provided for the next quarter/year?
3. **Q&A Highlights**: What were analysts most concerned about? What were management's answers?
4. **Key Operational Updates**: New products, delays, CapEx plans, cost-cutting, etc.

If the provided text does not contain actual transcripts or earnings summaries, state that the search results were not highly relevant.
Keep it strictly factual and concise. Format as a Markdown brief.
"""

    # Truncate text to avoid context limits
    safe_text = raw_text[:40000]

    try:
        messages = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=f"SEARCH RESULTS:\n{safe_text}")
        ]
        response = llm.invoke(messages)
        
        metadata = {
            "source": "Web Search (Earnings Calls)",
            "articles_analyzed": len(results)
        }
        return response.content, metadata
    except Exception as e:
        print(f"Earnings call LLM extraction failed: {e}")
        return f"Failed to extract earnings call insights: {e}", {"error": str(e)}
