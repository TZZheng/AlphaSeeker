"""Equity-specific harness skill adapters."""

from __future__ import annotations

from typing import Any

from src.agents.equity.tools.company_profile import fetch_company_profile
from src.agents.equity.tools.earnings_calls import research_earnings_call
from src.agents.equity.tools.financials import fetch_financial_metrics
from src.agents.equity.tools.insider_trading import fetch_insider_activity
from src.agents.equity.tools.market_data import fetch_historical_data
from src.agents.equity.tools.peers import (
    evaluate_candidates,
    extract_peers_from_text,
    fetch_peer_metrics,
)
from src.agents.equity.tools.sec_filings import search_and_read_filings
from src.agents.equity.tools.visualization import plot_price_history
from src.harness.skills.common import artifact_evidence, json_preview, make_result, note_evidence, safe_read, url_evidence
from src.harness.types import HarnessState, SkillMetrics, SkillResult, SkillSpec


def _recent_evidence_text(state: HarnessState, limit: int = 8) -> str:
    return "\n\n".join(item.content or "" for item in state.evidence_ledger[-limit:] if item.content)


def fetch_market_data_skill(arguments: dict[str, Any], _state: HarnessState) -> SkillResult:
    ticker = str(arguments.get("ticker") or "").strip().upper()
    period = str(arguments.get("period") or "1y")
    if not ticker:
        return make_result(
            "fetch_market_data",
            arguments,
            status="failed",
            summary="fetch_market_data requires a ticker.",
            error="Missing ticker.",
        )

    path = fetch_historical_data(ticker, period)
    return make_result(
        "fetch_market_data",
        arguments,
        status="ok",
        summary=f"Fetched {period} historical market data for {ticker}.",
        details={"ticker": ticker, "period": period, "path": path},
        metrics=SkillMetrics(
            evidence_count=1,
            artifact_count=1,
            sections_touched=["Valuation and Scenarios"],
        ),
        output_text=path,
        artifacts=[path],
        evidence=[artifact_evidence("fetch_market_data", f"Historical market data for {ticker}.", path)],
    )


def plot_price_history_skill(arguments: dict[str, Any], _state: HarnessState) -> SkillResult:
    data_path = str(arguments.get("data_path") or "").strip()
    ticker = str(arguments.get("ticker") or "").strip().upper()
    if not data_path or not ticker:
        return make_result(
            "plot_price_history",
            arguments,
            status="failed",
            summary="plot_price_history requires both data_path and ticker.",
            error="Missing data_path or ticker.",
        )

    chart_path = plot_price_history(data_path, ticker)
    return make_result(
        "plot_price_history",
        arguments,
        status="ok",
        summary=f"Generated a price chart for {ticker}.",
        details={"ticker": ticker, "chart_path": chart_path, "data_path": data_path},
        metrics=SkillMetrics(evidence_count=1, artifact_count=1, sections_touched=["Valuation and Scenarios"]),
        output_text=chart_path,
        artifacts=[chart_path],
        evidence=[artifact_evidence("plot_price_history", f"Price chart for {ticker}.", chart_path)],
    )


def fetch_company_profile_skill(arguments: dict[str, Any], _state: HarnessState) -> SkillResult:
    ticker = str(arguments.get("ticker") or "").strip().upper()
    if not ticker:
        return make_result(
            "fetch_company_profile",
            arguments,
            status="failed",
            summary="fetch_company_profile requires a ticker.",
            error="Missing ticker.",
        )

    path, metadata = fetch_company_profile(ticker)
    text = safe_read(path, max_chars=5000)
    return make_result(
        "fetch_company_profile",
        arguments,
        status="ok",
        summary=f"Fetched company profile data for {ticker}.",
        details={"ticker": ticker, "metadata": metadata, "path": path},
        metrics=SkillMetrics(
            evidence_count=1,
            artifact_count=1,
            sections_touched=["Equity Overview"],
        ),
        output_text=text,
        artifacts=[path],
        evidence=[artifact_evidence("fetch_company_profile", f"Company profile for {ticker}.", path, content=text, metadata=metadata)],
    )


def fetch_financials_skill(arguments: dict[str, Any], _state: HarnessState) -> SkillResult:
    ticker = str(arguments.get("ticker") or "").strip().upper()
    if not ticker:
        return make_result(
            "fetch_financials",
            arguments,
            status="failed",
            summary="fetch_financials requires a ticker.",
            error="Missing ticker.",
        )

    path, metadata = fetch_financial_metrics(ticker)
    text = safe_read(path, max_chars=5000)
    return make_result(
        "fetch_financials",
        arguments,
        status="ok",
        summary=f"Fetched financial metrics for {ticker}.",
        details={"ticker": ticker, "metadata": metadata, "path": path},
        metrics=SkillMetrics(
            evidence_count=1,
            artifact_count=1,
            sections_touched=["Key Findings", "Valuation and Scenarios"],
        ),
        output_text=text,
        artifacts=[path],
        evidence=[artifact_evidence("fetch_financials", f"Financial metrics for {ticker}.", path, content=text, metadata=metadata)],
    )


def search_sec_filings_skill(arguments: dict[str, Any], _state: HarnessState) -> SkillResult:
    company_name = str(arguments.get("company_name") or arguments.get("ticker") or "").strip()
    ticker = str(arguments.get("ticker") or "").strip().upper()
    form_types = arguments.get("form_types")
    max_filings = int(arguments.get("max_filings", 3))
    max_chars_per_filing = int(arguments.get("max_chars_per_filing", 12000))
    if not company_name:
        return make_result(
            "search_sec_filings",
            arguments,
            status="failed",
            summary="search_sec_filings requires company_name or ticker.",
            error="Missing company_name.",
        )

    resolved_company_name = company_name
    if ticker and (resolved_company_name.upper() == ticker or len(resolved_company_name) <= 5):
        profile_metadata: dict[str, Any] = {}
        for item in reversed(_state.evidence_ledger):
            if item.skill_name != "fetch_company_profile":
                continue
            item_ticker = str(item.metadata.get("symbol") or item.metadata.get("ticker") or "").upper()
            if item_ticker and item_ticker != ticker:
                continue
            if isinstance(item.metadata, dict):
                profile_metadata = item.metadata
                break
        if not profile_metadata:
            try:
                _, fetched_metadata = fetch_company_profile(ticker)
            except Exception:
                fetched_metadata = {}
            if isinstance(fetched_metadata, dict):
                profile_metadata = fetched_metadata
        candidate = (
            profile_metadata.get("company_name")
            or profile_metadata.get("longName")
            or profile_metadata.get("shortName")
            or profile_metadata.get("displayName")
            or profile_metadata.get("name")
        )
        if isinstance(candidate, str) and candidate.strip():
            resolved_company_name = candidate.strip()

    results = search_and_read_filings(
        company_name=resolved_company_name,
        ticker=ticker,
        form_types=form_types,
        max_filings=max_filings,
        max_chars_per_filing=max_chars_per_filing,
    )
    evidence = [
        url_evidence(
            "search_sec_filings",
            f"{item.get('form_type', 'SEC filing')} filed {item.get('filing_date', '')}",
            item.get("url", ""),
            content=item.get("text", ""),
            metadata={"company_name": resolved_company_name, "ticker": ticker},
        )
        for item in results
    ]
    output_text = "\n\n".join(
        f"### {item.get('form_type', '')} {item.get('filing_date', '')}\n{item.get('url', '')}\n\n{item.get('text', '')}"
        for item in results
    )
    display_company_name = resolved_company_name.rstrip(".")

    return make_result(
        "search_sec_filings",
        arguments,
        status="ok",
        summary=f"Read {len(results)} SEC filing(s) for {display_company_name}.",
        details={"company_name": resolved_company_name, "ticker": ticker, "results": results},
        metrics=SkillMetrics(
            evidence_count=len(evidence),
            urls_read=len(results),
            dated_evidence_count=sum(1 for item in results if item.get("filing_date")),
            filings_found=len(results),
            sections_touched=["Equity Overview", "Valuation and Scenarios"],
        ),
        output_text=output_text,
        evidence=evidence,
    )


def fetch_insider_activity_skill(arguments: dict[str, Any], _state: HarnessState) -> SkillResult:
    ticker = str(arguments.get("ticker") or "").strip().upper()
    if not ticker:
        return make_result(
            "fetch_insider_activity",
            arguments,
            status="failed",
            summary="fetch_insider_activity requires a ticker.",
            error="Missing ticker.",
        )

    path, metadata = fetch_insider_activity(ticker)
    if not path:
        return make_result(
            "fetch_insider_activity",
            arguments,
            status="partial",
            summary=f"No insider-activity artifact was produced for {ticker}.",
            details={"ticker": ticker, "metadata": metadata},
            error="Insider activity source unavailable or empty.",
        )
    text = safe_read(path, max_chars=4000)
    return make_result(
        "fetch_insider_activity",
        arguments,
        status="ok",
        summary=f"Fetched insider trading activity for {ticker}.",
        details={"ticker": ticker, "metadata": metadata, "path": path},
        metrics=SkillMetrics(
            evidence_count=1,
            artifact_count=1,
            sections_touched=["Risks and Counterevidence"],
        ),
        output_text=text,
        artifacts=[path],
        evidence=[artifact_evidence("fetch_insider_activity", f"Insider trading activity for {ticker}.", path, content=text, metadata=metadata)],
    )


def research_earnings_call_skill(arguments: dict[str, Any], _state: HarnessState) -> SkillResult:
    ticker = str(arguments.get("ticker") or "").strip().upper()
    company_name = str(arguments.get("company_name") or ticker).strip()
    if not ticker:
        return make_result(
            "research_earnings_call",
            arguments,
            status="failed",
            summary="research_earnings_call requires a ticker.",
            error="Missing ticker.",
        )

    text, metadata = research_earnings_call(ticker, company_name)
    return make_result(
        "research_earnings_call",
        arguments,
        status="ok" if not metadata.get("error") else "partial",
        summary=f"Collected earnings-call research for {ticker}.",
        details={"ticker": ticker, "company_name": company_name, "metadata": metadata},
        metrics=SkillMetrics(
            evidence_count=1,
            sections_touched=["Equity Overview", "Key Findings", "Risks and Counterevidence"],
        ),
        output_text=text,
        evidence=[note_evidence("research_earnings_call", f"Earnings call notes for {ticker}.", content=text, metadata=metadata)],
        error=metadata.get("error"),
    )


def analyze_peers_skill(arguments: dict[str, Any], state: HarnessState) -> SkillResult:
    ticker = str(arguments.get("ticker") or "").strip().upper()
    if not ticker:
        return make_result(
            "analyze_peers",
            arguments,
            status="failed",
            summary="analyze_peers requires a ticker.",
            error="Missing ticker.",
        )

    seed_text = str(arguments.get("seed_text") or "").strip()
    if not seed_text:
        seed_text = _recent_evidence_text(state)
    if not seed_text:
        return make_result(
            "analyze_peers",
            arguments,
            status="failed",
            summary="analyze_peers needs seed_text or prior evidence to extract competitors.",
            error="Missing seed_text.",
        )

    candidates = extract_peers_from_text(seed_text)
    categorized = evaluate_candidates(candidates, ticker)
    path, metadata = fetch_peer_metrics(categorized, target_ticker=ticker)
    text = safe_read(path, max_chars=5000)
    return make_result(
        "analyze_peers",
        arguments,
        status="ok",
        summary=f"Generated peer analysis for {ticker} with {sum(len(v) for v in categorized.values())} categorized peers.",
        details={"ticker": ticker, "candidates": candidates, "categorized": categorized, "metadata": metadata},
        metrics=SkillMetrics(
            evidence_count=1,
            artifact_count=1,
            sections_touched=["Peer and Competitive Pressure", "Risks and Counterevidence"],
            extra={"peer_count": sum(len(v) for v in categorized.values())},
        ),
        output_text=text or json_preview(categorized),
        artifacts=[path],
        evidence=[artifact_evidence("analyze_peers", f"Peer analysis for {ticker}.", path, content=text, metadata={"categorized": categorized, "metadata": metadata})],
    )


EQUITY_SKILLS = [
    SkillSpec(
        name="fetch_market_data",
        description="Fetch historical OHLCV market data for an equity ticker.",
        pack="equity",
        input_schema={"ticker": "string", "period": "string"},
        produces_artifacts=True,
        executor=fetch_market_data_skill,
    ),
    SkillSpec(
        name="plot_price_history",
        description="Generate a price chart from a saved market-data file.",
        pack="equity",
        input_schema={"data_path": "string", "ticker": "string"},
        produces_artifacts=True,
        executor=plot_price_history_skill,
    ),
    SkillSpec(
        name="fetch_company_profile",
        description="Fetch company profile, sector, industry, and ownership data.",
        pack="equity",
        input_schema={"ticker": "string"},
        produces_artifacts=True,
        executor=fetch_company_profile_skill,
    ),
    SkillSpec(
        name="fetch_financials",
        description="Fetch income statement, balance sheet, cash flow, and key ratios.",
        pack="equity",
        input_schema={"ticker": "string"},
        produces_artifacts=True,
        executor=fetch_financials_skill,
    ),
    SkillSpec(
        name="search_sec_filings",
        description="Search and read recent SEC filings for an equity issuer.",
        pack="equity",
        input_schema={
            "company_name": "string",
            "ticker": "string",
            "form_types": "string[]",
            "max_filings": "integer",
        },
        executor=search_sec_filings_skill,
    ),
    SkillSpec(
        name="fetch_insider_activity",
        description="Fetch recent insider trading activity from Form 4-style data sources.",
        pack="equity",
        input_schema={"ticker": "string"},
        produces_artifacts=True,
        executor=fetch_insider_activity_skill,
    ),
    SkillSpec(
        name="research_earnings_call",
        description="Collect and summarize recent earnings-call evidence.",
        pack="equity",
        input_schema={"ticker": "string", "company_name": "string"},
        executor=research_earnings_call_skill,
    ),
    SkillSpec(
        name="analyze_peers",
        description="Extract peer candidates, categorize them, and build a peer-comparison artifact.",
        pack="equity",
        input_schema={"ticker": "string", "seed_text": "string"},
        produces_artifacts=True,
        executor=analyze_peers_skill,
    ),
]
