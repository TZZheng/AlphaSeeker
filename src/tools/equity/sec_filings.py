"""
SEC EDGAR filing search and text extraction tool.

Uses the free EFTS (EDGAR Full-Text Search) API — no API key required.
Fetches recent 10-K, 10-Q, 8-K filings and extracts text via trafilatura.

IMPORTANT: SEC requires a User-Agent header with company name and email.
"""

import datetime
from html.parser import HTMLParser
import re
from urllib.parse import urljoin

import requests
import time
from typing import List, Dict, Optional

import trafilatura
from langchain_core.messages import SystemMessage, HumanMessage
from src.shared.llm_manager import get_llm
from src.shared.model_config import get_model
from src.shared.reliability import cached_retry_call, request_json, request_text

# SEC EDGAR EFTS base URL
EFTS_BASE = "https://efts.sec.gov/LATEST/search-index"

# SEC-compliant headers
SEC_HEADERS = {
    "User-Agent": "AlphaSeeker Research research@alphaseeker.dev",
    "Accept-Encoding": "gzip, deflate",
}



class _FilingDocumentParser(HTMLParser):
    """Extract the primary filing document link from an SEC filing detail page."""

    def __init__(self) -> None:
        super().__init__()
        self.in_table = False
        self.current_row: Dict[str, str] | None = None
        self.current_cell = ""
        self.current_cell_index = -1
        self.current_link = ""
        self.rows: List[Dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {key: value or "" for key, value in attrs}
        if tag == "table" and attr.get("summary") == "Document Format Files":
            self.in_table = True
        elif self.in_table and tag == "tr":
            self.current_row = {"seq": "", "description": "", "document": "", "type": "", "size": "", "href": ""}
            self.current_cell = ""
            self.current_cell_index = -1
        elif self.in_table and tag in {"td", "th"}:
            # SEC filing table columns: Seq, Description, Document, Type, Size.
            if self.current_row is None:
                return
            order = ["seq", "description", "document", "type", "size"]
            self.current_cell_index += 1
            self.current_cell = order[min(self.current_cell_index, len(order) - 1)]
        elif self.in_table and tag == "a" and self.current_row is not None:
            self.current_link = attr.get("href", "")

    def handle_endtag(self, tag: str) -> None:
        if tag == "table" and self.in_table:
            self.in_table = False
        elif self.in_table and tag == "tr" and self.current_row is not None:
            if self.current_row.get("href") and self.current_row.get("type"):
                self.rows.append(self.current_row)
            self.current_row = None
            self.current_cell = ""
            self.current_cell_index = -1
        elif self.in_table and tag == "a":
            self.current_link = ""

    def handle_data(self, data: str) -> None:
        if not self.in_table or self.current_row is None or not self.current_cell:
            return
        text = data.strip()
        if not text:
            return
        if self.current_cell in self.current_row:
            existing = self.current_row.get(self.current_cell, "")
            self.current_row[self.current_cell] = (existing + " " + text).strip()
        if self.current_link:
            self.current_row["href"] = self.current_link


def resolve_sec_primary_document_url(index_url: str) -> str:
    """Resolve an SEC filing detail/index page to its primary filing document URL."""

    if "-index.htm" not in index_url and "-index.html" not in index_url:
        return index_url
    html = request_text(
        index_url,
        headers=SEC_HEADERS,
        timeout=15,
        ttl_seconds=21600,
        attempts=2,
    )
    if not html:
        return index_url
    parser = _FilingDocumentParser()
    parser.feed(html)
    for row in parser.rows:
        filing_type = row.get("type", "").upper()
        href = row.get("href", "")
        if not href:
            continue
        if filing_type in {"10-K", "10-Q", "8-K", "DEF 14A"} or not filing_type.startswith("EX-"):
            # iXBRL links use /ix?doc=/Archives/...; trafilatura reads the raw filing better.
            if href.startswith("/ix?doc="):
                href = href.split("/ix?doc=", 1)[1]
            return urljoin(index_url, href)
    return index_url

def search_sec_filings(
    company_name: str,
    ticker: str = "",
    form_types: Optional[List[str]] = None,
    date_range: str = "",
    max_results: int = 10,
) -> List[Dict[str, str]]:
    """
    Search SEC EDGAR for recent filings.

    Args:
        company_name: Company name to search for.
        ticker: Optional ticker symbol for more precise search.
        form_types: List of form types to filter (e.g., ["10-K", "10-Q", "8-K"]).
        date_range: Date range filter (e.g., "custom" with dateRange param).
        max_results: Maximum number of filings to return.

    Returns:
        List of dicts with 'form_type', 'filing_date', 'company', 'url', 'description'.
    """
    if form_types is None:
        form_types = ["10-K", "10-Q", "8-K"]

    query = ticker or company_name

    # Dynamic date window (default: last 3 years) to avoid stale hardcoded bounds.
    today = datetime.date.today()
    default_start = today - datetime.timedelta(days=365 * 3)
    if date_range and ":" in date_range:
        startdt, enddt = date_range.split(":", 1)
    else:
        startdt, enddt = default_start.isoformat(), today.isoformat()

    filings: List[Dict[str, str]] = []
    seen_urls = set()

    try:
        # 1) SEC company Atom feed (stable fallback path)
        company_search_url = "https://www.sec.gov/cgi-bin/browse-edgar"
        company_params = {
            "action": "getcompany",
            "company": company_name,
            "CIK": ticker,
            "type": form_types[0] if form_types else "",
            "dateb": "",
            "owner": "include",
            "count": str(max_results),
            "search_text": "",
            "output": "atom",
        }

        atom_text = request_text(
            company_search_url,
            params=company_params,
            headers=SEC_HEADERS,
            timeout=15,
            ttl_seconds=1800,
            attempts=3,
        )

        if atom_text:
            # Parse the Atom feed for filing links
            from xml.etree import ElementTree as ET

            # SEC returns Atom XML
            try:
                root = ET.fromstring(atom_text)
                ns = {"atom": "http://www.w3.org/2005/Atom"}

                for entry in root.findall("atom:entry", ns)[:max_results]:
                    title_el = entry.find("atom:title", ns)
                    link_el = entry.find("atom:link", ns)
                    updated_el = entry.find("atom:updated", ns)
                    summary_el = entry.find("atom:summary", ns)
                    form_type = title_el.text if title_el is not None and title_el.text else ""
                    filing_date = updated_el.text[:10] if updated_el is not None and updated_el.text else ""
                    filing_url = link_el.attrib.get("href", "") if link_el is not None else ""
                    if not filing_url or filing_url in seen_urls:
                        continue
                    seen_urls.add(filing_url)

                    filings.append({
                        "form_type": form_type,
                        "filing_date": filing_date,
                        "company": company_name,
                        "url": filing_url,
                        "description": summary_el.text[:500] if summary_el is not None and summary_el.text else "",
                    })
            except ET.ParseError:
                print(f"SEC: Could not parse XML response for {company_name}")

        # 2) EFTS JSON endpoint (better for filtering + date windows)
        efts_query = f'"{company_name}" OR "{ticker}"' if ticker else f'"{company_name}"'
        efts_params = {
            "q": efts_query,
            "forms": ",".join(form_types),
            "dateRange": "custom",
            "startdt": startdt,
            "enddt": enddt,
        }

        try:
            data = request_json(
                EFTS_BASE,
                params=efts_params,
                headers=SEC_HEADERS,
                timeout=15,
                ttl_seconds=1800,
                attempts=3,
            )
            hits = data.get("hits", {}).get("hits", [])
            for hit in hits[:max_results]:
                source = hit.get("_source", {})
                file_url = source.get("file_url", "")
                if file_url and not file_url.startswith("http"):
                    file_url = f"https://www.sec.gov{file_url}"
                if not file_url or file_url in seen_urls:
                    continue
                seen_urls.add(file_url)

                filings.append({
                    "form_type": source.get("form_type", ""),
                    "filing_date": source.get("file_date", ""),
                    "company": source.get("entity_name", company_name),
                    "url": file_url,
                    "description": source.get("file_description", "")[:500],
                })
        except Exception as e:
            print(f"SEC: EFTS JSON query failed for {company_name} ({e})")

    except Exception as e:
        print(f"SEC filing search failed for {company_name}: {e}")

    return filings



def _extract_sec_text_from_html(html: str, max_chars: int) -> Optional[str]:
    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        favor_precision=False,
        deduplicate=True,
    )
    if not text:
        return None
    # Inline XBRL pages can begin with a very noisy taxonomy preamble. Keep the
    # filing narrative from the first recognizable annual/quarterly report marker.
    markers = [
        "UNITED STATES SECURITIES AND EXCHANGE COMMISSION",
        "FORM 10-K",
        "FORM 10-Q",
        "FORM 8-K",
    ]
    upper_text = text.upper()
    positions = [upper_text.find(marker) for marker in markers if upper_text.find(marker) >= 0]
    if positions:
        text = text[min(positions):]
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n... [truncated at {max_chars} chars]"
    return text

def fetch_filing_text(url: str, max_chars: int = 15000) -> Optional[str]:
    """
    Fetches and extracts readable text from a SEC filing URL.

    Args:
        url: URL to the filing document.
        max_chars: Maximum characters to return.

    Returns:
        Extracted text or None on failure.
    """
    if not url:
        return None

    def _load() -> Optional[str]:
        resolved_url = resolve_sec_primary_document_url(url)
        downloaded = trafilatura.fetch_url(resolved_url)
        if downloaded:
            text = _extract_sec_text_from_html(downloaded, max_chars)
            if text:
                return text
        html = request_text(
            resolved_url,
            headers=SEC_HEADERS,
            timeout=20,
            ttl_seconds=21600,
            attempts=2,
        )
        if not html:
            return None
        return _extract_sec_text_from_html(html, max_chars)
    try:
        return cached_retry_call(
            "sec_filing_text_v2",
            {"url": url, "max_chars": max_chars},
            _load,
            ttl_seconds=21600,
            attempts=2,
            min_wait_seconds=1,
            max_wait_seconds=5,
        )
    except Exception as e:
        print(f"SEC: failed to read filing URL {url} ({e})")
        return None


def search_and_read_filings(
    company_name: str,
    ticker: str = "",
    form_types: Optional[List[str]] = None,
    max_filings: int = 5,
    max_chars_per_filing: int = 15000,
) -> List[Dict[str, str]]:
    """
    Search for SEC filings and read their text content.

    Args:
        company_name: Company name.
        ticker: Ticker symbol.
        form_types: Filing types to search for.
        max_filings: Max number of filings to read.
        max_chars_per_filing: Max chars per filing.

    Returns:
        List of dicts with 'form_type', 'filing_date', 'url', 'text'.
    """
    filings = search_sec_filings(
        company_name=company_name,
        ticker=ticker,
        form_types=form_types,
        max_results=max_filings,
    )

    results = []
    for f in filings[:max_filings]:
        text = fetch_filing_text(f["url"], max_chars=max_chars_per_filing)
        results.append({
            "form_type": f["form_type"],
            "filing_date": f["filing_date"],
            "company": f.get("company", company_name),
            "url": f["url"],
            "text": text or "(could not extract text)",
        })
        time.sleep(0.5)  # Be polite to SEC servers

    return results


def extract_supply_chain_data(ticker: str, filings_text: str) -> str:
    """
    Uses an LLM to scan SEC filings text for Supply Chain, Major Customers,
    and Concentration Risk disclosures (usually found in Item 1 or footnotes).

    Args:
        ticker: The stock ticker (e.g., 'AAPL').
        filings_text: The concatenated text of the retrieved filings.

    Returns:
        A markdown-formatted summary of supply chain & customer disclosures.
    """
    if not filings_text or len(filings_text.strip()) < 100:
        return f"No sufficient SEC filing text found for {ticker} supply chain extraction."

    model_name = get_model("equity", "condense") # Using condense model for extraction
    llm = get_llm(model_name)

    sys_prompt = f"""You are a forensic forensic equity analyst reviewing SEC filings for {ticker}.
Your task is to extract ANY explicit mentions of:
1. Major Customers / Clients (named or described)
2. Revenue Concentration Risk (e.g., "One customer accounted for 15% of revenue")
3. Key Suppliers / Vendors / Manufacturing Partners
4. Supply Chain Risks or Dependencies

Do not guess. Only extract what is explicitly stated in the text.
If the text does NOT mention specific names or concentration metrics, state that explicitly.
Format your output as a concise Markdown brief.
"""

    # Truncate to avoid massive context blowups (take the first 40,000 chars roughly)
    safe_text = filings_text[:40000]

    try:
        messages = [
            SystemMessage(content=sys_prompt),
            HumanMessage(content=f"SEC FILING TEXT:\n\n{safe_text}")
        ]
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        print(f"Warning: Supply chain extraction failed: {e}")
        return f"Failed to extract supply chain data from SEC filings: {e}"
