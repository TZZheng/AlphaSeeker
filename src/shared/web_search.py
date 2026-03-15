"""
Shared Web Search Utilities — DDG + trafilatura page reader.

Three layers:
1. search_web / search_news — discover URLs via DuckDuckGo
2. read_url / read_urls_parallel — extract clean article text via trafilatura
3. deep_search — orchestrates search + read for maximum information extraction

All agents import from here. No domain-specific logic lives in this module.
"""

import time
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from configparser import ConfigParser

from ddgs import DDGS
import trafilatura
from trafilatura.settings import DEFAULT_CONFIG
from src.shared.reliability import cached_retry_call

# ---------------------------------------------------------------------------
# Layer 1: Search (returns URLs + snippets)
# ---------------------------------------------------------------------------

def search_web(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Performs a web search using DuckDuckGo text search.

    Args:
        query: The search query.
        max_results: Maximum number of results to return.

    Returns:
        List of dicts with 'title', 'href', and 'body' (snippet).
    """
    try:
        return cached_retry_call(
            "ddgs_text",
            {"query": query, "max_results": max_results},
            lambda: list(DDGS().text(query, max_results=max_results)),
            ttl_seconds=1800,
            attempts=3,
            min_wait_seconds=2,
            max_wait_seconds=10,
        )
    except Exception as e:
        print(f"Search failed for '{query[:60]}...': {e}")
        return []


def search_news(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Performs a news search using DuckDuckGo news endpoint.

    Args:
        query: The search query.
        max_results: Maximum number of results to return.

    Returns:
        List of dicts with 'title', 'url', 'body', 'date', 'source'.
    """
    def _load() -> List[Dict[str, str]]:
        results = list(DDGS().news(query, max_results=max_results))
        normalized = []
        for r in results:
            normalized.append({
                "title": r.get("title", ""),
                "href": r.get("url", r.get("href", "")),
                "body": r.get("body", ""),
                "date": r.get("date", ""),
                "source": r.get("source", ""),
            })
        return normalized

    try:
        return cached_retry_call(
            "ddgs_news",
            {"query": query, "max_results": max_results},
            _load,
            ttl_seconds=1800,
            attempts=3,
            min_wait_seconds=2,
            max_wait_seconds=10,
        )
    except Exception as e:
        print(f"News search failed for '{query[:60]}...': {e}")
        return []


# ---------------------------------------------------------------------------
# Layer 2: Read (extracts full article text from URLs)
# ---------------------------------------------------------------------------

def read_url(
    url: str,
    max_chars: int = 8000,
    download_timeout_seconds: int = 12,
    extraction_timeout_seconds: int = 12,
) -> Optional[str]:
    """
    Fetches and extracts clean article text from a URL using trafilatura.

    Args:
        url: The URL to read.
        max_chars: Maximum characters to return (truncate if longer).

    Returns:
        Extracted text or None if extraction failed.
    """
    def _load() -> Optional[str]:
        config = ConfigParser()
        config.read_dict({"DEFAULT": dict(DEFAULT_CONFIG.defaults())})
        config["DEFAULT"]["download_timeout"] = str(download_timeout_seconds)
        config["DEFAULT"]["extraction_timeout"] = str(extraction_timeout_seconds)

        downloaded = trafilatura.fetch_url(url, config=config)
        if not downloaded:
            return None
        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            favor_precision=False,
            deduplicate=True,
            config=config,
        )
        if text and len(text) > max_chars:
            text = text[:max_chars] + f"\n... [truncated at {max_chars} chars]"
        return text
    try:
        return cached_retry_call(
            "trafilatura_read",
            {
                "url": url,
                "max_chars": max_chars,
                "download_timeout_seconds": download_timeout_seconds,
                "extraction_timeout_seconds": extraction_timeout_seconds,
            },
            _load,
            ttl_seconds=21600,
            attempts=2,
            min_wait_seconds=1,
            max_wait_seconds=5,
        )
    except Exception as e:
        return None


def read_urls_parallel(
    urls: List[str],
    max_workers: int = 15,
    max_chars_per_url: int = 8000,
    download_timeout_seconds: int = 12,
    extraction_timeout_seconds: int = 12,
) -> Dict[str, str]:
    """
    Reads multiple URLs in parallel using ThreadPoolExecutor.

    Args:
        urls: List of URLs to read.
        max_workers: Number of concurrent threads.
        max_chars_per_url: Max chars per article.

    Returns:
        Dict mapping URL -> extracted text (only successful reads).
    """
    results: Dict[str, str] = {}
    unique_urls = list(set(urls))

    def _fetch(url: str) -> tuple[str, Optional[str]]:
        return url, read_url(
            url,
            max_chars=max_chars_per_url,
            download_timeout_seconds=download_timeout_seconds,
            extraction_timeout_seconds=extraction_timeout_seconds,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch, url): url for url in unique_urls}
        for future in as_completed(futures):
            try:
                url, text = future.result()
                if text:
                    results[url] = text
            except Exception as e:
                print(f"Read failed for URL {futures[future]}: {e}")

    return results


# ---------------------------------------------------------------------------
# Layer 3: Deep Search (search + read full pages)
# ---------------------------------------------------------------------------

def deep_search(
    queries: List[str],
    urls_per_query: int = 3,
    max_workers: int = 15,
    max_chars_per_url: int = 8000,
    search_delay: float = 0.3,
    use_news: bool = False,
    download_timeout_seconds: int = 12,
    extraction_timeout_seconds: int = 12,
) -> List[Dict[str, str]]:
    """
    Orchestrates: search multiple queries → collect top URLs → read full pages.

    Args:
        queries: List of search queries.
        urls_per_query: How many URLs to read per query.
        max_workers: Concurrent URL readers.
        max_chars_per_url: Max chars per article.
        search_delay: Seconds between search requests (rate limiting).
        use_news: If True, use news search instead of text search.

    Returns:
        List of dicts with:
        - 'query': the search query
        - 'title': article title
        - 'url': source URL
        - 'snippet': original search snippet
        - 'full_text': full article text (or None if read failed)
    """
    search_fn = search_news if use_news else search_web
    query_results: List[Dict] = []

    for q in queries:
        results = search_fn(q, max_results=urls_per_query + 2)
        for r in results[:urls_per_query]:
            url = r.get("href", "")
            if url:
                query_results.append({
                    "query": q,
                    "title": r.get("title", ""),
                    "url": url,
                    "snippet": r.get("body", ""),
                    "full_text": None,
                })
        time.sleep(search_delay)

    all_urls = [r["url"] for r in query_results if r["url"]]
    print(f"Deep search: {len(queries)} queries → {len(all_urls)} URLs to read")

    url_texts = read_urls_parallel(
        all_urls,
        max_workers=max_workers,
        max_chars_per_url=max_chars_per_url,
        download_timeout_seconds=download_timeout_seconds,
        extraction_timeout_seconds=extraction_timeout_seconds,
    )

    print(f"Deep search: Successfully read {len(url_texts)} / {len(all_urls)} URLs")

    for r in query_results:
        r["full_text"] = url_texts.get(r["url"])

    return query_results
