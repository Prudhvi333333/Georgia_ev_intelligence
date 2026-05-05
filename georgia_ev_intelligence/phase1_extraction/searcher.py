"""
Phase 1 — Searcher
Uses Tavily Search API to find documents for each company.

Tavily search_depth="advanced" = 2 API credits per call.
Returns structured results ready for the extractor.

Multiple Tavily keys are supported via shared.tavily — when one key is
exhausted (HTTP 401/402/403/429/432 or quota-style error body) the client
silently rotates to the next key. The pipeline checkpoints progress to disk,
so when ALL keys are exhausted the run can be resumed later without losing
the work that was already saved.
"""
from __future__ import annotations

import asyncio
from typing import Any

import httpx

from shared.logger import get_logger
from shared.tavily import (
    TavilyAllKeysExhausted,
    async_tavily_post,
)

logger = get_logger("phase1.searcher")

TAVILY_SEARCH_URL = "https://api.tavily.com/search"

# Domains known to have high-quality EV supply chain content
PRIORITY_DOMAINS = {
    "georgia.org", "selectgeorgia.com", "savannahjda.com",
    "sec.gov", "energy.gov", "epd.georgia.gov",
    "gaports.com", "hmgma.com", "kiageorgia.com",
    "skon.co", "autonews.com", "emobility.uga.edu",
    "reuters.com", "bloomberg.com",
}

# Domains to exclude from extraction (login walls, paywalls with no value)
BLOCKLIST_DOMAINS = {
    "linkedin.com", "facebook.com", "twitter.com", "x.com",
    "instagram.com", "youtube.com", "tiktok.com",
    "indeed.com", "glassdoor.com", "ziprecruiter.com",
    "maps.google.com", "google.com/maps",
}


def _is_blocked(url: str) -> bool:
    url_lower = url.lower()
    return any(domain in url_lower for domain in BLOCKLIST_DOMAINS)


def _is_priority(url: str) -> bool:
    return any(domain in url.lower() for domain in PRIORITY_DOMAINS)


async def tavily_search(
    query: str,
    max_results: int = 10,
    search_depth: str = "advanced",
) -> list[dict[str, Any]]:
    """
    Call Tavily Search API. Multi-key rotation is inside async_tavily_post.

    Raises TavilyAllKeysExhausted if every key has been disabled — callers
    propagate that so the pipeline can stop cleanly and checkpoint.
    """
    payload = {
        "query": query,
        "max_results": min(max_results, 20),
        "search_depth": search_depth,
        "include_answer": False,
        "include_images": False,
        "include_raw_content": False,
    }
    data = await async_tavily_post(TAVILY_SEARCH_URL, payload, timeout=30.0)

    raw_results = data.get("results", [])
    results = []
    for item in raw_results:
        url = item.get("url", "")
        if not url or _is_blocked(url):
            continue
        results.append({
            "url": url,
            "title": item.get("title", ""),
            "snippet": item.get("content", ""),
            "score": float(item.get("score", 0.0)),
            "is_priority": _is_priority(url),
        })

    logger.debug("Tavily search '%s' → %d results", query[:60], len(results))
    return results


async def search_company(
    company: dict[str, Any],
    queries: list[dict[str, Any]],
    concurrency: int = 3,
    delay_between_batches: float = 0.5,
) -> list[dict[str, Any]]:
    """
    Run all queries for a single company and collect unique URLs.

    Each result carries the originating Tavily query + query_family so the
    metadata builder can record search provenance per document.
    """
    company_name = company.get("company_name", "")
    logger.info("Searching for %s (%d queries)", company_name, len(queries))

    seen_urls: set[str] = set()
    all_results: list[dict[str, Any]] = []

    semaphore = asyncio.Semaphore(concurrency)

    async def _run_query(q: dict[str, Any]) -> list[dict[str, Any]]:
        async with semaphore:
            try:
                results = await tavily_search(
                    query=q["query_text"],
                    max_results=q.get("max_results", 10),
                    search_depth=q.get("search_depth", "advanced"),
                )
                for r in results:
                    r["query"] = q["query_text"]
                    r["family"] = q.get("family")
                return results
            except TavilyAllKeysExhausted:
                raise
            except httpx.HTTPStatusError as exc:
                logger.warning(
                    "Tavily search failed [%d] for '%s'",
                    exc.response.status_code, q["query_text"][:50],
                )
                return []
            except Exception as exc:
                logger.warning("Tavily search error for '%s': %s", q["query_text"][:50], exc)
                return []

    batch_size = concurrency * 2
    for batch_start in range(0, len(queries), batch_size):
        batch = queries[batch_start: batch_start + batch_size]
        tasks = [_run_query(q) for q in batch]
        batch_results = await asyncio.gather(*tasks)

        for query_results in batch_results:
            for result in query_results:
                url = result["url"]
                if url not in seen_urls:
                    seen_urls.add(url)
                    result["company_id"] = company.get("id")
                    result["company_name"] = company_name
                    all_results.append(result)

        if batch_start + batch_size < len(queries):
            await asyncio.sleep(delay_between_batches)

    all_results.sort(key=lambda r: (-int(r["is_priority"]), -r["score"]))
    logger.info(
        "Company '%s': %d unique URLs found (%d priority)",
        company_name,
        len(all_results),
        sum(1 for r in all_results if r["is_priority"]),
    )
    return all_results
