"""
Phase 1 — Main Pipeline Orchestrator

End-to-end:
  GNEM Excel → PostgreSQL company table
            → Tavily multi-key search per company
            → Free download (HTTP GET) OR Tavily Extract per URL
            → Backblaze B2 raw storage
            → PostgreSQL gev_documents row (RAG framework metadata)

Resumability:
  A JSON checkpoint at logs/phase1_checkpoint.json records every company and
  every URL that has been finished. On restart, those are skipped, so:

   - If Tavily keys run out mid-run, just add more keys to .env and re-run —
     the pipeline picks up exactly where it left off.
   - If the process is killed, re-run; nothing is re-uploaded or re-processed.

Usage:
  python -m phase1_extraction.pipeline                    # full run
  python -m phase1_extraction.pipeline --limit 3          # first 3 companies
  python -m phase1_extraction.pipeline --company "Hanwha" # single company
  python -m phase1_extraction.pipeline --load-only        # only sync GNEM → DB
  python -m phase1_extraction.pipeline --rerun            # ignore doc-count gate
  python -m phase1_extraction.pipeline --rerun-all        # also wipe the checkpoint
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from typing import Any

from phase1_extraction.checkpoint import get_checkpoint
from phase1_extraction.doc_storage import (
    get_document_count_for_company,
    mark_document_failed,
    save_document,
)
from phase1_extraction.extractor import extract_document
from phase1_extraction.kb_loader import (
    get_all_companies_from_db,
    load_companies_from_excel,
    sync_companies_to_db,
)
from phase1_extraction.query_generator import build_queries, estimate_query_count
from phase1_extraction.searcher import search_company
from shared.logger import get_logger
from shared.tavily import TavilyAllKeysExhausted

logger = get_logger("phase1.pipeline")

COMPANY_CONCURRENCY = 3
MAX_URLS_PER_COMPANY = 8
MIN_RELEVANCE_SCORE = 0.4


class _PipelineHalted(Exception):
    """Raised internally to stop the run cleanly when keys are exhausted."""


async def process_company(
    company: dict[str, Any],
    skip_if_has_docs: bool = True,
) -> dict[str, Any]:
    """Full pipeline for one company. Idempotent — safe to re-run."""
    company_name = company.get("company_name", "")
    start = time.monotonic()

    checkpoint = get_checkpoint()

    result = {
        "company": company_name,
        "urls_found": 0,
        "docs_extracted": 0,
        "docs_failed": 0,
        "skipped": False,
        "error": None,
    }

    if checkpoint.is_company_done(company_name):
        logger.info("Skipping '%s' — already in checkpoint", company_name)
        result["skipped"] = True
        return result

    if skip_if_has_docs and get_document_count_for_company(company_name) >= 5:
        logger.info("Skipping '%s' — already has ≥5 docs in DB", company_name)
        checkpoint.mark_company_done(company_name, {"reason": "already_has_docs"})
        result["skipped"] = True
        return result

    try:
        queries = build_queries(company)
        logger.info("[%s] Generated %d queries", company_name, len(queries))

        url_results = await search_company(company, queries)
        result["urls_found"] = len(url_results)

        url_results = [
            u for u in url_results
            if u.get("score", 0.0) >= MIN_RELEVANCE_SCORE
        ]
        url_results.sort(key=lambda u: u.get("score", 0), reverse=True)
        url_results = url_results[:MAX_URLS_PER_COMPANY]
        logger.info(
            "[%s] %d URLs after relevance filter (score >= %.1f)",
            company_name, len(url_results), MIN_RELEVANCE_SCORE,
        )

        for hit in url_results:
            url = hit["url"]
            if checkpoint.is_url_done(url):
                logger.info("[%s] Skipping URL — already saved: %s", company_name, url[:80])
                result["docs_extracted"] += 1
                continue

            try:
                extracted = await extract_document(url=url, company_name=company_name)
            except TavilyAllKeysExhausted:
                # Stop right here — checkpoint preserves state.
                raise _PipelineHalted(
                    f"All Tavily keys exhausted while processing '{company_name}'."
                )

            if extracted.error or not extracted.text:
                mark_document_failed(url, extracted.error or "no text", company, hit)
                result["docs_failed"] += 1
                continue

            doc_id = save_document(extracted=extracted, company=company, search_hit=hit)
            if doc_id is None:
                result["docs_failed"] += 1
                continue

            checkpoint.mark_url_done(url)
            result["docs_extracted"] += 1

    except _PipelineHalted:
        # Persist what we know and bubble up.
        checkpoint.mark_company_failed(
            company_name, "halted: tavily keys exhausted (resumable)"
        )
        raise
    except TavilyAllKeysExhausted:
        # Search step itself ran out of keys — same outcome.
        checkpoint.mark_company_failed(
            company_name, "halted: tavily keys exhausted during search"
        )
        raise _PipelineHalted(
            f"All Tavily keys exhausted while searching '{company_name}'."
        )
    except Exception as exc:
        logger.error("Pipeline error for '%s': %s", company_name, exc, exc_info=True)
        result["error"] = str(exc)[:200]
        checkpoint.mark_company_failed(company_name, str(exc))
        return result

    checkpoint.mark_company_done(company_name, {
        "urls_found": result["urls_found"],
        "docs_extracted": result["docs_extracted"],
        "docs_failed": result["docs_failed"],
    })
    elapsed = time.monotonic() - start
    logger.info(
        "[%s] Done in %.1fs — %d URLs | %d docs | %d failed",
        company_name, elapsed,
        result["urls_found"], result["docs_extracted"], result["docs_failed"],
    )
    return result


async def run_pipeline(
    companies: list[dict[str, Any]],
    concurrency: int = COMPANY_CONCURRENCY,
    skip_if_has_docs: bool = True,
) -> tuple[list[dict[str, Any]], bool]:
    """
    Process companies under a concurrency cap.
    Returns (results, halted). If halted=True, every Tavily key is gone —
    the pipeline persisted state and exited cleanly.
    """
    semaphore = asyncio.Semaphore(concurrency)
    results: list[dict[str, Any]] = []
    halted = False

    async def _process(company: dict[str, Any]) -> None:
        nonlocal halted
        if halted:
            return
        async with semaphore:
            if halted:
                return
            try:
                res = await process_company(company, skip_if_has_docs=skip_if_has_docs)
                results.append(res)
            except _PipelineHalted as exc:
                halted = True
                logger.error("HALT: %s", exc)

    tasks = [_process(c) for c in companies]
    await asyncio.gather(*tasks)
    return results, halted


def print_summary(results: list[dict[str, Any]], halted: bool) -> None:
    total = len(results)
    skipped = sum(1 for r in results if r["skipped"])
    processed = total - skipped
    total_docs = sum(r["docs_extracted"] for r in results)
    total_failed = sum(r["docs_failed"] for r in results)
    errors = [r for r in results if r.get("error")]

    print("\n" + "=" * 60)
    print("PHASE 1 PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Companies processed : {processed}/{total} (skipped: {skipped})")
    print(f"Documents extracted : {total_docs}")
    print(f"Documents failed    : {total_failed}")
    if errors:
        print(f"\nErrors ({len(errors)} companies):")
        for e in errors:
            print(f"  - {e['company']}: {e['error']}")
    if halted:
        print("\n!! PIPELINE HALTED: All Tavily keys are exhausted.")
        print("   Add more keys to .env (TAVILY_API_KEYS=...) and re-run.")
        print("   Progress is checkpointed at logs/phase1_checkpoint.json.")
    print("=" * 60 + "\n")


async def main_async(args: argparse.Namespace) -> None:
    checkpoint = get_checkpoint()
    if args.rerun_all:
        checkpoint.reset()

    logger.info(
        "Checkpoint: %s — %d companies / %d URLs already done",
        checkpoint.path,
        checkpoint.completed_company_count(),
        checkpoint.completed_url_count(),
    )

    logger.info("Loading GNEM Excel → PostgreSQL...")
    companies_data = load_companies_from_excel()
    inserted, updated = sync_companies_to_db(companies_data)
    logger.info("GNEM sync: %d inserted, %d updated", inserted, updated)

    if args.load_only:
        logger.info("--load-only flag set — stopping after DB sync.")
        return

    all_companies = get_all_companies_from_db()
    logger.info("Loaded %d companies from DB", len(all_companies))

    if args.company:
        target = args.company.lower()
        all_companies = [c for c in all_companies if target in c["company_name"].lower()]
        if not all_companies:
            logger.error("No company matching '%s'", args.company)
            sys.exit(1)
        logger.info("Filtered to %d matching company/ies", len(all_companies))

    if args.limit:
        all_companies = all_companies[: args.limit]
        logger.info("Limiting to first %d companies", args.limit)

    estimate = estimate_query_count(all_companies)
    logger.info(
        "Credit estimate: %d Tavily queries, ~%d credits (advanced depth)",
        estimate["total_queries"],
        estimate["estimated_tavily_credits"],
    )

    results, halted = await run_pipeline(
        companies=all_companies,
        concurrency=args.concurrency,
        skip_if_has_docs=not args.rerun,
    )

    print_summary(results, halted)


def main() -> None:
    parser = argparse.ArgumentParser(description="Georgia EV Intelligence — Phase 1 Pipeline")
    parser.add_argument("--company", type=str, default=None,
                        help="Run for a specific company name (partial match)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N companies")
    parser.add_argument("--load-only", action="store_true",
                        help="Only load GNEM Excel to DB, skip search + extraction")
    parser.add_argument("--rerun", action="store_true",
                        help="Re-process companies even if they already have ≥5 docs")
    parser.add_argument("--rerun-all", action="store_true",
                        help="Wipe the checkpoint and process everything from scratch")
    parser.add_argument("--concurrency", type=int, default=COMPANY_CONCURRENCY,
                        help=f"Parallel companies (default: {COMPANY_CONCURRENCY})")
    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
