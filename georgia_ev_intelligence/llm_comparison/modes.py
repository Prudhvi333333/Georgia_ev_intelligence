"""Per-mode generation runners.

Each mode returns a dict with the full set of fields needed by the
generations.xlsx schema (answer, retrieved_context, web_context, web_sources,
retrieved_count, rerank_top_n, generation_elapsed_s, prompt_used,
tavily_used, error).

LLM calls go straight to Ollama's /api/generate at temperature=0. Tavily
calls are issued directly (we don't import evaluate.format_runner because
it loads the heavy shared.config singleton at import time).
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx

from llm_comparison.config import GenerationConfig
from llm_comparison.prompts import build_prompt
from llm_comparison.retrieval import format_context, retrieve_and_rerank

logger = logging.getLogger("llm_comparison.modes")


# ── Ollama generation ───────────────────────────────────────────────────────

def _call_ollama(model: str, prompt: str, base_url: str, timeout: float = 600.0) -> str:
    response = httpx.post(
        f"{base_url}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 800,
                "num_ctx": 8192,
            },
        },
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    return (data.get("response") or "").strip()


# ── Tavily web search (mode 4 only) ─────────────────────────────────────────

def _tavily_search(question: str, api_key: str) -> tuple[str, list[dict[str, str]]]:
    """Return (formatted_context_string, [{title,url}, ...]). Empty on error."""
    if not api_key:
        return "", []
    try:
        from tavily import TavilyClient
    except ImportError as exc:
        raise RuntimeError(
            "tavily-python is not installed. Run: pip install tavily-python"
        ) from exc

    try:
        client = TavilyClient(api_key=api_key)
        result = client.search(
            query=f"Georgia EV supply chain {question}",
            max_results=3,
            search_depth="basic",
        )
    except Exception as exc:
        logger.warning("Tavily search failed: %s", exc)
        return "", []

    items = result.get("results") or []
    sources: list[dict[str, str]] = []
    blocks: list[str] = []
    for item in items:
        title = (item.get("title") or "").strip()
        url = (item.get("url") or "").strip()
        content = (item.get("content") or "").strip()
        if not content:
            continue
        sources.append({"title": title, "url": url})
        truncated = content[:600]
        blocks.append(f"[{title}] {truncated} (source: {url})")
    return "\n".join(blocks), sources


# ── Mode runners ────────────────────────────────────────────────────────────

def _empty_result() -> dict[str, Any]:
    return {
        "answer": "",
        "retrieved_context": "",
        "web_context": "",
        "web_sources": [],
        "retrieved_count": 0,
        "rerank_top_n": 0,
        "generation_elapsed_s": 0.0,
        "tavily_used": False,
        "prompt_used": "",
        "temperature": 0.0,
        "error": "",
    }


def run_mode(
    mode: str,
    question: str,
    model: str,
    cfg: GenerationConfig,
    top_k: int,
    rerank_top_n: int,
) -> dict[str, Any]:
    out = _empty_result()

    try:
        if mode == "no_rag":
            internal_context = ""
            web_context = ""
            web_sources: list[dict[str, str]] = []
            retrieved_count = 0
            actual_rerank_top_n = 0
            tavily_used = False
        elif mode in ("rag_only", "rag_pretrained", "rag_pretrained_web"):
            hits = retrieve_and_rerank(
                question=question,
                cfg=cfg,
                top_k=top_k,
                rerank_top_n=rerank_top_n,
            )
            retrieved_count = len(hits)
            actual_rerank_top_n = min(rerank_top_n, retrieved_count)
            internal_context = format_context(hits)
            if mode == "rag_pretrained_web":
                web_context, web_sources = _tavily_search(question, cfg.tavily_api_key)
                tavily_used = True
            else:
                web_context, web_sources = "", []
                tavily_used = False
        else:
            raise ValueError(f"Unknown mode: {mode!r}")

        prompt = build_prompt(
            mode=mode,
            question=question,
            internal_context=internal_context,
            web_context=web_context,
        )

        start = time.monotonic()
        answer = _call_ollama(model=model, prompt=prompt, base_url=cfg.ollama_base_url)
        elapsed = time.monotonic() - start

        out.update(
            {
                "answer": answer,
                "retrieved_context": internal_context,
                "web_context": web_context,
                "web_sources": web_sources,
                "retrieved_count": retrieved_count,
                "rerank_top_n": actual_rerank_top_n,
                "generation_elapsed_s": round(elapsed, 3),
                "tavily_used": tavily_used,
                "prompt_used": prompt,
                "temperature": 0.0,
                "error": "",
            }
        )
    except Exception as exc:
        logger.exception("Mode %s failed for q=%r model=%r: %s", mode, question[:60], model, exc)
        out["error"] = f"{type(exc).__name__}: {exc}"

    return out


def web_sources_to_str(sources: list[dict[str, str]]) -> str:
    if not sources:
        return ""
    return json.dumps(sources, ensure_ascii=False)
