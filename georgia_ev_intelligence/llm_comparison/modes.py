"""Per-mode generation runners.

Each mode returns a dict with the full set of fields needed by the
generations.xlsx schema (answer, retrieved_context, web_context, web_sources,
retrieved_count, rerank_top_n, generation_elapsed_s, prompt_used,
tavily_used, error).

LLM calls go straight to Ollama's /api/generate at temperature=0. Tavily
calls reuse evaluate.format_runner's per-question helper.
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
# We reuse evaluate.format_runner._tavily_search_structured so the per-question
# Tavily call has a single source of truth. Imported lazily so a missing
# dependency only matters when mode 4 is actually requested.

def _run_tavily(question: str, api_key: str) -> tuple[str, list[dict[str, str]]]:
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is required for rag_pretrained_web.")
    from evaluate.format_runner import _tavily_search_structured

    web_context, sources = _tavily_search_structured(question, api_key=api_key)
    if not web_context.strip():
        raise RuntimeError(
            "Tavily returned no web context for rag_pretrained_web; "
            "cannot satisfy the all-three-sources mode."
        )
    return web_context, sources


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
                web_context, web_sources = _run_tavily(question, cfg.tavily_api_key)
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
