"""Dense Qdrant retrieval + mandatory cross-encoder rerank.

Per the sreeja-arch spec:
  * Retrieve dense-only from the existing Qdrant collection (sparse/RRF
    is deliberately bypassed for the comparison runs).
  * Rerank the top-K candidates with cross-encoder/ms-marco-MiniLM-L12-v2.
  * Reranker load failure is fatal: no silent fallback to dense ordering.
  * The pipeline returns the top rerank_top_n hits for use in the prompt.

This module is self-contained: it talks to Qdrant via qdrant-client and
embeds queries via Ollama's /api/embed endpoint, without going through
shared.config.Config.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import httpx
from qdrant_client import QdrantClient, models

from llm_comparison.config import GenerationConfig

logger = logging.getLogger("llm_comparison.retrieval")


# ── Reranker (mandatory) ────────────────────────────────────────────────────

@lru_cache(maxsize=4)
def _load_cross_encoder(model_name: str):
    """Force-load the cross-encoder. Hard fail on any error."""
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise RuntimeError(
            "Reranker is mandatory; sentence-transformers is not installed. "
            "Run: pip install sentence-transformers"
        ) from exc

    try:
        return CrossEncoder(model_name)
    except Exception as exc:
        raise RuntimeError(
            f"Reranker is mandatory; failed to load CrossEncoder({model_name!r}): {exc}"
        ) from exc


def ensure_reranker(model_name: str) -> None:
    """Eagerly load the cross-encoder so a missing model fails fast."""
    _load_cross_encoder(model_name)


# ── Qdrant client (singleton per (url, key)) ────────────────────────────────

@lru_cache(maxsize=4)
def _get_qdrant_client(url: str, api_key: str) -> QdrantClient:
    return QdrantClient(url=url, api_key=api_key, timeout=60)


# ── Query embedding via Ollama ──────────────────────────────────────────────

def _embed_query(question: str, ollama_base_url: str, embed_model: str) -> list[float]:
    url = f"{ollama_base_url}/api/embed"
    response = httpx.post(
        url,
        json={"model": embed_model, "input": [question]},
        timeout=120.0,
    )
    response.raise_for_status()
    data = response.json()
    embeddings = data.get("embeddings", [])
    if not embeddings:
        raise RuntimeError(f"Ollama returned no embeddings for query: {question!r}")
    return embeddings[0]


# ── Dense Qdrant search ─────────────────────────────────────────────────────

def _search_dense(
    client: QdrantClient,
    collection: str,
    dense_name: str,
    query_vector: list[float],
    top_k: int,
) -> list[dict[str, Any]]:
    result = client.query_points(
        collection_name=collection,
        query=query_vector,
        using=dense_name,
        limit=top_k,
        with_payload=True,
    )
    hits: list[dict[str, Any]] = []
    for point in result.points:
        payload = point.payload or {}
        text = payload.get("parent_text") or payload.get("text") or ""
        hits.append(
            {
                "score": float(point.score) if point.score is not None else 0.0,
                "text": text,
                "company_name": payload.get("company_name", ""),
                "source_url": payload.get("source_url", ""),
                "chunk_id": payload.get("chunk_id", str(point.id)),
                "metadata": payload,
            }
        )
    return hits


# ── Rerank ──────────────────────────────────────────────────────────────────

def _rerank(question: str, hits: list[dict[str, Any]], reranker_model: str) -> list[dict[str, Any]]:
    if not hits:
        return hits
    encoder = _load_cross_encoder(reranker_model)
    pairs = [(question, hit["text"]) for hit in hits]
    scores = encoder.predict(pairs)
    for hit, score in zip(hits, scores):
        hit["rerank_score"] = float(score)
    hits.sort(key=lambda h: h["rerank_score"], reverse=True)
    return hits


# ── Public entrypoint ───────────────────────────────────────────────────────

def retrieve_and_rerank(
    question: str,
    cfg: GenerationConfig,
    top_k: int = 8,
    rerank_top_n: int = 4,
) -> list[dict[str, Any]]:
    """Embed → dense search → cross-encoder rerank → top-N."""
    client = _get_qdrant_client(cfg.qdrant_url, cfg.qdrant_api_key)
    query_vector = _embed_query(question, cfg.ollama_base_url, cfg.embedding_model)
    hits = _search_dense(
        client=client,
        collection=cfg.qdrant_collection,
        dense_name=cfg.qdrant_dense_name,
        query_vector=query_vector,
        top_k=top_k,
    )
    reranked = _rerank(question, hits, cfg.reranker_model)
    return reranked[:rerank_top_n]


def format_context(hits: list[dict[str, Any]]) -> str:
    """Render reranked hits into a single string for the LLM prompt."""
    if not hits:
        return ""
    blocks: list[str] = []
    for i, hit in enumerate(hits, start=1):
        company = hit.get("company_name") or ""
        url = hit.get("source_url") or ""
        prefix = f"[{i}]"
        if company:
            prefix += f" {company}"
        if url:
            prefix += f" ({url})"
        blocks.append(f"{prefix}\n{hit.get('text', '').strip()}")
    return "\n\n".join(blocks)
