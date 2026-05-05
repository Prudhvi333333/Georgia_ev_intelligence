"""Self-contained env-var loader for sreeja-arch llm_comparison.

Reads only the variables this pipeline actually needs. Does NOT touch
shared.config.Config (which mandates Neo4j / Postgres / B2 vars unrelated
to LLM comparison).

A .env file at the repo root or inside georgia_ev_intelligence/ is loaded
opportunistically via python-dotenv if available.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv_if_present() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    here = Path(__file__).resolve().parent
    for parent in [here, *here.parents]:
        candidate = parent / ".env"
        if candidate.exists():
            load_dotenv(candidate, override=False)
            return


_load_dotenv_if_present()


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"Set it in your .env (see .env.example)."
        )
    return value


def _optional(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


@dataclass(frozen=True)
class GenerationConfig:
    ollama_base_url: str
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection: str
    qdrant_dense_name: str
    qdrant_sparse_name: str
    embedding_model: str
    reranker_model: str
    tavily_api_key: str  # may be empty if mode 4 not used


def load_generation_config(
    embedding_model_override: str | None = None,
) -> GenerationConfig:
    return GenerationConfig(
        ollama_base_url=_optional("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
        qdrant_url=_require("QDRANT_URL"),
        qdrant_api_key=_require("QDRANT_API_KEY"),
        qdrant_collection=_optional("QDRANT_COLLECTION_NAME", "georgia_ev_chunks"),
        qdrant_dense_name=_optional("QDRANT_DENSE_VECTOR_NAME", "dense"),
        qdrant_sparse_name=_optional("QDRANT_SPARSE_VECTOR_NAME", "sparse"),
        embedding_model=embedding_model_override or _optional("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        reranker_model=_optional(
            "CROSS_ENCODER_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L12-v2"
        ),
        tavily_api_key=os.environ.get("TAVILY_API_KEY", "").strip(),
    )


@dataclass(frozen=True)
class JudgeConfig:
    judge_base_url: str
    judge_api_key: str
    judge_model: str
    ollama_base_url: str
    ragas_embedding_model: str


def load_judge_config(
    judge_model_override: str | None = None,
    judge_base_url_override: str | None = None,
    ragas_embedding_model_override: str | None = None,
) -> JudgeConfig:
    base_url = judge_base_url_override or _require("JUDGE_BASE_URL")
    api_key = _require("JUDGE_API_KEY")
    model = judge_model_override or _require("JUDGE_MODEL")
    return JudgeConfig(
        judge_base_url=base_url.rstrip("/"),
        judge_api_key=api_key,
        judge_model=model,
        ollama_base_url=_optional("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
        ragas_embedding_model=ragas_embedding_model_override
        or _optional("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
    )
