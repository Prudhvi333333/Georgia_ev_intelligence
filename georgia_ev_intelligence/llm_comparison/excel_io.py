"""Atomic xlsx I/O helpers + resume support for sreeja-arch.

Generations and evaluation reports are stored as Excel workbooks. We use
pandas to read/write because the column set is wide and the rows can carry
long strings (full prompts, retrieved context). All writes go through a
temp file + os.replace() to make rewrites atomic.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

GENERATION_COLUMNS = [
    "run_id",
    "question_id",
    "category",
    "question",
    "golden_answer",
    "model",
    "mode",
    "answer",
    "retrieved_context",
    "web_context",
    "web_sources",
    "retrieved_count",
    "rerank_top_n",
    "generation_elapsed_s",
    "embedding_model",
    "reranker_model",
    "tavily_used",
    "temperature",
    "prompt_used",
    "timestamp_utc",
    "error",
]


def _atomic_write_excel(df: pd.DataFrame, path: Path, sheet_name: str = "generations") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with pd.ExcelWriter(tmp, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    os.replace(tmp, path)


def write_generations_atomic(rows: list[dict], path: Path) -> None:
    df = pd.DataFrame(rows, columns=GENERATION_COLUMNS)
    _atomic_write_excel(df, path, sheet_name="generations")


def write_workbook_atomic(sheets: dict[str, pd.DataFrame], path: Path) -> None:
    """Write multiple sheets atomically (used by run_llm_evaluation)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with pd.ExcelWriter(tmp, engine="openpyxl") as writer:
        for sheet_name, df in sheets.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    os.replace(tmp, path)


def read_generations(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=GENERATION_COLUMNS)
    df = pd.read_excel(path, sheet_name="generations", engine="openpyxl")
    for col in GENERATION_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df


def completed_keys(df: pd.DataFrame) -> set[tuple[str, str, int]]:
    """Return (model, mode, question_id) triples that completed without error."""
    if df.empty:
        return set()
    keys: set[tuple[str, str, int]] = set()
    for _, row in df.iterrows():
        err = row.get("error", "")
        if isinstance(err, str) and err.strip():
            continue
        try:
            qid = int(row["question_id"])
        except (TypeError, ValueError):
            continue
        keys.add((str(row["model"]), str(row["mode"]), qid))
    return keys
