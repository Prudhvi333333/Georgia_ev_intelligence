"""Atomic xlsx I/O helpers + resume support for sreeja-arch.

Generations and evaluation reports are stored as Excel workbooks. We use
pandas to read/write because the column set is wide and the rows can carry
long strings (full prompts, retrieved context). All writes go through a
temp file + os.replace() to make rewrites atomic.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

EXCEL_MAX_CELL_CHARS = 32767
# XML 1.0 disallows most C0 control chars. openpyxl raises
# IllegalCharacterError if any of these appear in scraped/PDF web text.
_ILLEGAL_XLSX_CHARS = re.compile(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]")

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
    "top_k",
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


def _clean_excel_value(value: Any) -> Any:
    """Return a value safe for openpyxl/xlsx cells."""
    if not isinstance(value, str):
        return value
    value = _ILLEGAL_XLSX_CHARS.sub("", value)
    if len(value) > EXCEL_MAX_CELL_CHARS:
        return value[: EXCEL_MAX_CELL_CHARS - 32] + "\n...[truncated for Excel]..."
    return value


def _clean_excel_df(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize object columns before pandas hands values to openpyxl."""
    cleaned = df.copy()
    for col in cleaned.select_dtypes(include=["object"]).columns:
        cleaned[col] = cleaned[col].map(_clean_excel_value)
    return cleaned


def _atomic_write_excel(df: pd.DataFrame, path: Path, sheet_name: str = "generations") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    safe_df = _clean_excel_df(df)
    with pd.ExcelWriter(tmp, engine="openpyxl") as writer:
        safe_df.to_excel(writer, sheet_name=sheet_name, index=False)
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
            _clean_excel_df(df).to_excel(writer, sheet_name=sheet_name[:31], index=False)
    os.replace(tmp, path)


_STRING_COLS = {
    "run_id", "category", "question", "golden_answer", "model", "mode",
    "answer", "retrieved_context", "web_context", "web_sources",
    "embedding_model", "reranker_model", "prompt_used", "timestamp_utc",
    "error",
}


def read_generations(path: Path) -> pd.DataFrame:
    """Read generations.xlsx and normalise NaN to empty strings for string
    columns. Without this, downstream consumers crash on `(NaN or '').strip()`
    and `_split_contexts(NaN)` returns the literal string 'nan'.
    """
    if not path.exists():
        return pd.DataFrame(columns=GENERATION_COLUMNS)
    df = pd.read_excel(path, sheet_name="generations", engine="openpyxl")
    for col in GENERATION_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    # Cast string columns to object dtype, then fill NaN with "".
    for col in _STRING_COLS:
        if col in df.columns:
            df[col] = df[col].astype(object).where(df[col].notna(), "")
    return df


def _is_blank_error(err) -> bool:
    if err is None:
        return True
    try:
        if pd.isna(err):
            return True
    except (TypeError, ValueError):
        pass
    return not str(err).strip()


def completed_keys(df: pd.DataFrame) -> set[tuple[str, str, int]]:
    """Return (model, mode, question_id) triples that completed without error."""
    if df.empty:
        return set()
    keys: set[tuple[str, str, int]] = set()
    for _, row in df.iterrows():
        if not _is_blank_error(row.get("error", "")):
            continue
        try:
            qid = int(row["question_id"])
        except (TypeError, ValueError):
            continue
        keys.add((str(row["model"]), str(row["mode"]), qid))
    return keys
