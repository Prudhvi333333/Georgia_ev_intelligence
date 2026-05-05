"""sreeja-arch evaluation CLI.

Reads georgia_ev_intelligence/outputs/llm_comparison/<run_id>/generations.xlsx
and produces georgia_ev_intelligence/outputs/ragas_reports/<run_id>.xlsx
with five sheets: per_row, agg_model_mode, agg_model_mode_metric,
agg_category_mode, run_metadata.

The judge LLM is Kimi 2.6 (or any OpenAI-compatible chat model) wired via
langchain-openai. Embeddings used by ragas come from Ollama
(nomic-embed-text by default).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

THIS = Path(__file__).resolve()
PKG_ROOT = THIS.parent.parent  # .../georgia_ev_intelligence
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from llm_comparison.config import load_judge_config  # noqa: E402
from llm_comparison.excel_io import read_generations, write_workbook_atomic  # noqa: E402
from llm_comparison.ragas_runner import (  # noqa: E402
    ALL_METRICS,
    build_aggregations,
    evaluate_rows,
)

logger = logging.getLogger("scripts.run_llm_evaluation")

OUTPUT_ROOT = PKG_ROOT / "outputs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True, help="Locates outputs/llm_comparison/<run_id>/generations.xlsx")
    p.add_argument("--judge-model", default=None, help="Override JUDGE_MODEL env var (e.g. kimi-k2-0905-preview).")
    p.add_argument("--judge-base-url", default=None, help="Override JUDGE_BASE_URL env var.")
    p.add_argument("--ragas-embedding-model", default=None, help="Override OLLAMA_EMBED_MODEL for ragas.")
    p.add_argument("--resume", action="store_true", help="Skip rows already scored in the report.")
    return p.parse_args()


def _existing_keys(report_path: Path) -> set[tuple[str, str, int]]:
    if not report_path.exists():
        return set()
    try:
        df = pd.read_excel(report_path, sheet_name="per_row", engine="openpyxl")
    except Exception:
        return set()
    keys: set[tuple[str, str, int]] = set()
    for _, row in df.iterrows():
        try:
            qid = int(row["question_id"])
        except (TypeError, ValueError, KeyError):
            continue
        keys.add((str(row.get("model", "")), str(row.get("mode", "")), qid))
    return keys


def _existing_per_row(report_path: Path) -> pd.DataFrame:
    if not report_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_excel(report_path, sheet_name="per_row", engine="openpyxl")
    except Exception:
        return pd.DataFrame()


def _build_run_metadata(args, judge_cfg, gen_df: pd.DataFrame) -> pd.DataFrame:
    models = sorted(gen_df["model"].dropna().unique().tolist()) if not gen_df.empty else []
    modes = sorted(gen_df["mode"].dropna().unique().tolist()) if not gen_df.empty else []
    embedding_models = (
        sorted(gen_df["embedding_model"].dropna().unique().tolist()) if not gen_df.empty else []
    )
    reranker_models = (
        sorted(set(x for x in gen_df["reranker_model"].dropna().tolist() if x)) if not gen_df.empty else []
    )
    return pd.DataFrame(
        [
            {
                "run_id": args.run_id,
                "models": json.dumps(models),
                "modes": json.dumps(modes),
                "embedding_model": json.dumps(embedding_models),
                "reranker_model": json.dumps(reranker_models),
                "judge_model": judge_cfg.judge_model,
                "judge_base_url": judge_cfg.judge_base_url,
                "ragas_embedding_model": judge_cfg.ragas_embedding_model,
                "weights": json.dumps(
                    {
                        "faithfulness": 0.25,
                        "answer_relevancy": 0.20,
                        "context_precision": 0.20,
                        "context_recall": 0.20,
                        "answer_correctness": 0.15,
                    }
                ),
                "metrics": json.dumps(ALL_METRICS),
                "total_rows": int(len(gen_df)),
                "temperature": 0.0,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        ]
    )


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    judge_cfg = load_judge_config(
        judge_model_override=args.judge_model,
        judge_base_url_override=args.judge_base_url,
        ragas_embedding_model_override=args.ragas_embedding_model,
    )

    gen_path = PKG_ROOT / "outputs" / "llm_comparison" / args.run_id / "generations.xlsx"
    if not gen_path.exists():
        raise FileNotFoundError(
            f"generations.xlsx not found at {gen_path}. "
            f"Run scripts/run_llm_comparison.py with the same --run-id first."
        )

    gen_df = read_generations(gen_path)
    if gen_df.empty:
        logger.error("No rows in %s; nothing to evaluate.", gen_path)
        return 2

    report_path = OUTPUT_ROOT / "ragas_reports" / f"{args.run_id}.xlsx"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    if args.resume:
        already = _existing_keys(report_path)
        if already:
            keep_mask = ~gen_df.apply(
                lambda r: (str(r["model"]), str(r["mode"]), int(r["question_id"])) in already,
                axis=1,
            )
            to_score = gen_df[keep_mask].copy()
            logger.info("Resume: %d rows already scored; %d remaining.", len(already), len(to_score))
        else:
            to_score = gen_df
    else:
        to_score = gen_df

    new_per_row = evaluate_rows(to_score, judge_cfg) if not to_score.empty else pd.DataFrame()

    if args.resume:
        prior = _existing_per_row(report_path)
        per_row = pd.concat([prior, new_per_row], ignore_index=True) if not prior.empty else new_per_row
    else:
        per_row = new_per_row

    if per_row.empty:
        logger.warning("No per_row rows produced; writing only run_metadata.")
        per_row = pd.DataFrame()

    aggregations = build_aggregations(per_row)
    sheets: dict[str, pd.DataFrame] = {
        "per_row": per_row,
        **aggregations,
        "run_metadata": _build_run_metadata(args, judge_cfg, gen_df),
    }
    write_workbook_atomic(sheets, report_path)

    logger.info(
        "Wrote %s (per_row=%d rows, judge=%s).",
        report_path,
        len(per_row),
        judge_cfg.judge_model,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
