"""sreeja-arch LLM comparison CLI.

Runs (model x mode x question) generations and writes
georgia_ev_intelligence/outputs/llm_comparison/<run_id>/generations.xlsx.
By default it then evaluates that workbook with scripts/run_llm_evaluation.py.
Use --generation-only or --evaluation-only to split those stages.

Example:
    python -m georgia_ev_intelligence.scripts.run_llm_comparison \\
        --run-id smoke_001 \\
        --models qwen2.5:14b gemma3:27b \\
        --modes rag_only no_rag rag_pretrained rag_pretrained_web \\
        --question-ids 5 7 \\
        --judge-model "$JUDGE_MODEL"
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

THIS = Path(__file__).resolve()
PKG_ROOT = THIS.parent.parent  # .../georgia_ev_intelligence
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from llm_comparison.config import load_generation_config  # noqa: E402
from llm_comparison.excel_io import (  # noqa: E402
    GENERATION_COLUMNS,
    completed_keys,
    read_generations,
    write_generations_atomic,
)
from llm_comparison.generation import run_mode, web_sources_to_str  # noqa: E402
from llm_comparison.prompts import VALID_MODES  # noqa: E402
from llm_comparison.retrieval import ensure_reranker  # noqa: E402

logger = logging.getLogger("scripts.run_llm_comparison")

REPO_ROOT = PKG_ROOT.parent  # .../Georgia_ev_intelligence-1
QUESTIONS_PATH = REPO_ROOT / "kb" / "Human validated 50 questions.xlsx"
OUTPUT_ROOT = PKG_ROOT / "outputs" / "llm_comparison"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id", required=True, help="Stable id used in output filenames + resume.")
    p.add_argument(
        "--models",
        nargs="+",
        default=["qwen2.5:14b", "gemma3:27b"],
        help="One or more Ollama model tags.",
    )
    p.add_argument(
        "--modes",
        nargs="+",
        default=list(VALID_MODES),
        choices=list(VALID_MODES),
        help="One or more generation modes.",
    )
    p.add_argument("--questions", type=int, default=50, help="Cap on number of questions (default 50).")
    p.add_argument(
        "--question-ids",
        nargs="*",
        type=int,
        default=None,
        help="Specific Num values to run (overrides --questions).",
    )
    p.add_argument("--embedding-model", default=None, help="Override OLLAMA_EMBED_MODEL.")
    p.add_argument("--top-k", type=int, default=120, help="Dense retrieval top-K before rerank.")
    p.add_argument("--rerank-top-n", type=int, default=40, help="Final number of context chunks.")
    p.add_argument("--resume", action="store_true", help="Skip rows already in generations.xlsx without errors.")
    p.add_argument("--generation-only", action="store_true", help="Only generate generations.xlsx.")
    p.add_argument("--evaluation-only", action="store_true", help="Only evaluate an existing generations.xlsx.")
    p.add_argument("--judge-model", default=None, help="Override JUDGE_MODEL for evaluation.")
    p.add_argument("--judge-base-url", default=None, help="Override JUDGE_BASE_URL for evaluation.")
    p.add_argument("--ragas-embedding-model", default=None, help="Override OLLAMA_EMBED_MODEL for RAGAS.")
    p.add_argument("--questions-path", default=str(QUESTIONS_PATH), help="Path to the 50-question xlsx.")
    args = p.parse_args()
    if args.generation_only and args.evaluation_only:
        p.error("--generation-only and --evaluation-only are mutually exclusive")
    return args


def load_questions(path: Path, ids: list[int] | None, cap: int) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Questions file not found: {path}")
    df = pd.read_excel(path, engine="openpyxl")
    required = ["Num", "Use Case Category", "Question", "Human validated answers"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Questions xlsx is missing columns: {missing}")
    df = df.dropna(subset=["Num", "Question", "Human validated answers"])

    if ids:
        wanted = set(int(x) for x in ids)
        df = df[df["Num"].astype(int).isin(wanted)]
    else:
        df = df.head(cap)

    rows: list[dict] = []
    for _, r in df.iterrows():
        rows.append(
            {
                "question_id": int(r["Num"]),
                "category": str(r["Use Case Category"]).strip(),
                "question": str(r["Question"]).strip(),
                "golden_answer": str(r["Human validated answers"]).strip(),
            }
        )
    return rows


def run_generation(args: argparse.Namespace) -> Path | None:
    needs_rerank = any(m in args.modes for m in ("rag_only", "rag_pretrained", "rag_pretrained_web"))
    cfg = load_generation_config(
        embedding_model_override=args.embedding_model,
        require_qdrant=needs_rerank,
    )

    if needs_rerank:
        logger.info("Loading mandatory cross-encoder %s ...", cfg.reranker_model)
        ensure_reranker(cfg.reranker_model)

    questions = load_questions(Path(args.questions_path), args.question_ids, args.questions)
    if not questions:
        logger.error("No questions selected. Check --question-ids / --questions / xlsx contents.")
        return None

    out_dir = OUTPUT_ROOT / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "generations.xlsx"

    existing_df = read_generations(out_path) if args.resume else pd.DataFrame(columns=GENERATION_COLUMNS)
    skip_keys = completed_keys(existing_df) if args.resume else set()

    all_rows: list[dict] = existing_df.to_dict(orient="records") if args.resume else []
    new_count = 0
    fail_count = 0
    total_planned = len(args.models) * len(args.modes) * len(questions)

    logger.info(
        "Run %s: %d models x %d modes x %d questions = %d rows (resume=%s, already=%d)",
        args.run_id,
        len(args.models),
        len(args.modes),
        len(questions),
        total_planned,
        args.resume,
        len(skip_keys),
    )

    for model in args.models:
        for mode in args.modes:
            for q in questions:
                key = (model, mode, q["question_id"])
                if key in skip_keys:
                    logger.info("Skipping completed: %s", key)
                    continue

                logger.info(
                    "Generating: model=%s mode=%s qid=%s", model, mode, q["question_id"]
                )
                result = run_mode(
                    mode=mode,
                    question=q["question"],
                    model=model,
                    cfg=cfg,
                    top_k=args.top_k,
                    rerank_top_n=args.rerank_top_n,
                )
                row = {
                    "run_id": args.run_id,
                    "question_id": q["question_id"],
                    "category": q["category"],
                    "question": q["question"],
                    "golden_answer": q["golden_answer"],
                    "model": model,
                    "mode": mode,
                    "answer": result["answer"],
                    "retrieved_context": result["retrieved_context"],
                    "web_context": result["web_context"],
                    "web_sources": web_sources_to_str(result["web_sources"]),
                    "top_k": args.top_k,
                    "retrieved_count": result["retrieved_count"],
                    "rerank_top_n": args.rerank_top_n,
                    "generation_elapsed_s": result["generation_elapsed_s"],
                    "embedding_model": cfg.embedding_model,
                    "reranker_model": cfg.reranker_model if mode != "no_rag" else "",
                    "tavily_used": result["tavily_used"],
                    "temperature": result["temperature"],
                    "prompt_used": result["prompt_used"],
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "error": result["error"],
                }
                all_rows.append(row)
                new_count += 1
                if row["error"]:
                    fail_count += 1

                write_generations_atomic(all_rows, out_path)

    logger.info(
        "Done. Wrote %s. New rows: %d (errors: %d). Total rows in file: %d.",
        out_path,
        new_count,
        fail_count,
        len(all_rows),
    )
    return out_path


def _run_evaluation_from_args(args: argparse.Namespace) -> Path | None:
    from scripts.run_llm_evaluation import evaluate_run

    eval_args = argparse.Namespace(
        run_id=args.run_id,
        judge_model=args.judge_model,
        judge_base_url=args.judge_base_url,
        ragas_embedding_model=args.ragas_embedding_model,
        resume=args.resume,
    )
    return evaluate_run(eval_args)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    if args.evaluation_only:
        return 0 if _run_evaluation_from_args(args) is not None else 2

    if run_generation(args) is None:
        return 2

    if args.generation_only:
        return 0

    logger.info("Generation finished; starting RAGAS evaluation for run_id=%s", args.run_id)
    return 0 if _run_evaluation_from_args(args) is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
