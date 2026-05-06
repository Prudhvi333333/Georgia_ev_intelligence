"""Official RAGAS evaluation wired to a Kimi 2.6 OpenAI-compatible judge.

Per-row metrics:
  faithfulness       (skipped for no_rag — context is empty)
  answer_relevancy   (always)
  context_precision  (skipped for no_rag)
  context_recall     (skipped for no_rag)
  answer_correctness (always)

Final score weights (renormalized when some metrics are skipped):
  faithfulness        0.25
  answer_relevancy    0.20
  context_precision   0.20
  context_recall      0.20
  answer_correctness  0.15

Deterministic set metrics (computed without LLM):
  gold_companies           — company names found in golden_answer
  context_companies        — company names found in retrieved_context
  answer_companies         — company names found in answer
  context_company_recall   — |gold ∩ context| / |gold|
  answer_company_recall    — |gold ∩ answer| / |gold|
  answer_company_precision — |gold ∩ answer| / |answer|
  answer_company_f1        — harmonic mean of recall and precision
  count_exact_match        — 1 if golden count integer appears in answer
"""
from __future__ import annotations

import logging
import pathlib
import re
import time
from typing import Any

import pandas as pd

from llm_comparison.config import JudgeConfig

logger = logging.getLogger("llm_comparison.evaluation")

# ── Deterministic company-set metrics ────────────────────────────────────────

_KB_PATH = (
    pathlib.Path(__file__).resolve().parent.parent.parent
    / "kb"
    / "GNEM - Auto Landscape Lat Long Updated.xlsx"
)
_COMPANY_NAMES: frozenset[str] | None = None

# Strips common corporate suffixes so "Kia Georgia Inc." also matches "Kia Georgia".
_CORP_SUFFIX = re.compile(
    r",?\s+(?:inc|llc|ltd|corp|co\.|company|group|corporation|manufacturing|"
    r"industries|international|solutions|technologies?|systems?|services?|"
    r"associates?|partners?|ventures?|holdings?)\.?$",
    re.IGNORECASE,
)


def _normalize_company(name: str) -> str:
    """Strip common corporate suffixes for lenient matching."""
    return _CORP_SUFFIX.sub("", name).strip()


def _load_company_names() -> frozenset[str]:
    """Load the canonical company names from the GNEM KB Excel (once).

    Both the full name and the suffix-stripped form are added to the set so
    that "Kia Georgia" matches KB entry "Kia Georgia Inc."
    """
    global _COMPANY_NAMES
    if _COMPANY_NAMES is not None:
        return _COMPANY_NAMES
    try:
        df = pd.read_excel(_KB_PATH, usecols=["Company"])
        names_set: set[str] = set()
        for raw in df["Company"].dropna().tolist():
            name = str(raw).strip()
            if not name:
                continue
            name_lower = name.lower()
            names_set.add(name_lower)
            normalized = _normalize_company(name_lower)
            if normalized != name_lower and len(normalized) >= 3:
                names_set.add(normalized)
        _COMPANY_NAMES = frozenset(names_set)
        logger.info("Loaded %d company name variants from KB for deterministic metrics", len(_COMPANY_NAMES))
    except Exception as exc:
        logger.warning("Could not load company names from KB (%s) — set metrics will be empty", exc)
        _COMPANY_NAMES = frozenset()
    return _COMPANY_NAMES


def _extract_companies(text: str) -> set[str]:
    """Return the subset of known company names that appear in text."""
    names = _load_company_names()
    if not names or not text:
        return set()
    text_lower = text.lower()
    return {name for name in names if name in text_lower}


def _count_exact_match(golden: str, answer: str) -> int:
    """Return 1 if the expected count from golden appears (normalized) in answer.

    The golden answers often contain supporting numbers after the real count
    (for example "There are 100 areas... Top concentrations include 7 plants").
    This function first extracts the count-like number from the opening claim,
    then falls back to the first normalized number when no count phrase exists.
    """
    if not golden or not answer:
        return 0

    def _norm(num: str) -> str:
        return num.replace(",", "")

    opening = golden.strip().splitlines()[0]
    count_patterns = [
        r"\bthere\s+(?:are|is)\s+(?:only\s+)?(\d[\d,]*|\d)\b",
        r"\bthere\s+(?:are|is)\s+(?:only\s+)?(\d[\d,]*|\d)\s+[\w‑-]+",
        r"\bwith\s+a\s+total\s+of\s+(\d[\d,]*|\d)\b",
        r"\b(?:total|combined employment|employment size)\s+(?:of|:)?\s*(\d[\d,]*|\d)\b",
        r"\b(\d[\d,]*|\d)\s+(?:georgia|companies|company|suppliers|areas|counties|roles|facilities|plants)\b",
    ]

    candidates: list[str] = []
    for pattern in count_patterns:
        match = re.search(pattern, opening, flags=re.IGNORECASE)
        if match:
            candidates.append(_norm(match.group(1)))
            break

    if not candidates:
        raw_nums = re.findall(r"\b(\d[\d,]*\d|\d)\b", opening)
        if not raw_nums:
            raw_nums = re.findall(r"\b(\d[\d,]*\d|\d)\b", golden)
        if raw_nums:
            candidates.append(_norm(raw_nums[0]))

    if not candidates:
        return 0

    answer_normalized = re.sub(r"\b(\d[\d,]*\d|\d)\b", lambda m: m.group().replace(",", ""), answer.lower())
    for n in candidates:
        if re.search(rf"\b{re.escape(n)}\b", answer_normalized):
            return 1
    return 0


def _company_set_metrics(
    golden: str,
    context: str,
    answer: str,
) -> dict[str, Any]:
    gold = _extract_companies(golden)
    ctx = _extract_companies(context)
    ans = _extract_companies(answer)

    def recall(pred: set, ref: set) -> float:
        return round(len(pred & ref) / len(ref), 4) if ref else 0.0

    def precision(pred: set, ref: set) -> float:
        return round(len(pred & ref) / len(pred), 4) if pred else 0.0

    def f1(p: float, r: float) -> float:
        return round(2 * p * r / (p + r), 4) if (p + r) > 0 else 0.0

    ctx_recall = recall(ctx, gold)
    ans_recall = recall(ans, gold)
    ans_prec = precision(ans, gold)
    ans_f1 = f1(ans_prec, ans_recall)

    return {
        "gold_companies": ", ".join(sorted(gold)),
        "context_companies": ", ".join(sorted(ctx)),
        "answer_companies": ", ".join(sorted(ans)),
        "context_company_recall": ctx_recall,
        "answer_company_recall": ans_recall,
        "answer_company_precision": ans_prec,
        "answer_company_f1": ans_f1,
    }


WEIGHTS: dict[str, float] = {
    "faithfulness": 0.25,
    "answer_relevancy": 0.20,
    "context_precision": 0.20,
    "context_recall": 0.20,
    "answer_correctness": 0.15,
}

ALL_METRICS = list(WEIGHTS.keys())
NO_RAG_SKIP = {"faithfulness", "context_precision", "context_recall"}


def _split_contexts(raw: Any) -> list[str]:
    """Split a stored retrieved_context cell into a list[str] for ragas.

    Pandas returns NaN for empty cells; `str(NaN) == "nan"`, so guarding only
    against `None` is not enough. We must explicitly check pd.isna() and the
    literal "nan" sentinel before treating the value as content.
    """
    if raw is None:
        return []
    try:
        if pd.isna(raw):
            return []
    except (TypeError, ValueError):
        pass
    if isinstance(raw, list):
        return [str(x) for x in raw if str(x).strip()]
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        return []
    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    return chunks or [text]


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return not str(value).strip()


# ── Lazy imports so the module loads even without ragas/langchain_openai ──

def _build_judge_and_embeddings(cfg: JudgeConfig):
    from langchain_openai import ChatOpenAI

    try:
        from langchain_ollama import OllamaEmbeddings
    except ImportError:
        from langchain_community.embeddings import OllamaEmbeddings  # type: ignore

    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper

    chat = ChatOpenAI(
        model=cfg.judge_model,
        base_url=cfg.judge_base_url,
        api_key=cfg.judge_api_key,
        temperature=0.0,
        timeout=120,
    )
    emb = OllamaEmbeddings(model=cfg.ragas_embedding_model, base_url=cfg.ollama_base_url)
    return LangchainLLMWrapper(chat), LangchainEmbeddingsWrapper(emb)


def _build_metrics(judge_llm, judge_embeddings) -> dict[str, Any]:
    """Instantiate the official ragas metrics, keyed by canonical name."""
    from ragas.metrics import (
        AnswerCorrectness,
        AnswerSimilarity,
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    answer_similarity = AnswerSimilarity(embeddings=judge_embeddings)
    return {
        "faithfulness": Faithfulness(llm=judge_llm),
        "answer_relevancy": ResponseRelevancy(llm=judge_llm, embeddings=judge_embeddings),
        "context_precision": LLMContextPrecisionWithReference(llm=judge_llm),
        "context_recall": LLMContextRecall(llm=judge_llm),
        "answer_correctness": AnswerCorrectness(
            llm=judge_llm,
            embeddings=judge_embeddings,
            answer_similarity=answer_similarity,
        ),
    }


def _make_sample(question: str, answer: str, contexts: list[str], reference: str):
    from ragas.dataset_schema import SingleTurnSample

    return SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
        reference=reference,
    )


def _score_one(metric: Any, sample: Any) -> float:
    """Run a single ragas metric synchronously, returning a float in [0,1]."""
    if hasattr(metric, "single_turn_score"):
        return float(metric.single_turn_score(sample))
    # Fallback for async-only metrics
    import asyncio

    return float(asyncio.run(metric.single_turn_ascore(sample)))


def _final_score(scores: dict[str, float], skipped: set[str]) -> float:
    """Weighted average, renormalizing over the metrics that actually ran."""
    used = {m: w for m, w in WEIGHTS.items() if m not in skipped}
    total_weight = sum(used.values()) or 1.0
    return round(
        sum(scores[m] * (w / total_weight) for m, w in used.items()),
        4,
    )


def evaluate_rows(
    gen_df: pd.DataFrame,
    cfg: JudgeConfig,
) -> pd.DataFrame:
    """Score every generation row. Returns a per_row DataFrame."""
    judge_llm, judge_embeddings = _build_judge_and_embeddings(cfg)
    metrics = _build_metrics(judge_llm, judge_embeddings)

    out_rows: list[dict[str, Any]] = []
    total = len(gen_df)
    for i, row in enumerate(gen_df.to_dict(orient="records"), start=1):
        err = row.get("error", "")
        if not _is_blank(err):
            logger.warning(
                "Skipping row %d (model=%s mode=%s qid=%s): generation error=%s",
                i, row.get("model"), row.get("mode"), row.get("question_id"), err,
            )
            out_rows.append({**_blank_eval_row(row), "notes": f"skipped: {err}"})
            continue

        def _safe_str(value: Any) -> str:
            return "" if _is_blank(value) else str(value)

        question = _safe_str(row.get("question"))
        answer = _safe_str(row.get("answer"))
        reference = _safe_str(row.get("golden_answer"))
        contexts = _split_contexts(row.get("retrieved_context"))
        mode = _safe_str(row.get("mode"))

        scores: dict[str, float] = {m: 0.0 for m in ALL_METRICS}
        skipped: set[str] = set()
        notes_parts: list[str] = []

        # Only no_rag mode legitimately runs without contexts. A RAG mode
        # with zero contexts is a retrieval failure that must hurt the
        # score rather than be hidden by weight renormalisation.
        if mode == "no_rag":
            skipped |= NO_RAG_SKIP
            notes_parts.append("no_rag: no retrieved context")
        elif not contexts:
            notes_parts.append("retrieval failure: 0 contexts in RAG mode")

        sample = _make_sample(question, answer, contexts, reference)

        eval_start = time.monotonic()
        for metric_name in ALL_METRICS:
            if metric_name in skipped:
                continue
            if not contexts and metric_name in NO_RAG_SKIP:
                scores[metric_name] = 0.0
                continue
            try:
                scores[metric_name] = float(_score_one(metrics[metric_name], sample))
            except Exception as exc:
                logger.warning(
                    "Metric %s failed for row %d (model=%s mode=%s qid=%s): %s",
                    metric_name, i, row.get("model"), row.get("mode"), row.get("question_id"), exc,
                )
                scores[metric_name] = 0.0
                notes_parts.append(f"{metric_name}: {type(exc).__name__}")
        eval_elapsed = time.monotonic() - eval_start

        final = _final_score(scores, skipped)

        raw_context = _safe_str(row.get("retrieved_context"))
        det = _company_set_metrics(reference, raw_context, answer)
        det["count_exact_match"] = _count_exact_match(reference, answer)

        out_rows.append(
            {
                "run_id": row.get("run_id", ""),
                "question_id": row.get("question_id", ""),
                "category": row.get("category", ""),
                "question": question,
                "golden_answer": reference,
                "model": row.get("model", ""),
                "mode": mode,
                "answer": answer,
                "faithfulness": round(scores["faithfulness"], 4),
                "answer_relevancy": round(scores["answer_relevancy"], 4),
                "context_precision": round(scores["context_precision"], 4),
                "context_recall": round(scores["context_recall"], 4),
                "answer_correctness": round(scores["answer_correctness"], 4),
                "final_score": final,
                "judge_model": cfg.judge_model,
                "eval_elapsed_s": round(eval_elapsed, 3),
                "notes": "; ".join(notes_parts),
                **det,
            }
        )

        logger.info(
            "[%d/%d] scored model=%s mode=%s qid=%s final=%.3f (%.1fs)",
            i, total, row.get("model"), mode, row.get("question_id"), final, eval_elapsed,
        )

    return pd.DataFrame(out_rows)


def _blank_eval_row(gen_row: dict) -> dict[str, Any]:
    return {
        "run_id": gen_row.get("run_id", ""),
        "question_id": gen_row.get("question_id", ""),
        "category": gen_row.get("category", ""),
        "question": gen_row.get("question", ""),
        "golden_answer": gen_row.get("golden_answer", ""),
        "model": gen_row.get("model", ""),
        "mode": gen_row.get("mode", ""),
        "answer": gen_row.get("answer", ""),
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "context_precision": 0.0,
        "context_recall": 0.0,
        "answer_correctness": 0.0,
        "final_score": 0.0,
        "judge_model": "",
        "eval_elapsed_s": 0.0,
        "notes": "",
        "gold_companies": "",
        "context_companies": "",
        "answer_companies": "",
        "context_company_recall": 0.0,
        "answer_company_recall": 0.0,
        "answer_company_precision": 0.0,
        "answer_company_f1": 0.0,
        "count_exact_match": 0,
    }


def build_aggregations(per_row: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compute aggregation sheets for RAGAS metrics and deterministic set metrics."""
    metric_cols = ALL_METRICS + ["final_score"]
    det_cols = [
        "context_company_recall", "answer_company_recall",
        "answer_company_precision", "answer_company_f1", "count_exact_match",
    ]

    if per_row.empty:
        empty = pd.DataFrame(columns=["model", "mode", *metric_cols])
        return {
            "agg_model_mode": empty,
            "agg_model_mode_metric": pd.DataFrame(columns=["model", "mode", "metric", "mean", "std", "n"]),
            "agg_category_mode": pd.DataFrame(columns=["category", "mode", *metric_cols]),
            "agg_company_set": pd.DataFrame(columns=["model", "mode", *det_cols]),
        }

    agg_model_mode = (
        per_row.groupby(["model", "mode"])[metric_cols]
        .mean()
        .round(4)
        .reset_index()
    )

    long_rows: list[dict[str, Any]] = []
    for (model, mode), sub in per_row.groupby(["model", "mode"]):
        for metric in metric_cols:
            long_rows.append(
                {
                    "model": model,
                    "mode": mode,
                    "metric": metric,
                    "mean": round(sub[metric].mean(), 4),
                    "std": round(sub[metric].std(ddof=0), 4) if len(sub) > 0 else 0.0,
                    "n": int(len(sub)),
                }
            )
    agg_model_mode_metric = pd.DataFrame(long_rows)

    agg_category_mode = (
        per_row.groupby(["category", "mode"])[metric_cols]
        .mean()
        .round(4)
        .reset_index()
    )

    # Deterministic set metrics aggregated by model + mode.
    available_det = [c for c in det_cols if c in per_row.columns]
    if available_det:
        agg_company_set = (
            per_row.groupby(["model", "mode"])[available_det]
            .mean()
            .round(4)
            .reset_index()
        )
    else:
        agg_company_set = pd.DataFrame(columns=["model", "mode", *det_cols])

    return {
        "agg_model_mode": agg_model_mode,
        "agg_model_mode_metric": agg_model_mode_metric,
        "agg_category_mode": agg_category_mode,
        "agg_company_set": agg_company_set,
    }
