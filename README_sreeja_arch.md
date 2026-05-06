# sreeja-arch — LLM Comparison + Official RAGAS Evaluation

This document covers only the **sreeja-arch** branch's additions. The
original system documentation lives in `georgia_ev_intelligence/README.md`.

This branch adds a self-contained pipeline that:

1. Generates 50 golden questions × 2 generation models × 4 modes = 400 answers.
2. Evaluates every generated answer with the **official `ragas` package**,
   judged by **Kimi 2.6** (OpenAI-compatible API).
3. Produces Excel-only reports — no JSONL, no HTML dashboard.
4. Resumes from existing Excel state.

---

## Codebase inspection — reuse / extend / replace decisions

| Existing module | Decision | Notes |
|---|---|---|
| `phase2_embedding/embedder.py` | **Not imported.** | The new `llm_comparison/retrieval.py` calls Ollama `/api/embed` directly to avoid the `shared.config.Config` singleton (which mandates Neo4j / Postgres / B2 vars unrelated to this work). |
| `phase2_embedding/vector_store.py` | **Not imported.** | The new code uses `qdrant-client` directly with the `QDRANT_*` env vars only. |
| `phase4_agent/pipeline.py` | **Not imported.** | The Phase 4 `EVAgent` is tightly coupled to entity extraction and a different prompting style; it cannot be parameterised cleanly into the four required modes. |
| `evaluate/format_runner.py` | **Extended / reused.** | Removed its module-level `Config.get()` call and reused its Tavily helper from `llm_comparison`, while keeping legacy callers compatible. |
| `scripts/run_format_eval.py` | Reference only. | Used as a structural template for the per-question loop. |
| `scripts/run_ragas_eval.py` | **Replaced.** | Was a custom Ollama judge with weighted metrics. New evaluator (`scripts/run_llm_evaluation.py`) uses the official `ragas` package + Kimi 2.6 judge. |
| `scripts/generate_dashboard.py` | **Not called.** | HTML dashboard is out of scope; the file is left untouched but unwired. |
| Cross-encoder rerank | **Implemented as mandatory.** | `llm_comparison/retrieval.py` force-loads `cross-encoder/ms-marco-MiniLM-L12-v2` and hard-fails on load error. No silent fallback. |
| Qdrant collection | **Reuse existing data.** | Pipeline targets `georgia_ev_chunks` by default; override with `QDRANT_COLLECTION_NAME` only if the collection was reingested under a different name. |
| Retrieval | **Dense-only + rerank.** | Hybrid (dense+sparse RRF) is intentionally bypassed for the comparison runs, per the literal reading of the spec. |

---

## What's new

```
georgia_ev_intelligence/llm_comparison/
  __init__.py
  config.py          # env-only loader; no shared.config.Config dependency
  prompts.py         # 4 mode prompt builders with explicit allow/forbid + tagging
  retrieval.py       # dense Qdrant + mandatory cross-encoder rerank
  modes.py           # run_mode(mode, question, model, ...)
  excel_io.py        # atomic xlsx read/write + resume helpers
  ragas_runner.py    # official ragas eval wired to Kimi 2.6 judge

georgia_ev_intelligence/scripts/
  run_llm_comparison.py   # generation + optional evaluation CLI
  run_llm_evaluation.py   # evaluation CLI

.env.example              # env template (root)
README_sreeja_arch.md     # this file
```

`georgia_ev_intelligence/requirements.txt` adds:
- `ragas>=0.2.0`
- `langchain-openai>=0.2.0`
- `sentence-transformers>=3.0.0`

---

## Setup

1. **Create / enter the worktree.**

   ```bash
   git fetch --all
   git worktree add ../Georgia_ev_intelligence-sreeja-arch -b sreeja-arch main
   cd ../Georgia_ev_intelligence-sreeja-arch
   ```

2. **Install deps in a fresh venv.**

   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r georgia_ev_intelligence/requirements.txt
   ```

3. **Pull required Ollama models (once).**

   ```bash
   ollama pull qwen2.5:14b
   ollama pull gemma3:27b
   ollama pull nomic-embed-text
   ```

4. **Configure `.env`.** Copy the template and fill values:

   ```bash
   cp .env.example .env
   ```

5. **Smoke-test the reranker** (catches a bad install before you spend on LLM calls):

   ```bash
   python -c "from llm_comparison.retrieval import ensure_reranker; \
              from llm_comparison.config import load_generation_config; \
              ensure_reranker(load_generation_config(require_qdrant=False).reranker_model); \
              print('reranker OK')"
   ```

---

## Environment variables

| Variable | Purpose | Where to get it |
|---|---|---|
| `OLLAMA_BASE_URL` | Ollama server URL. Default `http://localhost:11434`. | Local Ollama install. |
| `OLLAMA_EMBED_MODEL` | Embedding model for queries + ragas. Default `nomic-embed-text`. | `ollama pull nomic-embed-text`. |
| `QDRANT_URL` | Qdrant Cloud / self-hosted URL. | Qdrant dashboard → cluster → endpoint. |
| `QDRANT_API_KEY` | Qdrant API key. | Qdrant dashboard → API keys. |
| `QDRANT_COLLECTION_NAME` | Optional collection override. Default `georgia_ev_chunks`. | Set only if your Qdrant collection uses a different name. |
| `QDRANT_DENSE_VECTOR_NAME` | Dense vector field name. Default `dense`. | Collection schema. |
| `QDRANT_SPARSE_VECTOR_NAME` | Sparse vector field name. Default `sparse`. Unused in the dense-only path. | Collection schema. |
| `TAVILY_API_KEY` | Required for mode `rag_pretrained_web`. | tavily.com → dashboard → API keys. |
| `JUDGE_BASE_URL` | OpenAI-compatible API host for Kimi 2.6 (e.g. `https://api.moonshot.ai/v1`). | Moonshot dashboard. |
| `JUDGE_API_KEY` | Kimi API key. | Moonshot dashboard. Never log this value. |
| `JUDGE_MODEL` | Kimi model id (e.g. `kimi-k2-0905-preview`). | Moonshot dashboard. |
| `CROSS_ENCODER_RERANKER_MODEL` | Override the reranker. Default `cross-encoder/ms-marco-MiniLM-L12-v2`. | Hugging Face. |

---

## How to run

`run_llm_comparison.py` runs generation and evaluation end-to-end by default.
Use `--generation-only` or `--evaluation-only` when you want to split the stages
or resume one stage independently.

### Command 1 — full run, all 50 questions × 2 models × 4 modes

```bash
PYTHONPATH=georgia_ev_intelligence python -m scripts.run_llm_comparison \
  --run-id full_$(date +%Y%m%d) \
  --models qwen2.5:14b gemma3:27b \
  --modes rag_only no_rag rag_pretrained rag_pretrained_web \
  --judge-model "$JUDGE_MODEL"
```

Outputs:
- `georgia_ev_intelligence/outputs/llm_comparison/<run_id>/generations.xlsx`
- `georgia_ev_intelligence/outputs/ragas_reports/<run_id>.xlsx`

### Command 2 — specific question numbers

Replace `5 7 10 22` with the `Num` values you want from
`kb/Human validated 50 questions.xlsx`:

```bash
PYTHONPATH=georgia_ev_intelligence python -m scripts.run_llm_comparison \
  --run-id subset_$(date +%Y%m%d) \
  --models qwen2.5:14b gemma3:27b \
  --modes rag_only no_rag rag_pretrained rag_pretrained_web \
  --question-ids 5 7 10 22 \
  --judge-model "$JUDGE_MODEL"
```

To evaluate an existing generation workbook without regenerating:

```bash
PYTHONPATH=georgia_ev_intelligence python -m scripts.run_llm_comparison \
  --run-id <existing_run_id> \
  --evaluation-only \
  --judge-model "$JUDGE_MODEL"
```

### Resume

Both scripts accept `--resume`. Generation resumes by reading the existing
`generations.xlsx` and skipping `(model, mode, question_id)` triples without
errors. Evaluation resumes by reading the existing `<run_id>.xlsx` `per_row`
sheet.

---

## Output schema

`generations.xlsx` (sheet `generations`) — one row per `(model, mode, question_id)` triple:

`run_id`, `question_id`, `category`, `question`, `golden_answer`, `model`,
`mode`, `answer`, `retrieved_context`, `web_context`, `web_sources`,
`top_k`, `retrieved_count`, `rerank_top_n`, `generation_elapsed_s`, `embedding_model`,
`reranker_model`, `tavily_used`, `temperature`, `prompt_used`,
`timestamp_utc`, `error`.

`<run_id>.xlsx`:
- `per_row` — every metric + `final_score` + `notes` per row.
- `agg_model_mode` — averages by `(model, mode)`.
- `agg_model_mode_metric` — long format with mean / std / n.
- `agg_category_mode` — averages by `(category, mode)`.
- `run_metadata` — single row with run_id, models, modes, embedding model,
  reranker model, judge model/base URL, top_k, rerank_top_n, temperature, total
  rows/questions, and timestamp.

`final_score` weights:
```
faithfulness        0.25
answer_relevancy    0.20
context_precision   0.20
context_recall      0.20
answer_correctness  0.15
```

For `no_rag` rows, faithfulness / context_precision / context_recall are
recorded as `0.0` with `notes="no_rag: no retrieved context"`, and the
remaining two metrics are renormalised so their weights sum to `1.0`.
