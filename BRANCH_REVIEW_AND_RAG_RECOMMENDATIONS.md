# Branch Review and RAG Recommendations

This review compares `new-Arch` and `sreeja-arch` for the Georgia EV Excel-based QA benchmark.

The benchmark is not open-world truth. The questions and human-validated golden answers are derived from the same Excel knowledge base, so the best system is the one that can retrieve, preserve, and answer from the Excel rows with high recall and precision. LLM judge metrics are useful as secondary signals, but structured set/count evaluation is more defensible for many of these questions.

## Executive Summary

`new-Arch` is stronger as a retrieval design for structured Excel QA. It treats the workbook as structured data, adds entity extraction and deterministic KB query planning, uses exact filters, supports broad list/count/top-N questions, and only uses semantic ranking when needed.

`sreeja-arch` is stronger as an experiment harness. It cleanly runs 2 models x 4 modes x 50 questions, writes `generations.xlsx`, supports generation-only/evaluation-only/resume, stores prompts, and writes an Excel RAGAS report.

The best direction is to keep the `sreeja-arch` experiment runner and evaluator, but replace or augment its RAG retrieval path with the KB-first structured retrieval ideas from `new-Arch`.

## Comparison Table

| Component | `new-Arch` method | `sreeja-arch` method | Better | Why | Risks / drawbacks |
|---|---|---|---|---|---|
| Chunking strategy | GNEM company records use multi-view chunks per row in `phase2_embedding/chunker.py`: master, role, product, OEM, classification, capability, location. Web docs use parent-child chunks, 256-token children and 800-token parents. | GNEM company records use 1 chunk per company row in current `phase2_embedding/chunker.py`; web docs still use parent-child chunking. | `new-Arch` | Multi-view chunks give structured Excel rows multiple semantic entry points without losing the full master row. | More points in Qdrant, more ingestion complexity, possible duplicates unless row IDs and source hashes are stable. |
| Embedding model | `OLLAMA_EMBED_MODEL`, default `nomic-embed-text`; collection name is model-specific, e.g. `georgia_ev_chunks__nomic-embed-text`. | `OLLAMA_EMBED_MODEL`, default `nomic-embed-text`; used for query embedding and RAGAS embedding. | Tie | Both use the same default embedding model. | Embeddings alone are weak for exact structured filters such as tier, county, role, OEM, employment thresholds, and list completeness. |
| Vector database | Qdrant, with dense and sparse vectors and payload indexes for structured fields in `phase2_embedding/vector_store.py`. | Qdrant via direct `qdrant-client` in `llm_comparison/retrieval.py`; dense retrieval only for the comparison pipeline. | `new-Arch` | It uses Qdrant as a structured retrieval store, not only a semantic store. | Requires careful payload schema/index maintenance. |
| Retrieval method | Load all company records from Qdrant, apply structured filters and deterministic plans, then semantic rank/rerank where needed. | Embed question, dense query Qdrant, rerank top candidates, pass top reranked chunks to prompt. | `new-Arch` | Excel QA needs exact row/set retrieval before LLM synthesis. | More code paths and rules to maintain. |
| Dense retrieval | Uses dense search primarily to rank or supplement structured candidates; semantic top-K is 120. | Dense-only Qdrant query; current CLI default `--top-k 70`. | `new-Arch` | Dense retrieval is useful, but should not be the first and only gate for structured list questions. | Dense ranking can still miss rare phrasing unless metadata and keyword logic catch it. |
| Sparse retrieval | Sparse/BM25 vector creation is present in vector store. | Sparse vector name is configured but deliberately bypassed in `llm_comparison/retrieval.py`. | `new-Arch` | Sparse matching helps exact product phrases, company names, county names, and acronyms. | Current sparse implementation is custom hash-based, so validate it against Qdrant schema. |
| Hybrid retrieval | Collection and vector store are designed for dense + sparse hybrid use. | Hybrid is intentionally bypassed for the comparison run. | `new-Arch` | Hybrid retrieval is better for structured Excel text where exact terms matter. | Hybrid scoring needs calibration and debug output. |
| Metadata filtering | Rich filters: source type, chunk type/view, tier, county, city, role, OEM, industry, facility, classification, affiliation, EV relevance, employment. | No metadata filters in the comparison retrieval path. Payload is returned but not used to filter. | `new-Arch` | Most benchmark questions contain explicit structured constraints. | Incorrect filters can drop valid rows if extraction is wrong. |
| Reranker usage | Cross-encoder reranker in `phase4_agent/reranker.py`; failures fall back to dense order. | Mandatory cross-encoder in `llm_comparison/retrieval.py`; load failure aborts. Current final context default is `--rerank-top-n 25`. | Mixed | `sreeja-arch` is stricter operationally; `new-Arch` reranks after better candidate selection. | Reranking cannot recover rows that dense retrieval never returned. It can also drop valid rows in exhaustive/list questions. |
| Top-K settings | Semantic top-K 120; default output limits 18, broad-list limit 30, semantic limit 12; rerank candidate cap defaults to 48 in `new-Arch` config. | Dense top-K 70, rerank top-N 25 in current CLI. | `new-Arch` | Top-K is intent-aware. Broad/list/aggregate questions need different treatment than narrow fact questions. | Broad limit 30 is still too small for questions with 39 or 100 expected areas. |
| Query rewriting / planning | Deterministic `KBQueryPlan` for aggregate, top employment, role list, product text, facility, area set difference, chemical infrastructure, R&D, wiring harness, etc. | No query rewriting or planning. The raw question is embedded. | `new-Arch` | Planning is the biggest advantage for Excel-grounded QA. | Rules must be tested against the 50 human questions and generalized carefully. |
| Prompt templates | Row-preserving answer synthesis prompt in `phase4_agent/streaming.py`: source of truth, list every row, copy company names exactly, compact format for large lists. Also has soft filter prompt for ambiguous filters. | Four mode prompts in `llm_comparison/prompts.py`: `rag_only`, `no_rag`, `rag_pretrained`, `rag_pretrained_web`, with source allow/forbid and tagging rules. | Mixed | `new-Arch` generation prompt is stronger for Excel rows. `sreeja-arch` prompts are better for controlled model comparison modes. | `sreeja-arch` mode 3/4 prompts can force irrelevant pretrained/web facts into an Excel-only benchmark. |
| Answer generation logic | RAG answer is generated after structured retrieval; context is formatted as a compact pipe table. | Every mode builds a prompt and calls Ollama `/api/generate` with temperature 0 and `num_ctx=8192`. | `new-Arch` for QA, `sreeja-arch` for experiments | Pipe tables preserve rows better. The four-mode runner is easier for experiments. | LLM can still omit rows unless prompt and context are complete. |
| Citation / context usage | Context table is the answer source of truth; not citation-heavy. | Stores retrieved context, web context, web sources, and full prompt per row in Excel. | `sreeja-arch` | Better auditability in generated workbooks. | Final answers do not cite individual retrieved rows. |
| Evaluation method | Custom RAGAS-like judge in `scripts/run_ragas_eval.py`, with generation-only/evaluate-only support and custom JSON judge prompts. | Official RAGAS metrics in `llm_comparison/ragas_runner.py`, writing Excel report. | `sreeja-arch` | Official RAGAS is more standard and easier to explain. | RAGAS alone is not enough for Excel set/count correctness. |
| RAGAS / custom judge | Custom Ollama judge for 5 named metrics. | Official RAGAS metrics: faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness. | `sreeja-arch` | More standard metric implementation. | `answer_correctness` needed explicit `AnswerSimilarity`; this has been patched in `ragas_runner.py`. |
| Scoring logic | Weighted custom scores: faithfulness 0.25, relevancy 0.20, context precision 0.20, context recall 0.20, correctness 0.15. | Same weights, final score renormalizes for `no_rag` when context metrics are skipped. | Tie | Comparable scoring design. | Weighted LLM scores can hide exact set misses. Add deterministic recall/precision/F1 for company sets. |
| Logging / debugging | Good retrieval logs and inspect mode; less standardized Excel debug output. | Generation workbook stores prompt, context, model, mode, top-K, reranker, embedding model, error. | `sreeja-arch` | Easier to audit every experiment row. | Current retrieved context lacks dense rank, rerank score, and structured field columns per hit. |
| Latency | More filtering/planning, but can reduce LLM context and avoid web/judge calls. | Dense + rerank + generation for every RAG mode; web mode adds Tavily; evaluation can be slow with local judge. | Depends | `new-Arch` can be faster for exact KB queries; `sreeja-arch` is more expensive but systematic. | Cross-encoder and RAGAS local judge are major latency drivers. |
| Maintainability | More retrieval intelligence but more moving parts. | Self-contained `llm_comparison` package, clean CLI, clean Excel IO. | `sreeja-arch` | Easier to run and explain as an experiment harness. | Retrieval simplicity is currently the source of low recall. |
| Suitability for Excel-sheet QA | High, because it treats Excel as structured data. | Medium, because dense-only retrieval is not reliable for exhaustive Excel questions. | `new-Arch` | The benchmark rewards complete row retrieval, exact filters, and set/count answers. | Needs better integration into the comparison runner. |

## Prompt Review

### Retrieval Prompts

`new-Arch` has a soft filter interpreter prompt in `phase4_agent/filter_interpreter.py`. It asks the model to choose extra structured filters using only values present in the current KB candidate set and to return strict JSON.

This is useful for ambiguous phrases such as "primary", "specific", "focused", "suitable", or "risk". It is stronger than raw dense retrieval because it turns ambiguity into candidate-set-aware filters. The risk is over-filtering: if the model chooses a filter that the question does not justify, it can remove valid rows.

Rewrite recommendation:

```text
Choose extra filters only when the question explicitly requires them or when every valid interpretation maps to the same KB value. If uncertain, return no extra filters. Never use a filter that would remove a row that still satisfies the literal question.
```

`sreeja-arch` has no retrieval prompt. It embeds the raw question directly. This is simple and stable, but it misses structured intent such as "all", "highest", "over 1000", "no Battery Cell/Pack", and "existing chemical manufacturing infrastructure".

Rewrite recommendation:

Add a query planner before retrieval that emits:

```text
intent: list | count | top_n | aggregate | exact_match | semantic
filters: tier, county, city, role, OEM, facility, industry, EV relevance, employment range, product keywords
exhaustive: true/false
```

For the Excel benchmark, prefer deterministic parsing over an LLM planner where possible.

### Reranking Prompts

Neither branch uses an LLM reranking prompt. Both use a cross-encoder model.

`new-Arch` reranks structured candidate rows after filtering. This is safer because the candidate set is already likely to contain valid rows.

`sreeja-arch` reranks dense top-K hits. This is risky because the reranker only sees the dense candidates. If the third correct company is not in the dense top-K, or if it is ranked below `rerank_top_n`, it disappears.

Rewrite recommendation:

For exhaustive/list questions, do not let reranking truncate the result set. Use reranking only for ordering after exact filtering, or pass all exact matches to the generator.

### Answer Generation Prompts

`new-Arch` uses a row-preserving prompt in `phase4_agent/streaming.py`. It explicitly says the retrieved table is the source of truth, list rows one by one, do not omit rows, and copy company names exactly. This is the stronger prompt for Excel QA.

Risk: the prompt has minor wording issues and can still be constrained by context length. It also asks the LLM to synthesize counts even when deterministic code should compute them.

Recommended rewrite:

```text
Use the table as the only source of truth. If the question asks for all/list/every/map/full network, include every row in the table. Preserve exact company names. For count, top-N, max/min, and grouped totals, use the values shown in the table and do not infer missing rows.
```

`sreeja-arch` `rag_only` prompt is clean about source restrictions, but it does not explicitly force row completeness for list questions. It can answer from partial context without saying retrieval was incomplete.

Recommended rewrite:

```text
If the question asks for all/list/every/map/full network/count, answer only if INTERNAL_CONTEXT contains the complete candidate set. Include every relevant row present in INTERNAL_CONTEXT. If the context appears incomplete for the requested set, say the retrieved context is incomplete instead of giving a partial answer.
```

`sreeja-arch` `rag_pretrained` and `rag_pretrained_web` prompts force the model to use additional sources. That satisfies the comparison-mode requirement, but it is risky for this benchmark because the golden answers are Excel-derived. These prompts can introduce irrelevant or non-Excel facts.

Recommended rewrite for benchmark runs:

```text
Use INTERNAL_CONTEXT as the source of truth for all Excel-specific companies, counts, roles, locations, and products. Use pretrained/web facts only for clearly labeled background that does not change the Excel-grounded answer.
```

If the experiment requires forced source mixing, keep the tags and treat those modes as stress tests, not as the best Excel-grounded QA design.

### Evaluator / Judge Prompts

`new-Arch` uses a custom judge prompt for five metrics and asks for JSON score/reasoning. It is understandable but not official RAGAS.

`sreeja-arch` uses official RAGAS metric prompts internally. This is stronger for standard reporting. However, official RAGAS still relies on an LLM for several metrics and cannot fully defend set completeness for structured Excel questions.

Recommended addition:

Add deterministic evaluation columns for structured questions:

- `gold_company_count`
- `answer_company_count`
- `company_set_recall`
- `company_set_precision`
- `company_set_f1`
- `context_company_recall`
- `count_exact_match`

Use RAGAS as a secondary score, not the only decision criterion.

### Fallback / No-Answer Prompts

`new-Arch` returns "No matching companies found." when no companies are retrieved.

`sreeja-arch` `rag_only` requires exactly "The retrieved context does not contain enough information to answer this question." when context is insufficient.

The `sreeja-arch` fallback is better for auditability because it separates "retrieval did not contain enough evidence" from "the KB has no matching companies." The system should only say "No matching companies found" when deterministic retrieval checked the full KB and found zero matches.

## Diagnosis: Why `sreeja-arch` Misses Companies

The failure pattern "golden answer has 3 companies, final answer gives 2, and the retrieved/reranked context also misses the third" is primarily a retrieval recall failure, not a generation failure.

Likely causes:

- Dense-only retrieval is the first gate. If the correct row does not appear in top-K, it is gone.
- Reranker cannot recover missing rows. It only reorders candidates already returned by dense search.
- `rerank_top_n` truncates the context. Even with current `25`, exhaustive questions can need more rows.
- There is no metadata filtering for exact fields such as county, tier, role, OEM, facility, classification, EV relevance, employment, or product keywords.
- The current `sreeja-arch` company chunk is one row-level text chunk. It does not have focused role/product/OEM/location views, so some questions embed poorly.
- Sparse/hybrid retrieval is bypassed, so exact phrase matches like "DC-to-DC", "powder coating", "chemical manufacturing", "R&D", or company names get no lexical boost.
- List/count/set questions are treated like semantic search questions. They need exhaustive retrieval over the KB.
- The generator only sees `format_context(hits)`. If the valid row is absent there, no prompt can reliably produce it without hallucinating.

## Concrete Fixes

1. Add a KB-first structured retriever to `sreeja-arch`.

   Keep `scripts/run_llm_comparison.py` and `llm_comparison` as the harness, but replace `retrieve_and_rerank()` for RAG modes with a two-stage path:

   - Stage A: deterministic Excel/Qdrant structured retrieval.
   - Stage B: dense/hybrid retrieval only for semantic fallback or ordering.

2. Port the useful `new-Arch` planning logic.

   Start with these files as references:

   - `phase4_agent/kb_query_planner.py`
   - `phase4_agent/vector_retriever.py`
   - `phase4_agent/entity_extractor.py`

   The first planner rules to port should cover the known 50 questions: role lists, tier lists, county/company exact filters, top employment, employment thresholds, OEM network, product text contains, R&D, chemical infrastructure, battery materials, wiring harnesses, area concentration, and area set differences.

3. Preserve structured metadata in retrieval output.

   Every hit should carry:

   - company name
   - row ID
   - tier
   - role
   - city
   - county
   - employment
   - primary OEMs
   - EV relevance
   - industry group
   - facility type
   - classification
   - supplier affiliation
   - product/service text
   - dense rank
   - dense score
   - rerank score
   - retrieval source: `structured`, `hybrid`, `dense`, or `fallback`

4. Use exact filters before vector ranking.

   Examples:

   - "Tier 1/2" should filter `tier == Tier 1/2`.
   - "Gwinnett County" should filter county exactly.
   - "Battery Cell or Battery Pack" should filter role in that role set.
   - "fewer than 200 employees" should filter employment `< 200`.
   - "over 1000 workers" should filter employment `> 1000`.

5. Do not truncate exhaustive answers.

   If a question asks "all", "every", "list", "map", "full supplier network", "how many areas", or a grouped aggregate, pass all matching rows or a deterministic aggregate table. Do not apply `rerank_top_n` as a hard cap.

6. Re-enable lexical or hybrid retrieval for semantic product phrases.

   Dense embeddings alone are weak for exact phrases. Use sparse/BM25 or direct substring search for product/service fields before reranking.

7. Add deterministic set evaluation.

   For every row, extract company names from:

   - golden answer
   - retrieved context
   - final answer

   Then compute:

   - context recall: golden companies found in retrieved context
   - answer recall: golden companies found in answer
   - answer precision: answer companies that are in the golden set
   - answer F1

   For count/aggregate questions, also compute exact numeric match where possible.

8. Keep the RAGAS fix.

   `answer_correctness` in RAGAS 0.4.3 requires `AnswerSimilarity` when using the default correctness weights. `sreeja-arch` now explicitly wires `AnswerSimilarity` into `AnswerCorrectness` in `llm_comparison/ragas_runner.py`.

## Recommended Target Architecture

Use this combined architecture:

1. `sreeja-arch` CLI and output design.
2. `new-Arch` KB-first retrieval and row-preserving prompt style.
3. Dense/hybrid retrieval only after exact structured filters or for genuinely semantic questions.
4. Reranker for ordering, not for deciding whether valid structured rows exist.
5. Official RAGAS report plus deterministic Excel-grounded set/count metrics.

This gives you a fair experimental harness and a retrieval design that matches the benchmark.

## Acceptance Criteria Status

| Item | Status |
|---|---|
| `sreeja-arch` branch exists | Satisfied. Current branch is `sreeja-arch`. |
| Separate worktree from `main` | Not verified from this checkout. Current checkout has `main`, `new-Arch`, and `sreeja-arch` branches. |
| Inspection notes written into README/report | This file contains the inspection notes. `README_sreeja_arch.md` should link to it. |
| Single command for all 50 x 2 models x 4 modes | Satisfied by `scripts.run_llm_comparison` without `--generation-only`. |
| Single command for selected questions | Satisfied by `--question-ids`. |
| Produces `generations.xlsx` | Satisfied. |
| Produces scored RAGAS Excel report | Satisfied when judge env vars are valid and RAGAS evaluation completes. |
| Five official RAGAS metrics | Satisfied after the `AnswerSimilarity` fix for `answer_correctness`. |
| Reranker mandatory in RAG modes | Satisfied in `sreeja-arch`; load failure aborts generation. |
| Mode 3/4 source mixing and tags | Satisfied in prompts, but risky for Excel-grounded accuracy. |
| No JSONL / HTML dashboard from new pipeline | Satisfied for `llm_comparison`; old output files still exist in repo history/working outputs. |
| `--resume` skips completed generation rows | Satisfied via `generations.xlsx`. |
| Secrets from env vars | Satisfied by `llm_comparison/config.py` and `.env.example`. |
| `.env.example` present | Satisfied. |
| Smoke test 2 question IDs x 2 models x 4 modes | Not rerun in this review. |

## Files Most Relevant To This Review

- `georgia_ev_intelligence/llm_comparison/retrieval.py`
- `georgia_ev_intelligence/llm_comparison/prompts.py`
- `georgia_ev_intelligence/llm_comparison/modes.py`
- `georgia_ev_intelligence/llm_comparison/ragas_runner.py`
- `georgia_ev_intelligence/scripts/run_llm_comparison.py`
- `georgia_ev_intelligence/scripts/run_llm_evaluation.py`
- `README_sreeja_arch.md`
- `.env.example`
- `new-Arch:georgia_ev_intelligence/phase4_agent/kb_query_planner.py`
- `new-Arch:georgia_ev_intelligence/phase4_agent/vector_retriever.py`
- `new-Arch:georgia_ev_intelligence/phase4_agent/streaming.py`
- `new-Arch:georgia_ev_intelligence/phase2_embedding/chunker.py`
