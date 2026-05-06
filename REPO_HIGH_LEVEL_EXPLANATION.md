# Georgia EV Intelligence Repo - High-Level Explanation

## 1. What This Project Does

This repository is a Georgia electric-vehicle supply-chain intelligence system.
It answers questions about companies, suppliers, roles, counties, products,
employment, OEM relationships, and EV relevance using a curated Georgia EV
knowledge base.

At a high level, the project does three things:

1. Loads and structures Georgia EV company data from Excel/KB sources.
2. Converts that data into searchable vector chunks in Qdrant.
3. Runs LLM-based question answering and evaluation pipelines.

The current `sreeja-arch` branch is mainly focused on **LLM comparison and RAGAS
evaluation**. It runs the same set of validated questions through multiple LLMs
and multiple prompting/retrieval modes, stores generated answers in Excel, and
then evaluates them later.

---

## 2. Big Picture Architecture

The system can be explained as this pipeline:

```text
Georgia EV KB / Human-validated questions
        |
        v
Data loading and cleaning
        |
        v
Chunking and embedding
        |
        v
Qdrant vector database
        |
        v
Retrieval + reranking
        |
        v
LLM answer generation
        |
        v
Excel output
        |
        v
RAGAS evaluation
```

In simple terms:

- The KB is the source of truth.
- Qdrant is the searchable memory.
- Ollama models generate answers locally.
- RAGAS scores the answers against human-validated answers.
- Excel files are used as the main output format.

---

## 3. Main Branch Purpose: `sreeja-arch`

The `sreeja-arch` branch adds a self-contained comparison pipeline under:

```text
georgia_ev_intelligence/llm_comparison/
```

This branch is not primarily a chatbot branch. It is an experiment runner.

Its goal is to compare:

- different generation models
- different RAG modes
- different retrieval/prompting strategies
- final answer quality through RAGAS metrics

The main script is:

```text
georgia_ev_intelligence/scripts/run_llm_comparison.py
```

It can:

- generate answers only
- evaluate existing answers only
- do both generation and evaluation in one run
- resume from existing Excel output

---

## 4. Important Folders

### `georgia_ev_intelligence/llm_comparison/`

This is the core of the `sreeja-arch` branch.

It contains the new comparison/evaluation pipeline.

Key files:

```text
config.py
```

Loads only the environment variables needed for the comparison pipeline.
This avoids requiring unrelated old-system variables such as Postgres, Neo4j,
or Backblaze B2 when running the comparison.

```text
retrieval.py
```

Handles Qdrant retrieval:

- embeds the question using Ollama
- searches Qdrant with dense vector search
- reranks results using a cross-encoder
- returns the top context chunks

```text
modes.py
```

Runs each generation mode. The supported modes are:

```text
no_rag
rag_only
rag_pretrained
rag_pretrained_web
```

```text
prompts.py
```

Builds different prompts depending on the selected mode.

```text
excel_io.py
```

Reads and writes Excel files safely. This is how generation results are stored.

```text
ragas_runner.py
```

Runs RAGAS metrics on generated answers.

---

### `georgia_ev_intelligence/phase1_extraction/`

This folder handles KB loading and data preparation.

Important file:

```text
kb_loader.py
```

It reads the Georgia EV Excel knowledge base, normalizes columns, parses company
fields, and creates company dictionaries used by later stages.

Example fields:

- company name
- tier
- EV supply chain role
- primary OEMs
- EV/battery relevance
- industry group
- facility type
- city/county/state
- employment
- products/services
- classification method
- supplier affiliation type

---

### `georgia_ev_intelligence/phase2_embedding/`

This folder handles chunking, embeddings, and Qdrant upload.

Important files:

```text
chunker.py
```

Converts company records and documents into chunks.

```text
embedder.py
```

Calls Ollama to create vector embeddings, usually with:

```text
nomic-embed-text
```

```text
vector_store.py
```

Uploads embedded chunks to Qdrant and supports vector search.

```text
pipeline.py
```

Original Phase 2 embedding pipeline. It can embed companies/documents, although
some older paths may require older environment variables such as Postgres.

---

### `georgia_ev_intelligence/phase4_agent/`

This is the older Phase 4 agent architecture.

It contains older logic for:

- entity extraction
- SQL/Cypher-style retrieval
- structured filtering
- answer generation
- streaming responses

In the current `sreeja-arch` comparison pipeline, this old `EVAgent` path is
mostly not used. The new branch uses `llm_comparison/` instead.

---

### `georgia_ev_intelligence/scripts/`

This folder contains runnable scripts.

Most important scripts in this branch:

```text
run_llm_comparison.py
```

Main generation/comparison runner.

```text
run_llm_evaluation.py
```

Evaluation-only runner. It reads a `generations.xlsx` file and creates a RAGAS
report.

---

### `georgia_ev_intelligence/outputs/`

This is where generated artifacts are stored.

Important output folders:

```text
outputs/llm_comparison/<run_id>/generations.xlsx
```

Stores generated answers.

```text
outputs/ragas_reports/<run_id>.xlsx
```

Stores RAGAS evaluation results.

---

## 5. Chunking: Detailed Explanation

Chunking is the process of converting large or structured source data into
smaller pieces that can be embedded and searched.

The project uses chunking because LLMs and vector databases work better when the
source data is split into meaningful searchable units.

There are two main chunking styles in this repo.

---

### 5.1 Company KB Chunking

The Georgia EV KB is mostly structured company data. Each row represents a
company or company-location entry.

For company data, the chunker builds structured text like:

```text
Company: WIKA USA |
Tier: OEM Supply Chain |
Industry: Automotive wiring systems |
Location: Lawrenceville | Gwinnett County | Georgia |
Facility Type: Manufacturing / Engineering |
EV Role: HV and LV wiring harnesses for EVs and ICE vehicles |
OEMs: Multiple OEMs |
EV / Battery Relevant: Yes |
Employment: 500 |
Products: Vehicle power and data solutions, wiring harnesses, connectors |
Classification: Supplier |
Affiliation: Automotive supply chain participant
```

This structured text becomes the searchable representation of that company.

Then the system:

1. Sends the chunk text to Ollama embedding model.
2. Gets a dense vector back.
3. Uploads the vector and metadata to Qdrant.

Each company chunk also carries metadata such as:

```text
company_name
tier
ev_supply_chain_role
primary_oems
ev_battery_relevant
industry_group
facility_type
location_city
location_county
employment
products_services
classification_method
supplier_affiliation_type
```

This metadata is important because it allows filtering and interpretation later.

---

### 5.2 Document Chunking

The repo also supports web/document chunking.

For long documents, the project uses a parent-child chunking strategy:

```text
Long document
     |
     v
Parent chunks, about 800 tokens each
     |
     v
Child chunks, about 256 tokens each
```

The idea is:

- child chunks are small and precise for search
- parent chunks preserve surrounding context for answer generation

So the system searches small child chunks but can return larger parent text to
the model.

This is useful for web pages, PDFs, press releases, or long company documents.

---

### 5.3 Why Chunking Matters

Good chunking directly affects answer quality.

If chunks are too large:

- vector search becomes noisy
- irrelevant facts get retrieved
- LLM may hallucinate or miss the answer

If chunks are too small:

- important context may be missing
- the LLM may get partial facts
- answer correctness drops

For this project, company rows are already structured, so company-level chunks
are usually better than arbitrary text splitting.

---

## 6. Retrieval Flow in `sreeja-arch`

For RAG modes, retrieval works like this:

```text
Question
   |
   v
Ollama embedding model
   |
   v
Qdrant dense vector search
   |
   v
Cross-encoder reranker
   |
   v
Top context chunks
   |
   v
Prompt sent to generation model
```

The important point is that `sreeja-arch` uses **dense vector retrieval plus
reranking**.

It does not use the older branch's deterministic KB planner or structured
query logic in the main comparison path.

That means it is good for comparing model behavior, but exact spreadsheet-style
questions may still be harder.

---

## 7. Generation Modes

The comparison runner supports four modes.

### `no_rag`

The LLM answers using only its own pretrained knowledge.

No internal KB context is provided.

### `rag_only`

The LLM receives retrieved Qdrant context from the internal KB.

This is the standard RAG mode.

### `rag_pretrained`

The LLM receives internal context, but the prompt also allows the model to use
its pretrained reasoning.

### `rag_pretrained_web`

The LLM receives:

- internal Qdrant context
- web context from Tavily
- its pretrained reasoning

This mode requires:

```text
TAVILY_API_KEY
```

---

## 8. Evaluation Flow

After generation, the repo evaluates answers with RAGAS.

The evaluation compares:

- question
- generated answer
- retrieved context
- golden human-validated answer

Metrics include:

```text
faithfulness
answer_relevancy
context_precision
context_recall
answer_correctness
```

The final score is a weighted combination:

```text
faithfulness        0.25
answer_relevancy    0.20
context_precision   0.20
context_recall      0.20
answer_correctness  0.15
```

For `no_rag`, context-related metrics are skipped or treated specially because
there is no retrieved context.

---

## 9. Typical Commands

Generate answers only:

```bash
PYTHONPATH=georgia_ev_intelligence python -m scripts.run_llm_comparison \
  --run-id full_$(date +%Y%m%d) \
  --models qwen2.5:14b gemma3:27b \
  --modes rag_only no_rag rag_pretrained rag_pretrained_web \
  --generation-only
```

Evaluate later:

```bash
PYTHONPATH=georgia_ev_intelligence python -m scripts.run_llm_comparison \
  --run-id full_20260505 \
  --evaluation-only \
  --judge-model mistral-small3.2:24b \
  --judge-base-url http://localhost:11434/v1
```

Run a small smoke test:

```bash
PYTHONPATH=georgia_ev_intelligence python -m scripts.run_llm_comparison \
  --run-id smoke_test \
  --models qwen2.5:14b \
  --modes rag_only \
  --question-ids 1 \
  --generation-only
```

---

## 10. Required External Services

### Ollama

Used for:

- local generation models
- local embedding model
- optionally local judge model

Common models:

```text
qwen2.5:14b
gemma3:27b
nomic-embed-text
mistral-small3.2:24b
```

### Qdrant

Used as the vector database.

Important environment variables:

```env
QDRANT_URL=
QDRANT_API_KEY=
QDRANT_COLLECTION_NAME=
QDRANT_DENSE_VECTOR_NAME=dense
QDRANT_SPARSE_VECTOR_NAME=sparse
```

### Tavily

Used only for `rag_pretrained_web`.

```env
TAVILY_API_KEY=
```

### Judge Model

Used for RAGAS evaluation.

Can be Kimi/Moonshot or a local Ollama OpenAI-compatible endpoint.

For local Ollama judge:

```env
JUDGE_BASE_URL=http://localhost:11434/v1
JUDGE_API_KEY=ollama
JUDGE_MODEL=mistral-small3.2:24b
```

---

## 11. How To Explain This Repo in One Minute

This repo is a Georgia EV supply-chain RAG evaluation system. It takes a curated
Excel knowledge base of Georgia EV companies, chunks and embeds the company
records, stores them in Qdrant, retrieves relevant chunks for each question, and
uses local LLMs through Ollama to generate answers. The `sreeja-arch` branch is
mainly designed to compare different models and RAG modes across 50 validated
questions. It saves generated answers to Excel and then uses RAGAS to score the
answers against human-validated golden answers.

The most important idea is that the project separates:

```text
data preparation -> retrieval -> generation -> evaluation
```

This makes it easier to test whether errors come from the KB, chunking,
retrieval, prompting, generation model, or evaluation judge.

---

## 12. Key Takeaway

The repo is not just an LLM chatbot. It is an evaluation framework for testing
how well different LLM + retrieval setups answer Georgia EV supply-chain
questions.

The current branch is especially useful for:

- comparing models
- comparing RAG modes
- producing Excel reports
- running RAGAS evaluation
- diagnosing whether retrieval or generation is responsible for low scores

