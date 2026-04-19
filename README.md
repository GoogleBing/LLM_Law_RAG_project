# Vietnamese Legal RAG Chatbox

A Retrieval-Augmented Generation (RAG) system for querying Vietnamese legal documents (~16,400 documents). Combines hybrid retrieval (FAISS + BM25 + cross-encoder reranking), freshness-aware scoring, parent-child chunking, and post-hoc citation verification.

---

## System Architecture

```
Raw .txt documents (export_1/)
        │
        │  parser.py
        ▼
parsed_documents.jsonl ──► doc_meta.pkl (status, dates, replaced_by)
        │
        │  build_index.py  [run once]
        ▼
index/
  ├── faiss.index        ← dense vector index (children)
  ├── chunks.jsonl       ← child chunk metadata
  ├── parents.jsonl      ← full Điều text (swapped in for LLM)
  └── bm25_corpus.pkl    ← tokenized corpus for BM25

────────────────────────────────────────  query time

HybridRetriever.retrieve(query)
  1. analyze_query        — intent / explicit so_hieu / from_date
  2. FAISS  top-30        — vector search on children
  3. BM25   top-30        — lexical search on children
  4. RRF fusion → top-40  — combine rankings
  5. bge-reranker-v2-m3  — cross-encoder rerank
  6. × freshness_weight  — penalise expired/outdated docs
  7. + 0.5 explicit boost — rescue docs named in query
  8. dedupe by parent_id → swap child → parent text
  9. per-doc cap + enrich (status, successor so_hieu)
        │
        ▼
RAGPipeline.answer(query, contexts)
  ├── build_prompt  — SYSTEM_INSTRUCTION + context + query
  ├── llm()         — Qwen2.5-14B / Gemini / any HF model
  └── verify_citations — ok / partial / hallucinated
        │
        ▼
     Answer + citation report
```

---

## Project Layout

```
RAG Chatbox/
├── export_1/                  raw legal .txt documents
├── parsed_documents.jsonl     parsed corpus (input to build_index.py)
├── index/                     built FAISS + BM25 index (generated)
├── doc_meta.pkl               metadata cache (generated)
├── RES.csv                    evaluation dataset (questions + reference answers)
├── requirements.txt
│
└── src/
    ├── config.py              centralized paths & hyper-params (env-var overrides)
    ├── parser.py              .txt → structured JSON parser
    ├── build_index.py         build FAISS + BM25 + parents (run once)
    ├── chat.py                interactive multi-turn chatbot
    ├── demo.py                one-shot demo (retrieval + optional LLM)
    ├── evaluate.py            retrieval / ablation / full-answer metrics
    ├── evaluate_report.py     baseline vs improved comparison report
    ├── rag_pipeline.py        backward-compat re-export facade
    │
    ├── indexing/
    │   ├── chunker.py         parent-child chunker (Điều → children)
    │   └── doc_metadata.py    sidecar: status, dates, replaced_by map
    │
    ├── retrieval/
    │   ├── query_analyzer.py  regex intent extraction (no LLM)
    │   ├── freshness.py       tinh_trang → scoring weight
    │   └── retriever.py       HybridRetriever (full pipeline)
    │
    └── generation/
        ├── prompt_builder.py  SYSTEM_INSTRUCTION + build_prompt
        ├── llm_providers.py   make_llm() — Gemini / vLLM / HuggingFace
        ├── citation.py        post-hoc citation verification
        └── pipeline.py        RAGPipeline (retriever + prompt + LLM + verify)
```

---

## Prerequisites

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.11 |
| RAM | 16 GB | 32 GB |
| GPU VRAM | — | 24 GB (for reranker + LLM) |
| Disk | 10 GB free | 20 GB free |

GPU is strongly recommended for the embedding and reranking steps. CPU works but is significantly slower (~10× for build_index.py).

---

## Step-by-Step Setup (from scratch)

### Step 0 — Create a virtual environment

```bash
# Using conda (recommended)
conda create -n rag python=3.11 -y
conda activate rag

# Or using venv
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate
```

### Step 0.1 — Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users (CUDA):** open `requirements.txt` and change `faiss-cpu` to `faiss-gpu` before running the command above.

Expected output: all packages install without error. The heaviest downloads are `torch` (~2 GB) and the model weights (downloaded automatically on first use).

### Step 0.2 — Configure LLM credentials

**Option A — Gemini API** (simplest, no local GPU needed for LLM):

Create a file named `.env` in the project root (same folder as `requirements.txt`):
```
GEMINI_API_KEY=your_key_here
```

**Option B — vLLM server** (used in this project with Qwen2.5-14B-Instruct):

```bash
# Linux / macOS
export VLLM_BASE_URL=http://your-server:8000/v1
export VLLM_API_KEY=EMPTY

# Windows (PowerShell)
$env:VLLM_BASE_URL = "http://your-server:8000/v1"
$env:VLLM_API_KEY  = "EMPTY"
```

Or add these lines to the `.env` file instead of exporting them.

**Option C — Local HuggingFace model:** no setup needed; pass `--quant 4bit` when running to reduce VRAM usage.

---

### Step 1 — Parse raw documents

Reads all `.txt` files from `export_1/` and converts them to structured JSON, writing `parsed_documents.jsonl`.

```bash
python src/check_parse.py
```

**Expected output:**
```
Processing 16400 files -> parsed_documents.jsonl
CSV report   -> parsed_documents_report.csv

============================================================
  PARSE SUMMARY  (16400/16400 files written, 0 errors)
============================================================

Field                  Empty   Filled  Empty %
---------------------- ------- ------- --------
  so_hieu              ...     ...     ...%
  loai_van_ban         ...     ...     ...%
  tinh_trang           ...     ...     ...%
  ...

  JSONL : parsed_documents.jsonl
  CSV   : parsed_documents_report.csv
```

**Output files:**
- `parsed_documents.jsonl` — one JSON object per line, one document per line
- `parsed_documents_report.csv` — field coverage table (useful for checking parse quality)

**Estimated time:** 2–5 minutes for ~16,400 documents.

> **Quick test on a subset (optional):**
> ```bash
> python src/check_parse.py --limit 200
> ```
> Run this first to confirm the parser works before processing the full corpus.

---

### Step 2 — Build the index

Reads `parsed_documents.jsonl`, chunks each document into parent/child segments, embeds all children with the Vietnamese bi-encoder, and saves FAISS + BM25 indices.

```bash
python src/build_index.py
```

**Expected output:**
```
Loading embedding model …
  Model: bkai-foundation-models/vietnamese-bi-encoder  dim=768  device=cuda

Collecting chunks …
  Children (indexed): 667085
  Parents  (context): 123671

Saving chunk metadata → index/chunks.jsonl
Saving parent metadata → index/parents.jsonl

Building BM25 corpus …
  Saved → index/bm25_corpus.pkl

Embedding chunks (this may take 10-30 min depending on GPU) …
  Embedded 667085 chunks in 28.3 min
  Saved FAISS index → index/faiss.index

Done. Index ready.
```

**Output files (in `index/`):**
- `faiss.index` — dense vector index over child chunks
- `chunks.jsonl` — metadata for every child chunk
- `parents.jsonl` — full Điều text, swapped in at retrieval time for the LLM
- `bm25_corpus.pkl` — tokenized corpus for BM25 lexical search

**Estimated time:** 25–40 minutes on GPU, 3–6 hours on CPU only.

> This step only needs to run **once**. If you add new documents later, re-run this step from scratch.

---

### Step 3 — Verify the index (retrieval only, no LLM)

Run a single question through the retrieval pipeline to confirm the index loaded correctly:

```bash
python src/demo.py "Quy định về lãi suất tiền gửi hiện hành?"
```

**Expected output:**
```
=== QUERY ===
  Quy định về lãi suất tiền gửi hiện hành?

=== QUERY ANALYSIS ===
  prefers_current : True
  explicit_so_hieu: (none)
  from_date       : (none)

Loading index …
  parents: 123671 (parent-child mode)
Loading embedding model: bkai-foundation-models/vietnamese-bi-encoder …
Loading document metadata sidecar …
  metadata: 16400 docs
Retriever ready.

=== RETRIEVED CONTEXTS (top 6) ===

[1] 20/2024/TT-NHNN   status=active
    score=0.872   rerank=0.891   freshness×=1.00
    Tiêu đề: Thông tư quy định về lãi suất ...
    ...

(no --llm supplied; retrieval-only mode)
```

If you see 6 retrieved chunks, the index is working correctly.

---

### Step 4 — Run the interactive chatbot

```bash
# Option A: vLLM server
python src/chat.py --llm vllm:Qwen2.5-14B-Instruct

# Option B: Gemini API
python src/chat.py --llm gemini-2.5-flash

# Option C: local HuggingFace model (4-bit quantised, needs ~7 GB VRAM)
python src/chat.py --llm Qwen/Qwen2.5-7B-Instruct --quant 4bit
```

**Expected startup output:**
```
========================================================================
  RAG LAW CHATBOT
  LLM : vllm:Qwen2.5-14B-Instruct
  Top-k: 6
  Commands: /quit  /retrieval  /clear
========================================================================

Loading retriever …
Loading LLM: vllm:Qwen2.5-14B-Instruct …

Sẵn sàng. Hãy đặt câu hỏi.

Bạn: _
```

**Chat commands:**

| Command | Action |
|---|---|
| `/retrieval` | Toggle showing retrieved document chunks before each answer |
| `/clear` | Clear the screen |
| `/quit` or `/exit` | Exit the chatbot |

---

## Evaluation

Run from the **project root** (not from inside `src/`).

### 5a — Retrieval Hit Rate (no LLM, fast)

Measures whether the correct source document appears in the top-k retrieved results:

```bash
python src/evaluate.py --mode retrieval --n 200
```

Output: `Hit@1`, `Hit@3`, `Hit@5` scores and number of evaluated questions.

### 5b — Ablation Study (no LLM)

Measures the contribution of each retrieval component incrementally:

```bash
python src/evaluate.py --mode ablation --n 100
```

Tests 5 configurations: **Dense only → +BM25 → +Reranker → +Freshness → +Boost (full)**.  
Output includes a Markdown table you can paste directly into a report.

### 5c — Full Answer Quality Metrics (requires LLM)

```bash
python src/evaluate.py --mode full --n 100 --llm vllm:Qwen2.5-14B-Instruct
```

Metrics reported: Cosine Similarity, Token Overlap, Jaccard, BLEU-1, ROUGE-L, Citation Accuracy, Citation Coverage.

### 5d — Baseline vs Improved Comparison Report

Runs both the baseline (dense-only) and the full improved pipeline, then writes a detailed text report:

```bash
python src/evaluate_report.py \
    --llm vllm:Qwen2.5-14B-Instruct \
    --n 200 \
    --out report_result.txt
```

Output: `report_result.txt` with a metric comparison table and 5 sample answers for each system.

---

## Evaluation Results (n=200, Qwen2.5-14B-Instruct)

| Metric | Baseline | Improved | Delta |
|---|---:|---:|---:|
| Cosine Similarity | 0.6240 | 0.7291 | +0.1051 |
| Token Overlap | 0.4260 | 0.6343 | +0.2083 |
| Jaccard Similarity | 0.3558 | 0.5436 | +0.1878 |
| BLEU-1 | 0.1746 | 0.3698 | +0.1952 |
| ROUGE-L | 0.2625 | 0.4411 | +0.1785 |

**Baseline**: Dense FAISS top-5 (no BM25, no reranker, no freshness, no parent-child).  
**Improved**: Hybrid BM25+FAISS (RRF) + Cross-encoder reranker + Freshness scoring + Parent-child retrieval.

---

## Key Configuration (src/config.py)

All values can be overridden with `RAG_*` environment variables.

| Parameter | Default | Description |
|---|---:|---|
| `EMBED_MODEL` | `bkai-foundation-models/vietnamese-bi-encoder` | Embedding model |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | Cross-encoder reranker |
| `VECTOR_K` | 30 | FAISS candidates per query |
| `BM25_K` | 30 | BM25 candidates per query |
| `RRF_K` | 60 | RRF damping factor |
| `RERANK_K` | 40 | Candidates sent to cross-encoder |
| `TOP_K` | 6 | Final chunks passed to the LLM |
| `MAX_CHUNKS_PER_DOC` | 3 | Per-document cap |
| `MAX_CONTEXT_CHARS` | 24000 | Max characters in LLM context (~12K tokens) |

---

## Models

| Role | Model |
|---|---|
| Embedding | `bkai-foundation-models/vietnamese-bi-encoder` (PhoBERT-based, no prefix needed) |
| Reranker | `BAAI/bge-reranker-v2-m3` (multilingual cross-encoder) |
| LLM (server) | `Qwen2.5-14B-Instruct` via vLLM |
| LLM (API) | Gemini 2.5 Flash |
| LLM (local) | Any HuggingFace causal-LM, supports 4-bit/8-bit quantisation |

---

## Parent-Child Chunking

The indexer splits each document at two granularities:

- **Parent** = full `Điều` (article) block, up to ~1500 chars. Stored in `parents.jsonl`. What the LLM reads.
- **Child** = sub-segment of a parent, up to ~500 chars. Stored in `chunks.jsonl`. What FAISS and BM25 index.

The cross-encoder scores short `(query, child)` pairs for precision, then the retriever deduplicates by `parent_id` and substitutes the full parent text so the LLM receives the complete article.

---

## Freshness-Aware Scoring

Document status (`tinh_trang`) is mapped to a score multiplier after reranking. Two regimes:

| Status | Base weight | Strict weight (`hiện hành` detected) |
|---|---:|---:|
| active | 1.00 | 1.00 |
| unknown | 0.80 | 0.55 |
| outdated | 0.60 | 0.25 |
| expired | 0.55 | 0.15 |

If the user explicitly names a document (e.g. `50/2024/TT-NHNN`), a +0.5 boost rescues it regardless of regime.

---

## Citation Verification

After the LLM responds, each `[Số hiệu: X, Điều N, Khoản M]` citation is checked against the retrieved context:

| Bucket | Meaning |
|---|---|
| `ok` | so_hieu **and** Điều N both present in a retrieved chunk |
| `partial` | so_hieu matches but Điều N is absent or wrong |
| `hallucinated` | so_hieu not in any retrieved chunk |
