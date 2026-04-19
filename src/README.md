# RAG Chatbox — Vietnamese Legal Documents

Hybrid-retrieval RAG over **~16,400 Vietnamese legal documents**
(`parsed_documents.jsonl`). Combines FAISS dense retrieval, BM25 lexical
retrieval, a cross-encoder reranker, freshness-aware scoring, parent-child
chunking, a 1-hop successor map, and post-hoc citation verification.

---

## Pipeline at a glance

```
raw docs (export_1/)                                      ← given
      │  parser.py
      ▼
parsed_documents.jsonl ── doc_metadata.py ──► doc_meta.pkl
      │                                         (tinh_trang,
      │                                          ngay_hieu_luc,
      │                                          replaced_by  ← 1-hop)
      │
      │  chunker.py  (parent = full Điều,
      │               child  = ~500-char sub-segment)
      ▼
index/ { faiss.index, chunks.jsonl (children),
         parents.jsonl, bm25_corpus.pkl }
      │
──────┼──────────────────────────────────────────────────────  query time
      ▼
HybridRetriever                       ┐
  1. analyze_query                    │
  2. FAISS  top-30   (on children)   │ retrieval/
  3. BM25   top-30   (on children)   │
  4. RRF → top-40                    │
  5. bge-reranker-v2-m3  (on child)  │
  6. × freshness_weight              │
  7. + 0.5 boost for explicit so_hieu│
  8. dedupe by parent_id →           │
     swap child text → parent text   │
  9. per-doc cap + enrich            │
     (status + successor so_hieu)    ┘
      │
      ▼
build_prompt ── SYSTEM_INSTRUCTION          ┐
  enforce [Số hiệu: X, Điều N, Khoản M]     │
  flag HẾT HIỆU LỰC + "thay thế bởi"        │ generation/
  refuse if context insufficient            │
      │                                     │
      ▼                                     │
make_llm → Gemini / HF (4-bit/8-bit)        │
      │                                     │
      ▼                                     │
verify_citations  (post-hoc check against   │
  retrieved chunks → ok / partial /         │
  hallucinated)                             ┘
      │
      ▼
    answer
```

---

## Folder layout

```
src/
├── config.py                  — paths + hyper-params (override via env vars)
├── rag_pipeline.py            — backward-compat facade (re-exports)
├── build_index.py             — CLI: build FAISS + BM25 + parents  (run once)
├── demo.py                    — CLI: one-shot question → contexts + answer + citation check
├── evaluate.py                — CLI: retrieval / full / ablation metrics
│
├── indexing/                  # Stage 1 : parsed_documents.jsonl → index
│   ├── chunker.py             — parent (Điều) + child (sub-segment) chunker
│   └── doc_metadata.py        — sidecar cache: status + dates + replaced_by map
│
├── retrieval/                 # Stage 2 : query → ranked contexts
│   ├── query_analyzer.py      — regex intent / so_hieu / from_date
│   ├── freshness.py           — tinh_trang → weight  (base / strict regimes)
│   └── retriever.py           — HybridRetriever (hybrid + rerank + parent swap)
│
└── generation/                # Stage 3 : contexts → LLM answer
    ├── prompt_builder.py      — SYSTEM_INSTRUCTION + build_prompt
    ├── llm_providers.py       — make_llm() dispatcher (Gemini / HF)
    ├── citation.py            — verify_citations (ok/partial/hallucinated)
    └── pipeline.py            — RAGPipeline (retriever + prompt + llm + verify)
```

---

## Setup

```bash
# Python 3.11 recommended.  GPU strongly recommended for embedding + rerank.
pip install faiss-cpu sentence-transformers rank-bm25 transformers \
            google-genai bitsandbytes pandas openpyxl tqdm python-dotenv
```

**Gemini** (optional): create a project-root `.env`:

```
GEMINI_API_KEY=...
```

`llm_providers.py` auto-loads `.env`; the key is never logged.

---

## Run order (first-time setup on a server)

```bash
# 0. cd into repo root, confirm parsed_documents.jsonl is present
ls parsed_documents.jsonl

# 1. (OPTIONAL) sanity-check the chunker without touching FAISS.
python src/indexing/chunker.py
#    → prints doc / parent / child counts

# 2. Build the index.  This reads parsed_documents.jsonl, emits:
#       index/chunks.jsonl    (children — what BM25/FAISS see)
#       index/parents.jsonl   (full Điều — swapped in for the LLM)
#       index/faiss.index
#       index/bm25_corpus.pkl
#    On a decent GPU this is 30 min – 2 h depending on corpus size.
python src/build_index.py

# 3. (OPTIONAL) build the metadata sidecar ahead of time. Otherwise the
#    retriever builds it lazily on first load (~30 s).
python src/indexing/doc_metadata.py

# 4. Retrieval-only smoke test (fast, no LLM, no API key).
python src/demo.py "Quy định về Soft OTP hiện hành?"

# 5. With Gemini (needs GEMINI_API_KEY).
python src/demo.py --llm gemini-2.5-flash "Điều 11 Thông tư 50/2024/TT-NHNN?"

# 6. With a local HF model, 4-bit on one GPU.
python src/demo.py --llm Qwen/Qwen2.5-7B-Instruct --quant 4bit "..."
```

---

## Evaluation

```bash
# Hit@{1,3,5} by so_hieu — fast, no LLM
python src/evaluate.py --mode retrieval --n 200

# Component ablation: dense only → +BM25 → +reranker → +freshness → +boost
python src/evaluate.py --mode ablation --n 200
#    → prints a table with Hit@1/3/5 for each config (+ markdown block you
#      can paste straight into the report).

# Full answer metrics (cosine / token_overlap / jaccard / BLEU-1 / ROUGE-L)
# PLUS citation_accuracy and citation_coverage.
python src/evaluate.py --mode full --n 100 --llm gemini-2.5-flash
python src/evaluate.py --mode full --n 100 --llm Qwen/Qwen2.5-7B-Instruct --quant 4bit
```

Reads `RES.xlsx` (`Câu Hỏi`, `Trả lời`, `Số hiệu VBPL (Trích xuất)`).
The embedding for cosine-sim is **reused from the retriever** — no second load.

---

## Hyper-params (config.py)

| name | default | what it controls |
| --- | ---: | --- |
| `VECTOR_K` | 30 | FAISS candidates per query |
| `BM25_K` | 30 | BM25 candidates per query |
| `RRF_K` | 60 | Reciprocal-rank-fusion damping (higher = flatter) |
| `RERANK_K` | 40 | How many RRF-fused chunks go to cross-encoder |
| `RERANK_BATCH` | 8 | Reranker batch size (tune for VRAM) |
| `RERANK_MAX_LEN` | 512 | Reranker max sequence length |
| `TOP_K` | 6 | Final chunks passed to prompt |
| `MAX_CHUNKS_PER_DOC` | 3 | Per-document cap after reranking |

Override any of these with env vars prefixed `RAG_` (e.g. `RAG_TOP_K=8`).
Paths follow the same pattern: `RAG_PARSED_JSONL`, `RAG_INDEX_DIR`, ...

**Chunker** constants live at the top of `indexing/chunker.py`:

| name | default | role |
| --- | ---: | --- |
| `MAX_PARENT_CHARS` | 1500 | parent (Điều) packing limit in paragraph fallback |
| `MAX_CHILD_CHARS` | 500 | child sub-segment cap (what gets embedded) |
| `CHILD_OVERLAP` | 50 | rolling overlap between children |

Raising `MAX_CHILD_CHARS` → fewer children → faster indexing, coarser match.

---

## Freshness logic

`tinh_trang` (from VBPL) is mapped to a multiplier applied **after** the
cross-encoder score. Two regimes (see `retrieval/freshness.py`):

| status | base | strict (`hiện hành` / `mới nhất` detected) |
| --- | ---: | ---: |
| active | 1.00 | 1.00 |
| unknown | 0.80 | 0.55 |
| outdated | 0.60 | 0.25 |
| expired | 0.55 | 0.15 |

If the user names a văn bản explicitly (e.g. `10/2021/TT-NHNN`), a `+0.5`
additive boost rescues it even under the strict regime — so old law is still
retrievable when asked by name.

---

## Parent-child retrieval

The chunker emits two granularities per document:

- **Parent** = one full `Điều` (or a paragraph-packed block when the doc has
  no `Điều` pattern). Stored in `parents.jsonl`. What the LLM reads.
- **Child**  = a ~500-char sub-segment of a parent. Stored in `chunks.jsonl`.
  What FAISS and BM25 actually index.

The reranker scores (query, child.text) — short, precise pairs — and then the
retriever dedupes by `parent_id` and substitutes the parent text for the LLM.
This gives the LLM the whole Điều (good for "các khoản của Điều 5") while
keeping the reranker and cross-encoder fast.

---

## 1-hop replaced_by map

`DocMetadata` parses each doc's `van_ban_lien_quan` entries with
`loai == "thay_the"`, extracts any số hiệu mentioned in the description, and
builds an inverse map `old_so_hieu → [new_so_hieu, ...]`.

When the retriever hits a chunk whose status is `expired` or `outdated`, it
appends the successor(s) to `_freshness_note`, e.g.:

> Lưu ý: văn bản này đã HẾT HIỆU LỰC từ 01/07/2024. Đã được thay thế bởi:
> 14/2022/TT-NHNN.

The LLM sees this in context and can steer its answer to the live document.

---

## Citation verification

After the LLM responds, `generation/citation.py` parses citations out of the
answer and compares every `(so_hieu, Điều N)` pair against the retrieved
chunks. Each citation is bucketed:

| bucket | meaning |
| --- | --- |
| `ok` | so_hieu **and** Điều N both present in a retrieved chunk |
| `partial` | so_hieu matches a chunk but Điều N is absent / unparsed |
| `hallucinated` | so_hieu not in any retrieved chunk — LLM fabricated it |

The demo CLI prints this as a one-screen summary; `evaluate.py --mode full`
reports `citation_accuracy` (ok / total) and `citation_coverage` (fraction of
answers that cited at least one source).

Regex tolerates minor format drift (`Đ.5`, `điều 5`, `ĐIỀU 5`, spaces between
separators) and builds (so_hieu, Điều) pairs by proximity rather than a rigid
single-regex match.

---

## Models

| role | model | why |
| --- | --- | --- |
| embedding | `bkai-foundation-models/vietnamese-bi-encoder` | Vietnamese legal bi-encoder (PhoBERT-based); no prefix required. |
| reranker | `BAAI/bge-reranker-v2-m3` | Strong multilingual cross-encoder; batched + truncated to fit consumer GPUs. |
| LLM (API) | Gemini 2.5 Flash | Fast, cheap, good Vietnamese. |
| LLM (local) | any HF causal-LM | `make_llm()` supports 4-bit / 8-bit quant via bitsandbytes. |

---

## Notes

- `rag_pipeline.py` is a **facade** kept only for backward compatibility; new
  code should import from the stage packages directly.
- `check_parse.py` and `parser.py` at repo root are pre-processing utilities —
  `parsed_documents.jsonl` is already built for the demo, so these are not on
  the hot path.
- All paths resolve from `PROJECT_ROOT` (parent of `src/`). The code is
  portable: drop the repo on a server, run `build_index.py`, and it just works.
