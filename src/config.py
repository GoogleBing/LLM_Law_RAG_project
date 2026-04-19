"""
config.py - Centralized paths and model names.

Paths default to project-relative so the code ports to any OS/server. Override
any value with an env var of the same name (e.g. RAG_INDEX_DIR=/data/index).
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _p(env_key: str, default: Path) -> str:
    return os.environ.get(env_key, str(default))


# ── Data & index locations ────────────────────────────────────────────────────
RAW_DOCS_DIR  = _p("RAG_RAW_DOCS_DIR",  PROJECT_ROOT / "export_1")
PARSED_JSONL  = _p("RAG_PARSED_JSONL",  PROJECT_ROOT / "parsed_documents.jsonl")
INDEX_DIR     = _p("RAG_INDEX_DIR",     PROJECT_ROOT / "index")

# Eval artefacts
RES_XLSX      = _p("RAG_RES_XLSX",      PROJECT_ROOT / "RES.xlsx")
RES_CSV       = _p("RAG_RES_CSV",       PROJECT_ROOT / "RES.csv")

# ── Model names ───────────────────────────────────────────────────────────────
EMBED_MODEL    = os.environ.get("RAG_EMBED_MODEL",    "bkai-foundation-models/vietnamese-bi-encoder")
RERANKER_MODEL = os.environ.get("RAG_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
LLM_MODEL      = os.environ.get("RAG_LLM_MODEL",      "Qwen2.5-14B-Instruct")

# ── Retrieval hyper-params ────────────────────────────────────────────────────
VECTOR_K            = int(os.environ.get("RAG_VECTOR_K",   "30"))
BM25_K              = int(os.environ.get("RAG_BM25_K",     "30"))
RRF_K               = int(os.environ.get("RAG_RRF_K",      "60"))
RERANK_K            = int(os.environ.get("RAG_RERANK_K",   "40"))
RERANK_BATCH        = int(os.environ.get("RAG_RERANK_BATCH", "8"))
RERANK_MAX_LEN      = int(os.environ.get("RAG_RERANK_MAX_LEN", "512"))
TOP_K               = int(os.environ.get("RAG_TOP_K",      "6"))
MAX_CHUNKS_PER_DOC  = int(os.environ.get("RAG_MAX_PER_DOC", "3"))
# Max chars of retrieved context sent to the LLM.
# Qwen2.5-14B: 32768-token window, ~2 chars/token → ~24 000 chars is ~12 000 tokens,
# leaving 20 000 tokens for system prompt, query, and output.
MAX_CONTEXT_CHARS   = int(os.environ.get("RAG_MAX_CONTEXT_CHARS", "24000"))

# Query/doc prefixes — empty for bkai (PhoBERT-based), "query: "/"passage: " for e5
QUERY_PREFIX = os.environ.get("RAG_QUERY_PREFIX", "")
DOC_PREFIX   = os.environ.get("RAG_DOC_PREFIX",   "")

# Backward-compat aliases (some older modules may import these names)
E5_QUERY_PREFIX = QUERY_PREFIX
E5_DOC_PREFIX   = DOC_PREFIX


def auto_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
