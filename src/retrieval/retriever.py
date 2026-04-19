"""
retriever.py - HybridRetriever: FAISS + BM25 + RRF + cross-encoder + freshness.

Pipeline per query:
  1. Analyze query (intent, explicit so_hieu, from_date) via query_analyzer.
  2. Vector search with multilingual-e5-base (top VECTOR_K).
  3. Lexical search with BM25 (top BM25_K).
  4. Reciprocal Rank Fusion → top RERANK_K candidates.
  5. Cross-encoder reranker (bge-reranker-v2-m3) scores (query, child.text).
  6. Multiply by freshness weight → preserves historical docs while surfacing
     the currently-in-force ones when the user asks for "hiện hành".
  7. Additive boost for chunks whose so_hieu matches an explicit mention.
  8. Dedupe by parent_id (keep best child per Điều) and swap child text for
     parent text so the LLM sees the whole Điều.
  9. Per-document cap + enrich each chunk with metadata (status, expiry note,
     successor văn bản từ 1-hop replaced_by map).

Retrieve accepts ablation flags (use_bm25 / use_reranker / use_freshness /
explicit_boost / use_parents) so evaluate.py can report a clean component
study.

Reranker is lazy-loaded; if the model fails to load, retrieval gracefully
falls back to RRF order (see `_rerank`).
"""
from __future__ import annotations

import json
import os
import pickle
from typing import Optional

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import (
    INDEX_DIR, PARSED_JSONL, EMBED_MODEL, RERANKER_MODEL,
    VECTOR_K, BM25_K, RRF_K, RERANK_K, RERANK_BATCH, RERANK_MAX_LEN,
    TOP_K, MAX_CHUNKS_PER_DOC, QUERY_PREFIX, auto_device,
)
from indexing.doc_metadata import DocMetadata, STATUS_EXPIRED, STATUS_OUTDATED, STATUS_UNKNOWN
from retrieval.freshness import freshness_weight
from retrieval.query_analyzer import analyze_query, tokenize_vi


class HybridRetriever:
    def __init__(self, lazy_reranker: bool = True, device: Optional[str] = None):
        self.device = device or auto_device()

        print("Loading index …")
        self.chunks: list[dict] = []
        with open(os.path.join(INDEX_DIR, "chunks.jsonl"), encoding="utf-8") as f:
            for line in f:
                self.chunks.append(json.loads(line.strip()))

        self._parents: dict[str, dict] = {}
        parents_path = os.path.join(INDEX_DIR, "parents.jsonl")
        if os.path.exists(parents_path):
            with open(parents_path, encoding="utf-8") as f:
                for line in f:
                    p = json.loads(line.strip())
                    self._parents[p["parent_id"]] = p
            print(f"  parents: {len(self._parents)} (parent-child mode)")
        else:
            print("  parents.jsonl not found — running without parent expansion.")

        self.faiss_index = faiss.read_index(os.path.join(INDEX_DIR, "faiss.index"))

        with open(os.path.join(INDEX_DIR, "bm25_corpus.pkl"), "rb") as f:
            corpus = pickle.load(f)
        self.bm25 = BM25Okapi(corpus)

        print(f"Loading embedding model: {EMBED_MODEL} (device={self.device}) …")
        self.embed_model = SentenceTransformer(EMBED_MODEL, device=self.device)

        print("Loading document metadata sidecar …")
        self.meta = DocMetadata(PARSED_JSONL)
        print(f"  metadata: {len(self.meta)} docs")

        self._reranker = None
        if not lazy_reranker:
            self._load_reranker()
        print("Retriever ready.")

    # ── Reranker ──────────────────────────────────────────────────────────────

    def _load_reranker(self) -> None:
        if self._reranker is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            print(f"Loading reranker: {RERANKER_MODEL} (device={self.device}) …")
            self._reranker = CrossEncoder(
                RERANKER_MODEL, device=self.device, max_length=RERANK_MAX_LEN,
            )
        except Exception as e:
            print(f"  ! reranker unavailable ({e}). Falling back to RRF order.")
            self._reranker = False

    # ── Candidate stages ──────────────────────────────────────────────────────

    def _vector_search(self, query: str, k: int) -> list[tuple[int, float]]:
        vec = self.embed_model.encode(
            [QUERY_PREFIX + query],
            normalize_embeddings=True, convert_to_numpy=True,
        ).astype("float32")
        scores, idxs = self.faiss_index.search(vec, k)
        return [(int(i), float(s)) for i, s in zip(idxs[0], scores[0]) if i >= 0]

    def _bm25_search(self, query: str, k: int) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize_vi(query))
        top = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top]

    def _rrf_fusion(
        self,
        vec_hits: list[tuple[int, float]],
        bm25_hits: list[tuple[int, float]],
    ) -> list[int]:
        rrf: dict[int, float] = {}
        for rank, (idx, _) in enumerate(vec_hits):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, (idx, _) in enumerate(bm25_hits):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
        return [i for i, _ in sorted(rrf.items(), key=lambda x: x[1], reverse=True)]

    def _rerank(self, query: str, idxs: list[int]) -> list[tuple[int, float]]:
        self._load_reranker()
        if not self._reranker:
            return [(idx, 1.0 - rank * 0.01) for rank, idx in enumerate(idxs)]
        pairs = [(query, self.chunks[i]["text"]) for i in idxs]
        raw = self._reranker.predict(
            pairs, batch_size=RERANK_BATCH, show_progress_bar=False,
        )
        sig = 1.0 / (1.0 + np.exp(-np.asarray(raw, dtype=np.float32)))
        return list(zip(idxs, sig.tolist()))

    # ── Enrichment & final selection ──────────────────────────────────────────

    def _enrich(self, idx: int) -> dict:
        chunk = dict(self.chunks[idx])
        sh = chunk.get("so_hieu", "")
        m  = self.meta.get(sh)
        chunk["tinh_trang"]   = m.get("tinh_trang_raw", "")
        chunk["_status"]      = m.get("status", STATUS_UNKNOWN)
        chunk["_expire_date"] = m.get("expire_date")

        note = ""
        if chunk["_status"] == STATUS_EXPIRED:
            exp = m.get("expire_date")
            if exp:
                note = f"Lưu ý: văn bản này đã HẾT HIỆU LỰC từ {exp[2]:02d}/{exp[1]:02d}/{exp[0]}."
            else:
                note = "Lưu ý: văn bản này đã HẾT HIỆU LỰC."
        elif chunk["_status"] == STATUS_OUTDATED:
            note = "Lưu ý: văn bản này không còn phù hợp."

        # 1-hop replaced_by: if this doc is superseded, tell the reader what
        # replaced it. Requires DocMetadata.replaced_by built from the inverse
        # "thay_the" map.
        if chunk["_status"] in (STATUS_EXPIRED, STATUS_OUTDATED):
            successors = self.meta.replaced_by(sh) if hasattr(self.meta, "replaced_by") else []
            if successors:
                chunk["_successors"] = successors
                note = (note + " " if note else "") + \
                       f"Đã được thay thế bởi: {', '.join(successors)}."

        chunk["_freshness_note"] = note
        return chunk

    def _select_top(
        self,
        reranked: list[tuple[int, float]],
        analysis: dict,
        top_k: int,
        *,
        use_freshness: bool = True,
        explicit_boost: bool = True,
        use_parents:    bool = True,
    ) -> list[dict]:
        explicit = set(analysis["explicit_so_hieu"]) if explicit_boost else set()
        scored: list[tuple[float, dict]] = []
        for idx, base in reranked:
            ch = self._enrich(idx)
            w  = freshness_weight(ch["_status"], analysis["prefers_current"]) \
                 if use_freshness else 1.0
            score = base * w
            if explicit:
                so_hieu_norm = (ch.get("so_hieu") or "").strip().lower()
                if any(e and (e in so_hieu_norm or so_hieu_norm in e) for e in explicit):
                    score += 0.5
            ch["_score"]     = float(score)
            ch["_rerank"]    = float(base)
            ch["_freshness"] = float(w)
            scored.append((score, ch))

        scored.sort(key=lambda x: x[0], reverse=True)

        seen_parents: set[str] = set()
        per_doc:      dict[str, int] = {}
        out: list[dict] = []
        for _, ch in scored:
            pid = ch.get("parent_id") or ""
            if pid and pid in seen_parents:
                continue

            key = ch.get("so_hieu") or ch.get("_file") or ""
            if per_doc.get(key, 0) >= MAX_CHUNKS_PER_DOC:
                continue

            if pid:
                seen_parents.add(pid)
            per_doc[key] = per_doc.get(key, 0) + 1

            # Swap child text → parent text (LLM gets the whole Điều; reranker
            # already scored on the more precise child).
            if use_parents and pid and pid in self._parents:
                ch["_child_text"] = ch["text"]
                ch["text"]        = self._parents[pid]["text"]

            out.append(ch)
            if len(out) >= top_k:
                break
        return out

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K,
        *,
        use_bm25:       bool = True,
        use_reranker:   bool = True,
        use_freshness:  bool = True,
        explicit_boost: bool = True,
        use_parents:    bool = True,
    ) -> list[dict]:
        analysis  = analyze_query(query)
        vec_hits  = self._vector_search(query, VECTOR_K)
        bm25_hits = self._bm25_search(query, BM25_K) if use_bm25 else []
        fused     = self._rrf_fusion(vec_hits, bm25_hits)[:RERANK_K]
        if use_reranker:
            reranked = self._rerank(query, fused)
        else:
            reranked = [(idx, 1.0 - rank * 0.01) for rank, idx in enumerate(fused)]
        return self._select_top(
            reranked, analysis, top_k,
            use_freshness=use_freshness,
            explicit_boost=explicit_boost,
            use_parents=use_parents,
        )
