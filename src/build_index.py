"""
build_index.py - Build FAISS vector index + save BM25 corpus for hybrid retrieval.

Uses bkai-foundation-models/vietnamese-bi-encoder (specialised for Vietnamese legal text).
No special prefix needed (unlike e5); controlled via DOC_PREFIX in config.py.

Run once:
    python src/build_index.py

Outputs (in index/ folder):
    index/faiss.index      - FAISS flat IP index
    index/chunks.jsonl     - chunk metadata (text + fields)
    index/parents.jsonl    - parent (full Điều) metadata for context expansion
    index/bm25_corpus.pkl  - tokenized corpus for BM25
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
from pathlib import Path

import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from indexing.chunker import iter_chunks_and_parents
from config           import PARSED_JSONL as JSONL_PATH, INDEX_DIR, EMBED_MODEL, \
                             DOC_PREFIX, auto_device

BATCH_SIZE = 64
DEVICE     = auto_device()


def tokenize_vi(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    import re
    text = text.lower()
    tokens = re.findall(r"[\w\u00C0-\u024F\u1E00-\u1EFF]+", text)
    return tokens


def main():
    os.makedirs(INDEX_DIR, exist_ok=True)
    chunks_path   = os.path.join(INDEX_DIR, "chunks.jsonl")
    parents_path  = os.path.join(INDEX_DIR, "parents.jsonl")
    faiss_path    = os.path.join(INDEX_DIR, "faiss.index")
    bm25_path     = os.path.join(INDEX_DIR, "bm25_corpus.pkl")

    print("Loading embedding model …")
    model = SentenceTransformer(EMBED_MODEL, device=DEVICE)
    dim = model.get_embedding_dimension()
    print(f"  Model: {EMBED_MODEL}  dim={dim}  device={DEVICE}")

    # ── Collect all chunks + parents ──────────────────────────────────────────
    print("\nCollecting chunks …")
    all_chunks:  list[dict] = []
    all_parents: list[dict] = []
    for children, parents in tqdm(iter_chunks_and_parents(JSONL_PATH), desc="chunking"):
        all_chunks.extend(children)
        all_parents.extend(parents)
    print(f"  Children (indexed): {len(all_chunks)}")
    print(f"  Parents  (context): {len(all_parents)}")

    # ── Save chunk + parent metadata ──────────────────────────────────────────
    print(f"\nSaving chunk metadata → {chunks_path}")
    with open(chunks_path, "w", encoding="utf-8") as f:
        for ch in all_chunks:
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")

    print(f"Saving parent metadata → {parents_path}")
    with open(parents_path, "w", encoding="utf-8") as f:
        for pa in all_parents:
            f.write(json.dumps(pa, ensure_ascii=False) + "\n")

    # ── Build BM25 corpus ─────────────────────────────────────────────────────
    print("\nBuilding BM25 corpus …")
    bm25_corpus = [tokenize_vi(ch["text"]) for ch in tqdm(all_chunks, desc="tokenizing")]
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_corpus, f)
    print(f"  Saved → {bm25_path}")

    # ── Build FAISS index ─────────────────────────────────────────────────────
    print("\nEmbedding chunks (this may take 10-30 min depending on GPU) …")
    texts = [DOC_PREFIX + ch["text"] for ch in all_chunks]

    index = faiss.IndexFlatIP(dim)   # inner product → cosine after normalisation

    t0 = time.time()
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="embedding"):
        batch = texts[i : i + BATCH_SIZE]
        vecs = model.encode(
            batch,
            normalize_embeddings=True,   # cosine via IP
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        vecs = np.ascontiguousarray(vecs, dtype=np.float32)
        index.add(vecs)

    elapsed = time.time() - t0
    print(f"  Embedded {index.ntotal} chunks in {elapsed/60:.1f} min")

    faiss.write_index(index, faiss_path)
    print(f"  Saved FAISS index → {faiss_path}")
    print("\nDone. Index ready.")


if __name__ == "__main__":
    main()
