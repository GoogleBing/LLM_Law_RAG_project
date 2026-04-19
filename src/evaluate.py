"""
evaluate.py - Evaluate RAG pipeline on RES dataset.

Modes:
  --mode retrieval   Hit@{1,3,5} by so_hieu (no LLM needed).
  --mode ablation    Hit@{1,3,5} across 5 incremental retrieval configs.
  --mode full        Cosine / Token Overlap / Jaccard / BLEU-1 / ROUGE-L
                     + citation accuracy. Requires --llm.

Examples:
    python src/evaluate.py --mode retrieval --n 200
    python src/evaluate.py --mode ablation  --n 100
    python src/evaluate.py --mode full      --n 100 --llm vllm:Qwen2.5-14B-Instruct
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

from config import RES_CSV as RES_PATH


# ── Text metrics ──────────────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\w\u00C0-\u024F\u1E00-\u1EFF]+", text.lower())


def token_overlap(pred: str, ref: str) -> float:
    p, r = set(_tokenize(pred)), set(_tokenize(ref))
    return len(p & r) / len(r) if r else 0.0


def jaccard(pred: str, ref: str) -> float:
    p, r = set(_tokenize(pred)), set(_tokenize(ref))
    return len(p & r) / len(p | r) if (p | r) else 0.0


def bleu1(pred: str, ref: str) -> float:
    p_tok, r_tok = _tokenize(pred), _tokenize(ref)
    if not p_tok:
        return 0.0
    p_cnt, r_cnt = Counter(p_tok), Counter(r_tok)
    overlap   = sum(min(c, r_cnt[t]) for t, c in p_cnt.items())
    precision = overlap / len(p_tok)
    bp = 1.0 if len(p_tok) >= len(r_tok) else np.exp(1 - len(r_tok) / len(p_tok))
    return bp * precision


def rouge_l(pred: str, ref: str) -> float:
    p_tok, r_tok = _tokenize(pred), _tokenize(ref)
    if not p_tok or not r_tok:
        return 0.0
    m, n = len(p_tok), len(r_tok)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if p_tok[i-1] == r_tok[j-1] \
                       else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    P, R = lcs / m, lcs / n
    return 2 * P * R / (P + R) if (P + R) else 0.0


def cosine_sim(pred: str, ref: str, embed_fn) -> float:
    a, b = np.asarray(embed_fn([pred, ref]))
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_and_filter(res_path: str, meta: dict) -> pd.DataFrame:
    df = pd.read_csv(res_path, encoding="utf-8", on_bad_lines="skip")
    df.columns = [c.strip() for c in df.columns]
    col_q  = next(c for c in df.columns if "Câu" in c and "Hỏi" in c)
    col_a  = next(c for c in df.columns if "Trả lời" in c and "URAx" not in c and "đúng" not in c)
    col_id = next(c for c in df.columns if "Số hiệu" in c or "VBPL" in c)
    df = df[[col_q, col_a, col_id]].copy()
    df.columns = ["question", "reference", "so_hieu"]
    df = df.dropna(subset=["question", "reference"])
    df["so_hieu"] = df["so_hieu"].fillna("").astype(str).str.strip()

    before = len(df)
    df = df[df["so_hieu"].apply(
        lambda sh: sh not in ("", "nan", "Không trích xuất được")
                   and bool(meta.get(sh.lower()))
    )].reset_index(drop=True)
    print(f"  Loaded {before} rows → {len(df)} after filtering to indexed docs")
    return df


# ── Retrieval evaluation ──────────────────────────────────────────────────────

def _hits_for(retriever, df: pd.DataFrame, n: int, kw: dict) -> dict:
    hits = {1: 0, 3: 0, 5: 0}
    total = 0
    for _, row in tqdm(df.head(n).iterrows(), total=min(n, len(df)), desc="eval"):
        so_hieu = row["so_hieu"].lower()
        results = retriever.retrieve(str(row["question"]), top_k=5, **kw)
        ids = [(r.get("so_hieu") or "").strip().lower() for r in results]
        for k in (1, 3, 5):
            if any(so_hieu in rid or rid in so_hieu for rid in ids[:k]):
                hits[k] += 1
        total += 1
    total = total or 1
    return {"Hit@1": hits[1]/total, "Hit@3": hits[3]/total,
            "Hit@5": hits[5]/total, "n": total}


def eval_retrieval(retriever, df: pd.DataFrame, n: int) -> dict:
    return _hits_for(retriever, df, n, kw={})


def eval_ablation(retriever, df: pd.DataFrame, n: int) -> list[dict]:
    configs = [
        ("Dense only",       dict(use_bm25=False, use_reranker=False,
                                  use_freshness=False, explicit_boost=False)),
        ("+ BM25 (RRF)",     dict(use_bm25=True,  use_reranker=False,
                                  use_freshness=False, explicit_boost=False)),
        ("+ Reranker",       dict(use_bm25=True,  use_reranker=True,
                                  use_freshness=False, explicit_boost=False)),
        ("+ Freshness",      dict(use_bm25=True,  use_reranker=True,
                                  use_freshness=True,  explicit_boost=False)),
        ("+ Boost (full)",   dict(use_bm25=True,  use_reranker=True,
                                  use_freshness=True,  explicit_boost=True)),
    ]
    rows = []
    for name, kw in configs:
        print(f"\n--- {name} ---")
        rows.append({"config": name, **_hits_for(retriever, df, n, kw)})
    return rows


def _print_table(rows: list[dict]) -> None:
    print(f"\n{'Config':<22} {'Hit@1':>7} {'Hit@3':>7} {'Hit@5':>7}")
    print("-" * 45)
    for r in rows:
        print(f"  {r['config']:<20} {r['Hit@1']:>7.3f} {r['Hit@3']:>7.3f} {r['Hit@5']:>7.3f}")
    print("\nMarkdown:")
    print("| Config | Hit@1 | Hit@3 | Hit@5 |")
    print("|---|---:|---:|---:|")
    for r in rows:
        print(f"| {r['config']} | {r['Hit@1']:.3f} | {r['Hit@3']:.3f} | {r['Hit@5']:.3f} |")


# ── Full answer evaluation ────────────────────────────────────────────────────

def eval_full(pipeline, embed_fn, df: pd.DataFrame, n: int) -> dict:
    agg: dict[str, list] = {k: [] for k in
                             ("cosine", "token_overlap", "jaccard", "bleu1", "rouge_l",
                              "citation_accuracy", "citation_coverage")}
    for _, row in tqdm(df.head(n).iterrows(), total=min(n, len(df)), desc="full eval"):
        result = pipeline.answer(str(row["question"]))
        pred   = result.get("response") or ""
        ref    = str(row["reference"])
        agg["cosine"].append(cosine_sim(pred, ref, embed_fn))
        agg["token_overlap"].append(token_overlap(pred, ref))
        agg["jaccard"].append(jaccard(pred, ref))
        agg["bleu1"].append(bleu1(pred, ref))
        agg["rouge_l"].append(rouge_l(pred, ref))
        cit = result.get("citations") or {}
        agg["citation_accuracy"].append(cit.get("score", 0.0))
        agg["citation_coverage"].append(1.0 if cit.get("total", 0) > 0 else 0.0)
    return {k: float(np.mean(v)) for k, v in agg.items()}


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate RAG pipeline on RES dataset.")
    p.add_argument("--mode",  choices=["retrieval", "ablation", "full"], default="retrieval")
    p.add_argument("--n",     type=int, default=200, help="Max questions to evaluate")
    p.add_argument("--llm",   default="vllm:Qwen2.5-14B-Instruct",
                   help="LLM for --mode full (vllm:<model> | gemini-* | HF repo)")
    p.add_argument("--quant", choices=["none", "4bit", "8bit"], default="none")
    return p.parse_args()


def main():
    args = parse_args()

    from retrieval.retriever import HybridRetriever
    retriever = HybridRetriever()

    print(f"\nLoading dataset: {RES_PATH}")
    df = load_and_filter(RES_PATH, retriever.meta)
    n  = min(args.n, len(df))

    if args.mode == "retrieval":
        print("\n=== RETRIEVAL HIT RATE ===")
        r = eval_retrieval(retriever, df, n)
        for k in ("Hit@1", "Hit@3", "Hit@5"):
            print(f"  {k}: {r[k]:.3f}")
        print(f"  Evaluated: {r['n']} questions")

    elif args.mode == "ablation":
        print("\n=== ABLATION STUDY ===")
        rows = eval_ablation(retriever, df, n)
        _print_table(rows)

    else:
        from generation.llm_providers import make_llm
        from generation.pipeline import RAGPipeline
        print(f"\nLoading LLM: {args.llm} ...")
        llm_fn   = make_llm(args.llm, quant=args.quant)
        embed_fn = lambda texts: retriever.embed_model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True)
        pipeline = RAGPipeline(llm=llm_fn, retriever=retriever)

        print("\n=== FULL RAG EVALUATION ===")
        results = eval_full(pipeline, embed_fn, df, n)
        print(f"\n{'Metric':<22} {'Score':>8}")
        print("-" * 32)
        for k, v in results.items():
            print(f"  {k:<20} {v:>8.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
