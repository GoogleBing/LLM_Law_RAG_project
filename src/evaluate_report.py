"""
evaluate_report.py - Compare Baseline vs Improved RAG on RES dataset.

Baseline  : Dense FAISS top-5 (no BM25, no reranker, no freshness, no parent-child).
            Same embedding (bkai) and LLM as improved — isolates retrieval gains.
Improved  : Full hybrid pipeline (BM25+FAISS RRF + cross-encoder + freshness +
            parent-child).

Metrics   : Cosine Similarity, Token Overlap, Jaccard, BLEU-1, ROUGE-L
            (as required by RAG_Assignment.pdf Section 2).

Output    : report.txt (human-readable comparison table + sample answers).

Usage:
    python src/evaluate_report.py \\
        --llm vllm:Qwen2.5-14B-Instruct \\
        --res "RES ver2.csv" \\
        --n 200 \\
        --out report.txt
"""
from __future__ import annotations

import argparse
import re
import sys
import textwrap
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

from config import INDEX_DIR, EMBED_MODEL, LLM_MODEL, RES_CSV, TOP_K  # noqa: E402


# ── Text metrics (per RAG_Assignment.pdf §2) ──────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\w\u00C0-\u024F\u1E00-\u1EFF]+", text.lower())


def cosine_sim(pred: str, ref: str, embed_fn) -> float:
    vecs = embed_fn([pred, ref])
    a, b = np.asarray(vecs[0]), np.asarray(vecs[1])
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


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
    overlap  = sum(min(c, r_cnt[t]) for t, c in p_cnt.items())
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


def _compute_metrics(pred: str, ref: str, embed_fn) -> dict:
    return {
        "cosine":        cosine_sim(pred, ref, embed_fn),
        "token_overlap": token_overlap(pred, ref),
        "jaccard":       jaccard(pred, ref),
        "bleu1":         bleu1(pred, ref),
        "rouge_l":       rouge_l(pred, ref),
    }


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_res(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    df.columns = [c.strip() for c in df.columns]

    col_q   = next(c for c in df.columns if "Câu" in c and "Hỏi" in c)
    col_ref = next(c for c in df.columns if "Trả lời" in c and "URAx" not in c
                   and "đúng" not in c)
    col_id  = next(c for c in df.columns if "Số hiệu" in c or "VBPL" in c)

    df = df[[col_q, col_ref, col_id]].copy()
    df.columns = ["question", "reference", "so_hieu"]
    df = df.dropna(subset=["question", "reference"])
    df["so_hieu"] = df["so_hieu"].fillna("").astype(str).str.strip()
    return df


def filter_to_dataset(df: pd.DataFrame, meta) -> pd.DataFrame:
    """Keep rows whose so_hieu resolves to a doc in the metadata store."""
    def _in_meta(sh: str) -> bool:
        if not sh or sh in ("nan", "Không trích xuất được"):
            return False
        return bool(meta.get(sh.lower()))

    mask = df["so_hieu"].apply(_in_meta)
    return df[mask].reset_index(drop=True)


# ── Evaluation loop ───────────────────────────────────────────────────────────

def run_eval(pipeline, embed_fn, df: pd.DataFrame, n: int,
             retrieve_kwargs: dict, label: str) -> tuple[dict, list[dict]]:
    agg = {k: [] for k in ("cosine", "token_overlap", "jaccard", "bleu1", "rouge_l")}
    samples: list[dict] = []

    for _, row in tqdm(df.head(n).iterrows(), total=min(n, len(df)), desc=label):
        q   = str(row["question"])
        ref = str(row["reference"])

        result = pipeline.answer(q, retrieve_kwargs=retrieve_kwargs)
        pred   = result.get("response") or ""

        m = _compute_metrics(pred, ref, embed_fn)
        for k, v in m.items():
            agg[k].append(v)

        if len(samples) < 5:
            samples.append({"question": q, "reference": ref,
                            "prediction": pred, **m})

    return {k: float(np.mean(v)) if v else 0.0 for k, v in agg.items()}, samples


# ── Report formatter ──────────────────────────────────────────────────────────

METRIC_LABELS = {
    "cosine":        "Cosine Similarity",
    "token_overlap": "Token Overlap",
    "jaccard":       "Jaccard Similarity",
    "bleu1":         "BLEU-1",
    "rouge_l":       "ROUGE-L",
}


def build_report(
    baseline: dict, improved: dict,
    baseline_samples: list[dict], improved_samples: list[dict],
    meta: dict,  # report metadata (n, llm, res_path, embed)
) -> str:
    lines: list[str] = []

    def hr(ch="=", w=72):
        lines.append(ch * w)

    def h(text):
        lines.append(f"\n{text}")
        lines.append("-" * len(text))

    hr()
    lines.append("  RAG EVALUATION REPORT")
    lines.append(f"  Date            : {date.today()}")
    lines.append(f"  RES dataset     : {meta['res_path']}")
    lines.append(f"  Questions tested: {meta['n']}")
    lines.append(f"  Embedding model : {meta['embed']}")
    lines.append(f"  LLM             : {meta['llm']}")
    hr()

    h("SYSTEM CONFIGURATIONS")
    lines.append("  Baseline : Dense FAISS top-5 (no BM25 | no reranker | "
                 "no freshness | no parent-child)")
    lines.append("  Improved : Hybrid BM25+FAISS (RRF) + Cross-encoder reranker "
                 "+ Freshness scoring + Parent-child retrieval")

    h("METRIC COMPARISON")
    col_w = 22
    lines.append(f"  {'Metric':<{col_w}} {'Baseline':>10} {'Improved':>10} {'Delta':>10}")
    lines.append(f"  {'-'*col_w} {'':->10} {'':->10} {'':->10}")
    for key, label in METRIC_LABELS.items():
        b = baseline.get(key, 0.0)
        im = improved.get(key, 0.0)
        delta = im - b
        sign  = "+" if delta >= 0 else ""
        lines.append(f"  {label:<{col_w}} {b:>10.4f} {im:>10.4f} {sign}{delta:>9.4f}")

    h("MARKDOWN TABLE (for report paste)")
    lines.append("| Metric | Baseline | Improved | Delta |")
    lines.append("|---|---:|---:|---:|")
    for key, label in METRIC_LABELS.items():
        b = baseline.get(key, 0.0)
        im = improved.get(key, 0.0)
        delta = im - b
        sign = "+" if delta >= 0 else ""
        lines.append(f"| {label} | {b:.4f} | {im:.4f} | {sign}{delta:.4f} |")

    h("SAMPLE ANSWERS — BASELINE (first 5 questions)")
    for i, s in enumerate(baseline_samples, 1):
        lines.append(f"\n  [{i}] Q: {s['question'][:120]}")
        lines.append(f"      REF  : {s['reference'][:200]}")
        lines.append(f"      PRED : {s['prediction'][:200]}")
        lines.append(f"      cosine={s['cosine']:.3f}  BLEU={s['bleu1']:.3f}  "
                     f"ROUGE-L={s['rouge_l']:.3f}")

    h("SAMPLE ANSWERS — IMPROVED (first 5 questions)")
    for i, s in enumerate(improved_samples, 1):
        lines.append(f"\n  [{i}] Q: {s['question'][:120]}")
        lines.append(f"      REF  : {s['reference'][:200]}")
        lines.append(f"      PRED : {s['prediction'][:200]}")
        lines.append(f"      cosine={s['cosine']:.3f}  BLEU={s['bleu1']:.3f}  "
                     f"ROUGE-L={s['rouge_l']:.3f}")

    hr()
    lines.append("  END OF REPORT")
    hr()
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate baseline vs improved RAG report.")
    p.add_argument("--llm",  default=LLM_MODEL,
                   help=f"vllm:<model> | gemini-* | HF repo id (default: {LLM_MODEL})")
    p.add_argument("--res",  default=RES_CSV,
                   help=f"Path to RES CSV (default: {RES_CSV})")
    p.add_argument("--n",    type=int, default=200,
                   help="Max questions to evaluate (min 100 used)")
    p.add_argument("--out",  default="report.txt",
                   help="Output .txt file (default: report.txt)")
    p.add_argument("--quant", choices=["none", "4bit", "8bit"], default="none")
    p.add_argument("--no-filter", action="store_true",
                   help="Don't filter to docs-in-dataset; use all rows.")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Load data ──────────────────────────────────────────────────────────────
    res_path = Path(args.res)
    if not res_path.is_absolute():
        # try project root first, then cwd
        project_root = Path(__file__).resolve().parent.parent
        candidate = project_root / args.res
        if candidate.exists():
            res_path = candidate

    print(f"Loading RES dataset: {res_path}")
    df = load_res(str(res_path))
    print(f"  Loaded {len(df)} rows")

    # ── Load retriever + metadata ──────────────────────────────────────────────
    from retrieval.retriever import HybridRetriever
    retriever = HybridRetriever()

    if not args.no_filter:
        df = filter_to_dataset(df, retriever.meta)
        print(f"  After filtering to docs in dataset: {len(df)} rows")

    n = max(100, min(args.n, len(df)))
    print(f"  Will evaluate: {n} questions")

    if len(df) < 100:
        print("WARNING: fewer than 100 questions available after filtering. "
              "Add more documents or use --no-filter.")

    # ── Load LLM ──────────────────────────────────────────────────────────────
    print(f"Loading LLM: {args.llm} …")
    from generation.llm_providers import make_llm
    llm_fn = make_llm(args.llm, quant=args.quant)

    from generation.pipeline import RAGPipeline
    pipeline = RAGPipeline(llm=llm_fn, retriever=retriever)

    embed_fn = lambda texts: retriever.embed_model.encode(
        texts, normalize_embeddings=True, convert_to_numpy=True
    )

    # ── Run evaluations ────────────────────────────────────────────────────────
    baseline_kwargs = dict(
        use_bm25=False, use_reranker=False,
        use_freshness=False, explicit_boost=False, use_parents=False,
    )
    improved_kwargs = dict(
        use_bm25=True, use_reranker=True,
        use_freshness=True, explicit_boost=True, use_parents=True,
    )

    print("\n[1/2] Evaluating BASELINE …")
    baseline_scores, baseline_samples = run_eval(
        pipeline, embed_fn, df, n, baseline_kwargs, "Baseline"
    )

    print("\n[2/2] Evaluating IMPROVED system …")
    improved_scores, improved_samples = run_eval(
        pipeline, embed_fn, df, n, improved_kwargs, "Improved"
    )

    # ── Build & save report ────────────────────────────────────────────────────
    report_meta = {
        "res_path": str(res_path.name),
        "n": n,
        "embed": EMBED_MODEL,
        "llm": args.llm,
    }
    report = build_report(
        baseline_scores, improved_scores,
        baseline_samples, improved_samples,
        report_meta,
    )

    out_path = Path(args.out)
    out_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved → {out_path.resolve()}")
    print("\n" + report)


if __name__ == "__main__":
    main()
