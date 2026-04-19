"""
evaluate_rag.py - Evaluate RAG pipeline on RES dataset.

Modes:
  --mode retrieval   Hit@{1,3,5} by so_hieu (no LLM needed).
  --mode ablation    Hit@{1,3,5} across 5 incremental retrieval configs.
  --mode full        Cosine / Token Overlap / Jaccard / BLEU-1 / ROUGE-L
                     + citation accuracy. Requires --llm.
  --mode compare     Baseline vs Improved Hit@k (no LLM needed).
                     Add --llm to compare generation metrics instead.

Output:
  Results are always saved to --out (default: eval_<mode>_<n>.txt).
  Pass --no-save to print only.

Examples:
    python src/evaluate_rag.py --mode retrieval --n 200
    python src/evaluate_rag.py --mode ablation  --n 100
    python src/evaluate_rag.py --mode full      --n 100 --llm vllm:Qwen2.5-14B-Instruct
    python src/evaluate_rag.py --mode compare   --n 200
    python src/evaluate_rag.py --mode compare   --n 100 --llm vllm:Qwen2.5-14B-Instruct
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from datetime import date
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
        ("Dense only",     dict(use_bm25=False, use_reranker=False,
                                use_freshness=False, explicit_boost=False)),
        ("+ BM25 (RRF)",   dict(use_bm25=True,  use_reranker=False,
                                use_freshness=False, explicit_boost=False)),
        ("+ Reranker",     dict(use_bm25=True,  use_reranker=True,
                                use_freshness=False, explicit_boost=False)),
        ("+ Freshness",    dict(use_bm25=True,  use_reranker=True,
                                use_freshness=True,  explicit_boost=False)),
        ("+ Boost (full)", dict(use_bm25=True,  use_reranker=True,
                                use_freshness=True,  explicit_boost=True)),
    ]
    rows = []
    for name, kw in configs:
        print(f"\n--- {name} ---")
        rows.append({"config": name, **_hits_for(retriever, df, n, kw)})
    return rows


def _print_retrieval_table(rows: list[dict]) -> list[str]:
    lines = []
    lines.append(f"\n{'Config':<22} {'Hit@1':>7} {'Hit@3':>7} {'Hit@5':>7}")
    lines.append("-" * 45)
    for r in rows:
        lines.append(f"  {r['config']:<20} {r['Hit@1']:>7.3f} {r['Hit@3']:>7.3f} {r['Hit@5']:>7.3f}")
    lines.append("\nMarkdown:")
    lines.append("| Config | Hit@1 | Hit@3 | Hit@5 |")
    lines.append("|---|---:|---:|---:|")
    for r in rows:
        lines.append(f"| {r['config']} | {r['Hit@1']:.3f} | {r['Hit@3']:.3f} | {r['Hit@5']:.3f} |")
    return lines


# ── Generation metrics ────────────────────────────────────────────────────────

METRIC_LABELS = {
    "cosine":             "Cosine Similarity",
    "token_overlap":      "Token Overlap",
    "jaccard":            "Jaccard Similarity",
    "bleu1":              "BLEU-1",
    "rouge_l":            "ROUGE-L",
    "citation_accuracy":  "Citation Accuracy",
    "citation_coverage":  "Citation Coverage",
}


def _eval_generation(pipeline, embed_fn, df: pd.DataFrame, n: int,
                     retrieve_kwargs: dict, label: str) -> tuple[dict, list[dict]]:
    agg: dict[str, list] = {k: [] for k in METRIC_LABELS}
    samples: list[dict] = []

    for _, row in tqdm(df.head(n).iterrows(), total=min(n, len(df)), desc=label):
        result = pipeline.answer(str(row["question"]),
                                 retrieve_kwargs=retrieve_kwargs)
        pred = result.get("response") or ""
        ref  = str(row["reference"])

        agg["cosine"].append(cosine_sim(pred, ref, embed_fn))
        agg["token_overlap"].append(token_overlap(pred, ref))
        agg["jaccard"].append(jaccard(pred, ref))
        agg["bleu1"].append(bleu1(pred, ref))
        agg["rouge_l"].append(rouge_l(pred, ref))
        cit = result.get("citations") or {}
        agg["citation_accuracy"].append(cit.get("score", 0.0))
        agg["citation_coverage"].append(1.0 if cit.get("total", 0) > 0 else 0.0)

        if len(samples) < 5:
            samples.append({"question": str(row["question"]),
                            "reference": ref, "prediction": pred})

    return {k: float(np.mean(v)) for k, v in agg.items()}, samples


def eval_full(pipeline, embed_fn, df: pd.DataFrame, n: int) -> tuple[dict, list[dict]]:
    return _eval_generation(pipeline, embed_fn, df, n, {}, "full eval")


def eval_compare_retrieval(retriever, df: pd.DataFrame, n: int) -> tuple[dict, dict]:
    baseline_kw = dict(use_bm25=False, use_reranker=False,
                       use_freshness=False, explicit_boost=False, use_parents=False)
    improved_kw = dict(use_bm25=True,  use_reranker=True,
                       use_freshness=True,  explicit_boost=True,  use_parents=True)
    print("\n[1/2] Evaluating BASELINE retrieval ...")
    b = _hits_for(retriever, df, n, baseline_kw)
    print("\n[2/2] Evaluating IMPROVED retrieval ...")
    im = _hits_for(retriever, df, n, improved_kw)
    return b, im


def _build_compare_retrieval_report(b: dict, im: dict, n: int) -> list[str]:
    hr = "=" * 52
    lines = [hr, "  RETRIEVAL COMPARISON (Baseline vs Improved)",
             f"  n = {n}", hr, ""]
    lines += [f"  {'Config':<22} {'Hit@1':>7} {'Hit@3':>7} {'Hit@5':>7}",
              "  " + "-" * 45,
              f"  {'Baseline':<22} {b['Hit@1']:>7.3f} {b['Hit@3']:>7.3f} {b['Hit@5']:>7.3f}",
              f"  {'Improved':<22} {im['Hit@1']:>7.3f} {im['Hit@3']:>7.3f} {im['Hit@5']:>7.3f}"]
    delta = {k: im[k] - b[k] for k in ("Hit@1", "Hit@3", "Hit@5")}
    lines.append(f"  {'Delta':<22} {delta['Hit@1']:>+7.3f} {delta['Hit@3']:>+7.3f} {delta['Hit@5']:>+7.3f}")
    lines += ["", "Markdown:",
              "| Config | Hit@1 | Hit@3 | Hit@5 |",
              "|---|---:|---:|---:|",
              f"| Baseline | {b['Hit@1']:.3f} | {b['Hit@3']:.3f} | {b['Hit@5']:.3f} |",
              f"| Improved | {im['Hit@1']:.3f} | {im['Hit@3']:.3f} | {im['Hit@5']:.3f} |",
              f"| **Delta** | **{delta['Hit@1']:+.3f}** | **{delta['Hit@3']:+.3f}** | **{delta['Hit@5']:+.3f}** |"]
    return lines


def eval_compare(pipeline, embed_fn, df: pd.DataFrame, n: int) -> tuple[dict, dict, list[dict], list[dict]]:
    baseline_kw = dict(use_bm25=False, use_reranker=False,
                       use_freshness=False, explicit_boost=False, use_parents=False)
    improved_kw = dict(use_bm25=True,  use_reranker=True,
                       use_freshness=True,  explicit_boost=True,  use_parents=True)
    print("\n[1/2] Evaluating BASELINE ...")
    b_scores, b_samples = _eval_generation(pipeline, embed_fn, df, n, baseline_kw, "Baseline")
    print("\n[2/2] Evaluating IMPROVED ...")
    i_scores, i_samples = _eval_generation(pipeline, embed_fn, df, n, improved_kw, "Improved")
    return b_scores, i_scores, b_samples, i_samples


# ── Report builders ───────────────────────────────────────────────────────────

def _build_retrieval_report(r: dict, n: int, mode: str) -> list[str]:
    lines = ["=== RETRIEVAL HIT RATE ==="]
    for k in ("Hit@1", "Hit@3", "Hit@5"):
        lines.append(f"  {k}: {r[k]:.3f}")
    lines.append(f"  Evaluated: {r['n']} questions")
    return lines


def _build_ablation_report(rows: list[dict], n: int) -> list[str]:
    lines = ["=== ABLATION STUDY ==="]
    lines += _print_retrieval_table(rows)
    return lines


def _build_full_report(scores: dict, samples: list[dict], n: int, llm: str) -> list[str]:
    lines = ["=== FULL RAG EVALUATION ===",
             f"  LLM : {llm}  |  n = {n}",
             "",
             f"  {'Metric':<22} {'Score':>8}",
             "  " + "-" * 32]
    for k, label in METRIC_LABELS.items():
        lines.append(f"  {label:<22} {scores.get(k, 0):>8.4f}")
    return lines


def _build_compare_report(b: dict, im: dict,
                           b_samples: list[dict], im_samples: list[dict],
                           n: int, llm: str) -> list[str]:
    lines = []
    hr = "=" * 72

    lines += [hr, "  RAG COMPARISON REPORT",
              f"  Date  : {date.today()}",
              f"  LLM   : {llm}",
              f"  n     : {n}",
              hr, ""]

    lines += ["SYSTEM CONFIGURATIONS",
              "  Baseline : Dense FAISS top-5 (no BM25 | no reranker | no freshness | no parent-child)",
              "  Improved : Hybrid BM25+FAISS (RRF) + Cross-encoder reranker + Freshness + Parent-child",
              ""]

    col = 22
    lines += ["METRIC COMPARISON",
              f"  {'Metric':<{col}} {'Baseline':>10} {'Improved':>10} {'Delta':>10}",
              f"  {'-'*col} {'':->10} {'':->10} {'':->10}"]
    for key, label in METRIC_LABELS.items():
        bv = b.get(key, 0.0)
        iv = im.get(key, 0.0)
        delta = iv - bv
        sign = "+" if delta >= 0 else ""
        lines.append(f"  {label:<{col}} {bv:>10.4f} {iv:>10.4f} {sign}{delta:>9.4f}")

    lines += ["", "MARKDOWN TABLE",
              "| Metric | Baseline | Improved | Delta |",
              "|---|---:|---:|---:|"]
    for key, label in METRIC_LABELS.items():
        bv = b.get(key, 0.0)
        iv = im.get(key, 0.0)
        delta = iv - bv
        sign = "+" if delta >= 0 else ""
        lines.append(f"| {label} | {bv:.4f} | {iv:.4f} | {sign}{delta:.4f} |")

    for tag, slist in [("BASELINE", b_samples), ("IMPROVED", im_samples)]:
        lines += ["", f"SAMPLE ANSWERS — {tag} (first 5 questions)", "-" * 50]
        for i, s in enumerate(slist, 1):
            lines += [f"\n  [{i}] Q: {s['question'][:120]}",
                      f"      REF : {s['reference'][:200]}",
                      f"      PRED: {s['prediction'][:200]}"]

    lines += ["", hr, "  END OF REPORT", hr]
    return lines


# ── Output helper ─────────────────────────────────────────────────────────────

def _save_and_print(lines: list[str], out_path: str | None) -> None:
    text = "\n".join(lines)
    print(text)
    if out_path:
        Path(out_path).write_text(text, encoding="utf-8")
        print(f"\nSaved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate RAG pipeline on RES dataset.")
    p.add_argument("--mode", choices=["retrieval", "ablation", "full", "compare"],
                   default="retrieval")
    p.add_argument("--n",    type=int, default=200,
                   help="Max questions to evaluate")
    p.add_argument("--llm",  default=None,
                   help="LLM for --mode full/compare with generation metrics. "
                        "Omit to run --mode compare as retrieval-only (no LLM needed).")
    p.add_argument("--quant", choices=["none", "4bit", "8bit"], default="none")
    p.add_argument("--out",  default=None,
                   help="Output file path (default: eval_<mode>_<n>.txt)")
    p.add_argument("--no-save", action="store_true",
                   help="Print only, do not save to file")
    return p.parse_args()


def main():
    args = parse_args()

    out_path = None
    if not args.no_save:
        out_path = args.out or f"eval_{args.mode}_{args.n}.txt"

    from retrieval.retriever import HybridRetriever
    retriever = HybridRetriever()

    print(f"\nLoading dataset: {RES_PATH}")
    df = load_and_filter(RES_PATH, retriever.meta)
    n  = min(args.n, len(df))

    if args.mode == "retrieval":
        r = eval_retrieval(retriever, df, n)
        lines = _build_retrieval_report(r, n, args.mode)
        _save_and_print(lines, out_path)

    elif args.mode == "ablation":
        rows = eval_ablation(retriever, df, n)
        lines = _build_ablation_report(rows, n)
        _save_and_print(lines, out_path)

    elif args.mode == "compare" and args.llm is None:
        b, im = eval_compare_retrieval(retriever, df, n)
        lines = _build_compare_retrieval_report(b, im, n)
        _save_and_print(lines, out_path)

    else:
        llm_name = args.llm or "vllm:Qwen2.5-14B-Instruct"
        from generation.llm_providers import make_llm
        from generation.pipeline import RAGPipeline
        print(f"\nLoading LLM: {llm_name} ...")
        llm_fn   = make_llm(llm_name, quant=args.quant)
        embed_fn = lambda texts: retriever.embed_model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True)
        pipeline = RAGPipeline(llm=llm_fn, retriever=retriever)

        if args.mode == "full":
            scores, samples = eval_full(pipeline, embed_fn, df, n)
            lines = _build_full_report(scores, samples, n, llm_name)
            _save_and_print(lines, out_path)

        else:  # compare
            b, im, b_samp, im_samp = eval_compare(pipeline, embed_fn, df, n)
            lines = _build_compare_report(b, im, b_samp, im_samp, n, args.llm)
            _save_and_print(lines, out_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
