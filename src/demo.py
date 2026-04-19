"""
demo.py - One-shot CLI for report screenshots and sanity checks.

Takes a single question, prints the query analysis, the top-k retrieved
chunks (with so_hieu / status / score / freshness note), and optionally the
LLM answer. Output is plain text so it screenshots cleanly.

Examples:
    # retrieval only (no LLM, fast)
    python src/demo.py "Quy định về Soft OTP mới nhất?"

    # with Gemini (needs GEMINI_API_KEY in .env)
    python src/demo.py --llm gemini-2.5-flash "Quy định về Soft OTP mới nhất?"

    # with a local HF model, 4-bit quantised
    python src/demo.py --llm Qwen/Qwen2.5-7B-Instruct --quant 4bit "…"
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

from config import TOP_K, LLM_MODEL
from generation.pipeline      import RAGPipeline
from retrieval.query_analyzer import analyze_query
from retrieval.retriever      import HybridRetriever


def _wrap(text: str, width: int = 100, indent: str = "    ") -> str:
    return textwrap.fill(
        text, width=width, initial_indent=indent, subsequent_indent=indent,
    )


def _print_contexts(contexts: list[dict], preview_chars: int) -> None:
    print(f"\n=== RETRIEVED CONTEXTS (top {len(contexts)}) ===")
    for i, ch in enumerate(contexts, 1):
        so_hieu = ch.get("so_hieu") or "(no so_hieu)"
        status  = ch.get("_status") or "unknown"
        score   = ch.get("_score", 0.0)
        rerank  = ch.get("_rerank", 0.0)
        fresh   = ch.get("_freshness", 1.0)
        note    = ch.get("_freshness_note") or ""
        tieu_de = (ch.get("tieu_de") or "").strip()

        print(f"\n[{i}] {so_hieu}   status={status}")
        print(f"    score={score:.3f}   rerank={rerank:.3f}   freshness×={fresh:.2f}")
        if tieu_de:
            print(_wrap(f"Tiêu đề: {tieu_de}"))
        if note:
            print(f"    ⚠ {note}")
        preview = (ch.get("text") or "").replace("\n", " ")
        if len(preview) > preview_chars:
            preview = preview[:preview_chars] + " …"
        print(_wrap(preview))


def parse_args():
    p = argparse.ArgumentParser(
        description="One-shot RAG demo for a single question.",
    )
    p.add_argument("question", help="The user question (wrap in quotes).")
    p.add_argument("--llm",     default=None,
                   help=f"Optional LLM. vllm:<model> | gemini-* | HF repo id. "
                        f"Omit for retrieval-only. (config default: {LLM_MODEL})")
    p.add_argument("--quant",   choices=["none", "4bit", "8bit"], default="none")
    p.add_argument("--top-k",   type=int, default=TOP_K)
    p.add_argument("--preview", type=int, default=350,
                   help="Chars of chunk text to preview. Use 0 for full text.")
    return p.parse_args()


def main():
    args = parse_args()

    print("=== QUERY ===")
    print(f"  {args.question}")

    analysis = analyze_query(args.question)
    print("\n=== QUERY ANALYSIS ===")
    print(f"  prefers_current : {analysis['prefers_current']}")
    print(f"  explicit_so_hieu: {analysis['explicit_so_hieu'] or '(none)'}")
    print(f"  from_date       : {analysis['from_date'] or '(none)'}")

    retriever = HybridRetriever()

    llm_fn = None
    if args.llm:
        print(f"\nLoading LLM: {args.llm} (quant={args.quant}) …")
        from generation.llm_providers import make_llm
        llm_fn = make_llm(args.llm, quant=args.quant)

    pipeline = RAGPipeline(llm=llm_fn, retriever=retriever)
    result = pipeline.answer(args.question, top_k=args.top_k)

    preview = len(result["contexts"][0]["text"]) if args.preview == 0 \
              and result["contexts"] else args.preview
    _print_contexts(result["contexts"], preview)

    if llm_fn:
        print("\n=== LLM ANSWER ===")
        print(result["response"] or "(empty)")
        cit = result.get("citations")
        if cit:
            from generation.citation import format_report
            print("\n=== CITATION CHECK ===")
            print(format_report(cit))
    else:
        print("\n(no --llm supplied; retrieval-only mode)")


if __name__ == "__main__":
    main()
