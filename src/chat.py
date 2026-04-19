"""
chat.py - Interactive Q&A chatbot loop (one session, multi-turn).

Usage:
    python src/chat.py --llm vllm:Qwen2.5-14B-Instruct
    python src/chat.py --llm vllm:Qwen2.5-14B-Instruct --no-citation
    python src/chat.py --llm gemini-2.5-flash

Commands inside the chat:
    /quit or /exit   -- end session
    /clear           -- clear screen
    /retrieval       -- toggle showing retrieved chunks
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).parent))

from config import TOP_K, LLM_MODEL
from generation.llm_providers import make_llm
from generation.pipeline import RAGPipeline
from retrieval.retriever import HybridRetriever


def _hr(char: str = "─", width: int = 72) -> str:
    return char * width


def _wrap(text: str, width: int = 90, indent: str = "  ") -> str:
    return textwrap.fill(text, width=width, initial_indent=indent,
                         subsequent_indent=indent)


def _print_contexts(contexts: list[dict]) -> None:
    print(f"\n  [{len(contexts)} chunks retrieved]")
    for i, ch in enumerate(contexts, 1):
        so_hieu = ch.get("so_hieu") or "(no id)"
        tieu_de = (ch.get("tieu_de") or "").strip()
        loai    = ch.get("loai_van_ban") or ""
        co_quan = ch.get("co_quan_ban_hanh") or ""
        status  = ch.get("_status") or "?"
        score   = ch.get("_score", 0.0)
        note    = ch.get("_freshness_note") or ""
        preview = (ch.get("text") or "").replace("\n", " ")[:200]

        print(f"  [{i}] {so_hieu}  status={status}  score={score:.3f}")
        if loai or co_quan:
            print(f"      Loại: {loai} | Ban hành: {co_quan}")
        if tieu_de:
            print(f"      Tên: {tieu_de[:120]}")
        if note:
            print(f"      ⚠ {note}")
        print(f"      ↳ {preview} …")


def parse_args():
    p = argparse.ArgumentParser(description="Interactive RAG law chatbot.")
    p.add_argument("--llm", default=LLM_MODEL,
                   help=f"LLM to use. vllm:<model> | gemini-* | HF repo id (default: {LLM_MODEL})")
    p.add_argument("--quant", choices=["none", "4bit", "8bit"], default="none")
    p.add_argument("--top-k", type=int, default=TOP_K)
    p.add_argument("--no-citation", action="store_true",
                   help="Skip citation verification output.")
    return p.parse_args()


def main():
    args = parse_args()

    print(_hr("="))
    print("  RAG LAW CHATBOT")
    print(f"  LLM : {args.llm}")
    print(f"  Top-k: {args.top_k}")
    print("  Commands: /quit  /retrieval  /clear")
    print(_hr("="))

    print("\nLoading retriever …")
    retriever = HybridRetriever()

    print(f"Loading LLM: {args.llm} …")
    llm_fn = make_llm(args.llm, quant=args.quant)

    pipeline = RAGPipeline(llm=llm_fn, retriever=retriever)

    show_retrieval = False
    print("\nSẵn sàng. Hãy đặt câu hỏi.\n")

    while True:
        try:
            query = input("Bạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nThoát.")
            break

        if not query:
            continue

        cmd = query.lower()
        if cmd in ("/quit", "/exit", "quit", "exit"):
            print("Thoát.")
            break
        if cmd == "/clear":
            os.system("cls" if os.name == "nt" else "clear")
            continue
        if cmd == "/retrieval":
            show_retrieval = not show_retrieval
            print(f"  [retrieval display: {'ON' if show_retrieval else 'OFF'}]")
            continue

        print()
        # Retrieve first so we can show chunks before the (slow) LLM call
        contexts = pipeline.retriever.retrieve(query, top_k=args.top_k)
        if show_retrieval:
            _print_contexts(contexts)

        result = pipeline.answer(query, top_k=args.top_k, contexts=contexts)

        print(_hr())
        answer = result.get("response") or "(Không có câu trả lời)"
        for line in answer.splitlines():
            print(_wrap(line) if line.strip() else "")

        if not args.no_citation:
            cit = result.get("citations")
            if cit and cit.get("total", 0) > 0:
                from generation.citation import format_report
                print(f"\n  {format_report(cit)}")

        print(_hr())
        print()


if __name__ == "__main__":
    main()
