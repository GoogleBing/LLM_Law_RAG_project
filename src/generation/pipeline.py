"""
pipeline.py - RAGPipeline: wires retriever + prompt builder + LLM.

Kept deliberately thin so each responsibility lives in its own module and can
be swapped (e.g. new prompt format, different LLM provider) without touching
the retrieval code.
"""
from __future__ import annotations

from typing import Callable, Optional

from config import TOP_K
from generation.citation import verify_citations
from generation.prompt_builder import build_prompt
from retrieval.query_analyzer import analyze_query
from retrieval.retriever import HybridRetriever


class RAGPipeline:
    def __init__(
        self,
        llm: Optional[Callable[[str], str]] = None,
        lazy_reranker: bool = True,
        retriever: Optional[HybridRetriever] = None,
    ):
        self.retriever = retriever or HybridRetriever(lazy_reranker=lazy_reranker)
        self.llm = llm

    def answer(self, query: str, top_k: int = TOP_K,
               retrieve_kwargs: Optional[dict] = None,
               contexts: Optional[list] = None) -> dict:
        if contexts is None:
            kw = retrieve_kwargs or {}
            contexts = self.retriever.retrieve(query, top_k, **kw)
        prompt   = build_prompt(query, contexts)
        response = self.llm(prompt) if self.llm else ""
        result = {
            "query":    query,
            "contexts": contexts,
            "prompt":   prompt,
            "response": response,
            "analysis": analyze_query(query),
        }
        if response:
            result["citations"] = verify_citations(response, contexts)
        return result
