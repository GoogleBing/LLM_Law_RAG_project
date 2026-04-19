"""
rag_pipeline.py - Thin backward-compat facade.

The pipeline lives in three stage packages:
  - indexing/   : chunker, doc_metadata
  - retrieval/  : query_analyzer, freshness, retriever
  - generation/ : prompt_builder, llm_providers, pipeline

This file re-exports public symbols so older callers still work.
"""
from config                    import TOP_K  # noqa: F401
from generation.pipeline        import RAGPipeline  # noqa: F401
from generation.prompt_builder  import SYSTEM_INSTRUCTION, build_prompt  # noqa: F401
from retrieval.query_analyzer   import analyze_query  # noqa: F401
from retrieval.retriever        import HybridRetriever  # noqa: F401
