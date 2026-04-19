"""
chunker.py - Parent-child chunker for Vietnamese legal documents.

Two granularities per document:
  - parent  : a full Điều (or a paragraph-packed block for docs without Điều)
  - child   : a smaller sub-segment of a parent (paragraph-level, ~500 chars)
              that BM25 / FAISS actually index.

At retrieval time the pipeline searches children (precise lexical / semantic
match) then swaps in the parent text for the LLM (rich context). This keeps
the reranker pair short while giving the LLM the whole Điều.

Output:
  iter_chunks(jsonl)              → yields child dicts     (backward compat)
  iter_parents(jsonl)             → yields parent dicts
  iter_chunks_and_parents(jsonl)  → yields (children, parents) per doc
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Iterator

# Parent (full Điều) limits — only used for paragraph-fallback packing.
MAX_PARENT_CHARS = 1500
PARENT_OVERLAP   = 150
MIN_PARENT_CHARS = 100

# Child (index unit) limits — smaller for precise retrieval.
MAX_CHILD_CHARS = 500
CHILD_OVERLAP   = 50
MIN_CHILD_CHARS = 40

RE_DIEU = re.compile(
    r"(?:^|\n)(Điều\s+\d+[\.\:]?[^\n]{0,120})",
    re.IGNORECASE | re.UNICODE,
)


def _metadata_prefix(doc: dict) -> str:
    parts = []
    so_hieu = (doc.get("so_hieu") or "").strip()
    tieu_de = (doc.get("tieu_de") or "").strip()
    co_quan = (doc.get("co_quan_ban_hanh") or "").strip()
    loai_vb = (doc.get("loai_van_ban") or "").strip()
    if so_hieu:
        parts.append(f"Số hiệu: {so_hieu}")
    if loai_vb:
        parts.append(f"Loại: {loai_vb}")
    if co_quan:
        parts.append(f"Cơ quan: {co_quan}")
    if tieu_de:
        parts.append(f"Tiêu đề: {tieu_de}")
    return " | ".join(parts)


def _split_by_dieu(text: str) -> list[str]:
    """Split text at Điều X boundaries."""
    positions = [m.start() for m in RE_DIEU.finditer(text)]
    if len(positions) < 2:
        return []
    segments = []
    for i, start in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        segments.append(text[start:end].strip())
    return [s for s in segments if len(s) >= MIN_PARENT_CHARS]


def _split_by_paragraph(text: str) -> list[str]:
    """Paragraph-based splitting with character overlap (parent fallback)."""
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paras:
        if len(current) + len(para) + 2 > MAX_PARENT_CHARS and current:
            chunks.append(current.strip())
            current = current[-PARENT_OVERLAP:] + "\n\n" + para
        else:
            current = (current + "\n\n" + para).lstrip()
    if current.strip():
        chunks.append(current.strip())
    return [c for c in chunks if len(c) >= MIN_PARENT_CHARS]


def _split_into_children(parent_text: str) -> list[str]:
    """Split a parent segment into small children (paragraph-packed)."""
    # Prefer newline-separated paragraphs; fall back to sentence split if the
    # Điều is one long paragraph.
    paras = [p.strip() for p in re.split(r"\n+", parent_text) if p.strip()]
    if len(paras) <= 1:
        paras = [p.strip() for p in re.split(r"(?<=[\.\?\!])\s+", parent_text) if p.strip()]

    children: list[str] = []
    current = ""
    for p in paras:
        if len(current) + len(p) + 1 > MAX_CHILD_CHARS and current:
            children.append(current.strip())
            current = current[-CHILD_OVERLAP:] + " " + p
        else:
            current = (current + "\n" + p).lstrip()
    if current.strip():
        children.append(current.strip())

    # If the whole segment is smaller than MAX_CHILD_CHARS we still want one
    # child covering it verbatim.
    if not children:
        children = [parent_text[:MAX_CHILD_CHARS]]

    return [c for c in children if len(c) >= MIN_CHILD_CHARS] or [parent_text[:MAX_CHILD_CHARS]]


def _base_meta(doc: dict) -> dict:
    return {
        "so_hieu":          doc.get("so_hieu", ""),
        "tieu_de":          doc.get("tieu_de", ""),
        "loai_van_ban":     doc.get("loai_van_ban", ""),
        "co_quan_ban_hanh": doc.get("co_quan_ban_hanh", ""),
        "ngay_ban_hanh":    doc.get("ngay_ban_hanh", ""),
        "_file":            doc.get("_file", ""),
    }


def _file_id(doc: dict) -> str:
    f = doc.get("_file") or (doc.get("so_hieu") or "").replace("/", "_")
    return str(f) or "doc"


def chunk_document(doc: dict) -> dict:
    """Return {'children': [...], 'parents': [...]} for one document."""
    noi_dung = (doc.get("noi_dung") or "").strip()
    if not noi_dung:
        return {"children": [], "parents": []}

    prefix  = _metadata_prefix(doc)
    fid     = _file_id(doc)
    base    = _base_meta(doc)

    segments = _split_by_dieu(noi_dung)
    if not segments:
        segments = _split_by_paragraph(noi_dung)
    if not segments:
        segments = [noi_dung[:MAX_PARENT_CHARS]]

    parents: list[dict]  = []
    children: list[dict] = []

    for pi, seg in enumerate(segments):
        parent_id   = f"{fid}::p{pi}"
        parent_text = f"{prefix}\n\n{seg}" if prefix else seg
        parents.append({
            "parent_id": parent_id,
            "text":      parent_text,
            **base,
        })

        for ci, ct in enumerate(_split_into_children(seg)):
            child_text = f"{prefix}\n\n{ct}" if prefix else ct
            children.append({
                "chunk_id":  f"{parent_id}::c{ci}",
                "parent_id": parent_id,
                "text":      child_text,
                **base,
            })

    return {"children": children, "parents": parents}


def _iter_docs(jsonl_path: str) -> Iterator[dict]:
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def iter_chunks(jsonl_path: str) -> Iterator[dict]:
    """Yield child chunks (what BM25 / FAISS index). Backward-compat name."""
    for d in _iter_docs(jsonl_path):
        yield from chunk_document(d)["children"]


def iter_parents(jsonl_path: str) -> Iterator[dict]:
    """Yield parent records (full Điều text, swapped in at retrieval)."""
    for d in _iter_docs(jsonl_path):
        yield from chunk_document(d)["parents"]


def iter_chunks_and_parents(jsonl_path: str) -> Iterator[tuple[list[dict], list[dict]]]:
    """Yield (children, parents) per document — one pass over the JSONL."""
    for d in _iter_docs(jsonl_path):
        r = chunk_document(d)
        yield r["children"], r["parents"]


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import PARSED_JSONL as jsonl
    total_docs = total_children = total_parents = 0
    for children, parents in iter_chunks_and_parents(jsonl):
        total_docs     += 1
        total_children += len(children)
        total_parents  += len(parents)
    print(f"Docs: {total_docs}  Parents: {total_parents}  Children: {total_children}")
    print(f"Avg parents/doc:  {total_parents/max(1,total_docs):.1f}")
    print(f"Avg children/doc: {total_children/max(1,total_docs):.1f}")
    print(f"Avg children/parent: {total_children/max(1,total_parents):.2f}")
