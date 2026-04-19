"""
query_analyzer.py - Lightweight regex-based intent detection for user queries.

Extracts three signals without calling any LLM:
  - prefers_current : user said "hiện hành"/"mới nhất" → boost in-force docs
  - explicit_so_hieu: user named a văn bản (e.g. "50/2024/TT-NHNN") → always retrieve it
  - from_date       : "từ ngày DD/MM/YYYY" cutoff

Kept separate from retriever.py so the report can document intent extraction
as its own concern and so rules are easy to extend.
"""
from __future__ import annotations

import re

# "50/2024/TT-NHNN", "94/2025/NĐ-CP", "23-TT/LB", "1370/QĐ-NHNN"
_SO_HIEU_RE = re.compile(
    r"\b(\d{1,4}[/-]\d{2,4}[/-][A-ZĐ]+(?:[-/][A-ZĐ0-9]+)*)\b"
)
_FROM_DATE_RE = re.compile(r"từ\s+ngày\s+(\d{1,2}/\d{1,2}/\d{4})", re.IGNORECASE)
_CURRENT_LAW_KW = ("hiện hành", "mới nhất", "đang có hiệu lực", "còn hiệu lực")


def analyze_query(q: str) -> dict:
    """Return {prefers_current, explicit_so_hieu, from_date}."""
    ql = q.lower()
    so_hieu_hits = [m.group(1) for m in _SO_HIEU_RE.finditer(q)]
    from_date = None
    m = _FROM_DATE_RE.search(q)
    if m:
        from_date = m.group(1)
    return {
        "prefers_current":  any(k in ql for k in _CURRENT_LAW_KW),
        "explicit_so_hieu": [s.lower() for s in so_hieu_hits],
        "from_date":        from_date,
    }


def tokenize_vi(text: str) -> list[str]:
    """Unicode-aware word tokenizer shared by BM25 and metric code."""
    return re.findall(r"[\w\u00C0-\u024F\u1E00-\u1EFF]+", text.lower())
