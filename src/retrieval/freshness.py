"""
freshness.py - Status-aware weights for the retrieval re-scoring step.

The Vietnamese legal corpus mixes active, expired, and outdated documents.
Rather than hard-filtering (which would hide legitimate historical queries),
we multiply each candidate's rerank score by a freshness factor derived from
its `tinh_trang` field. Two regimes:

  - neutral (base)  : the user did not signal a preference → gentle penalty
                      on expired docs, still retrievable.
  - strict (current): the user said "hiện hành"/"mới nhất" → expired docs
                      are pushed far down but not zeroed (explicit so_hieu
                      can still rescue them via the boost in retriever.py).

Weights are intentionally smooth multipliers in (0, 1] so RRF order is
preserved when all candidates share a status.
"""
from __future__ import annotations

# Re-export status constants so consumers don't have to import doc_metadata
from indexing.doc_metadata import (  # noqa: F401
    STATUS_ACTIVE, STATUS_EXPIRED, STATUS_OUTDATED, STATUS_UNKNOWN,
)

_BASE_WEIGHT = {
    STATUS_ACTIVE:   1.00,
    STATUS_UNKNOWN:  0.80,
    STATUS_OUTDATED: 0.60,
    STATUS_EXPIRED:  0.55,
}
_STRICT_WEIGHT = {
    STATUS_ACTIVE:   1.00,
    STATUS_UNKNOWN:  0.55,
    STATUS_OUTDATED: 0.25,
    STATUS_EXPIRED:  0.15,
}


def freshness_weight(status: str, prefers_current: bool) -> float:
    table = _STRICT_WEIGHT if prefers_current else _BASE_WEIGHT
    return table.get(status, _BASE_WEIGHT[STATUS_UNKNOWN])
