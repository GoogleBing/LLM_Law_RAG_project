"""
doc_metadata.py - Sidecar metadata lookup for documents.

The existing index/chunks.jsonl doesn't carry `tinh_trang`, `ngay_hieu_luc`, or
`van_ban_lien_quan`. Rather than reindexing, this module reads them from
parsed_documents.jsonl once, caches to a pickle, and joins at retrieval time.

Provides:
  - DocMetadata.get(so_hieu) -> dict with status, effective_date, replaces, ...
  - DocMetadata.status(so_hieu) -> one of {"active", "expired", "outdated", "unknown"}
  - DocMetadata.effective_year(so_hieu) -> int | None
"""
from __future__ import annotations

import json
import os
import pickle
import re
from pathlib import Path
from typing import Optional

_DATE_RE = re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4})")

# Copy of the regex in retrieval.query_analyzer so this module stays in
# indexing/ with no upward dependency on retrieval/.
_SO_HIEU_RE = re.compile(
    r"\b(\d{1,4}[/-]\d{2,4}[/-][A-ZĐ]+(?:[-/][A-ZĐ0-9]+)*)\b"
)

STATUS_ACTIVE   = "active"
STATUS_EXPIRED  = "expired"
STATUS_OUTDATED = "outdated"
STATUS_UNKNOWN  = "unknown"


def _normalize_so_hieu(s: str) -> str:
    return (s or "").strip().lower()


def _parse_ddmmyyyy(s: str) -> Optional[tuple[int, int, int]]:
    if not s:
        return None
    m = _DATE_RE.search(s)
    if not m:
        return None
    d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if not (1 <= d <= 31 and 1 <= mo <= 12 and 1900 <= y <= 2100):
        return None
    return (y, mo, d)


def _parse_status(raw: str) -> tuple[str, Optional[tuple[int, int, int]]]:
    """Return (status_code, expire_date_or_None)."""
    if not raw:
        return STATUS_UNKNOWN, None
    r = raw.strip()
    rl = r.lower()
    if "còn hiệu lực" in rl:
        return STATUS_ACTIVE, None
    if rl.startswith("hết hiệu lực"):
        return STATUS_EXPIRED, _parse_ddmmyyyy(r)
    if "không còn phù hợp" in rl:
        return STATUS_OUTDATED, None
    return STATUS_UNKNOWN, None


class DocMetadata:
    """Loads per-document metadata from parsed_documents.jsonl (cached to pickle)."""

    # Bumped from 1 → 2 when the _replaced_by inverse map was added so old
    # caches are transparently rebuilt on first run.
    CACHE_VERSION = 2

    def __init__(self, jsonl_path: str, cache_path: Optional[str] = None):
        self.jsonl_path = jsonl_path
        if cache_path is None:
            cache_path = os.path.join(os.path.dirname(jsonl_path), "doc_meta.pkl")
        self.cache_path = cache_path
        self._by_so_hieu: dict[str, dict]      = {}
        self._replaced_by: dict[str, list[str]] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "rb") as f:
                    cached = pickle.load(f)
                if cached.get("version") == self.CACHE_VERSION:
                    self._by_so_hieu  = cached["by_so_hieu"]
                    self._replaced_by = cached.get("replaced_by", {})
                    return
            except Exception:
                pass
        self._build_from_jsonl()
        self._save()

    def _build_from_jsonl(self) -> None:
        if not os.path.exists(self.jsonl_path):
            return
        with open(self.jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                so_hieu = _normalize_so_hieu(d.get("so_hieu", ""))
                if not so_hieu:
                    continue
                status, expire = _parse_status(d.get("tinh_trang", ""))
                eff   = _parse_ddmmyyyy(d.get("ngay_hieu_luc", ""))
                banh  = _parse_ddmmyyyy(d.get("ngay_ban_hanh", ""))
                replaces = [
                    r.get("mo_ta", "")
                    for r in (d.get("van_ban_lien_quan") or [])
                    if r.get("loai") == "thay_the"
                ]
                self._by_so_hieu[so_hieu] = {
                    "status":         status,
                    "expire_date":    expire,
                    "effective_date": eff,
                    "issue_date":     banh,
                    "replaces_raw":   replaces,
                    "tinh_trang_raw": d.get("tinh_trang", ""),
                }

                # Build 1-hop inverse map: if this doc's "thay_the" entries
                # name an older so_hieu, that older doc is replaced by this one.
                for desc in replaces:
                    for m in _SO_HIEU_RE.finditer(desc or ""):
                        old = _normalize_so_hieu(m.group(1))
                        if old and old != so_hieu:
                            bucket = self._replaced_by.setdefault(old, [])
                            if so_hieu not in bucket:
                                bucket.append(so_hieu)

    def _save(self) -> None:
        try:
            with open(self.cache_path, "wb") as f:
                pickle.dump(
                    {
                        "version":      self.CACHE_VERSION,
                        "by_so_hieu":   self._by_so_hieu,
                        "replaced_by":  self._replaced_by,
                    },
                    f,
                )
        except Exception:
            pass

    def get(self, so_hieu: str) -> dict:
        return self._by_so_hieu.get(_normalize_so_hieu(so_hieu), {})

    def status(self, so_hieu: str) -> str:
        return self.get(so_hieu).get("status", STATUS_UNKNOWN)

    def effective_year(self, so_hieu: str) -> Optional[int]:
        meta = self.get(so_hieu)
        eff = meta.get("effective_date") or meta.get("issue_date")
        return eff[0] if eff else None

    def replaced_by(self, so_hieu: str) -> list[str]:
        """Return successor so_hieus that replaced this document (1-hop)."""
        return list(self._replaced_by.get(_normalize_so_hieu(so_hieu), []))

    def __len__(self) -> int:
        return len(self._by_so_hieu)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.stdout.reconfigure(encoding="utf-8")
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import PARSED_JSONL
    meta = DocMetadata(PARSED_JSONL)
    print(f"Loaded metadata for {len(meta)} documents.")
    print(f"Inverse replaced_by entries: {len(meta._replaced_by)}")
    for probe in ["10/2021/TT-NHNN", "14/2022/TT-NHNN", "13/2020/TT-NHNN"]:
        m = meta.get(probe)
        print(f"\n  {probe}")
        print(f"    status       : {m.get('status')}")
        print(f"    effective    : {m.get('effective_date')}")
        print(f"    replaces (n) : {len(m.get('replaces_raw') or [])}")
        print(f"    replaced_by  : {meta.replaced_by(probe) or '(none)'}")
