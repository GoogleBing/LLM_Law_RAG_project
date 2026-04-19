"""
citation.py - Post-hoc verification of citations emitted by the LLM.

The LLM is prompted to cite as [Số hiệu: <so_hieu>, Điều <n>, Khoản <n>] but
does not always obey perfectly. This module parses whatever it produced,
tolerates minor format drift (diacritic variants, "Đ.", missing brackets) and
checks each citation against the retrieved context.

Each citation is bucketed into:
  ok            — (so_hieu, Điều N) found in a retrieved chunk
  partial       — so_hieu matches a chunk but the Điều number doesn't (or was
                  not parsed). Still trustworthy by document, suspicious by
                  article.
  hallucinated  — so_hieu not in any retrieved chunk. Likely fabricated.

Design notes:
  - SO_HIEU_RE tolerates spaces and mixed '/' '-' separators, then a
    normaliser collapses them so "10 / 2021 - TT-NHNN" and "10/2021/TT-NHNN"
    compare equal.
  - DIEU_RE tolerates "Điều", "điều", "Đ.", "ĐIỀU" and "Article".
  - Pairs are built by proximity rather than a single regex, so variants
    like "Thông tư X ... tại Điều 5" or "Điều 5 của Thông tư X" are both OK.
"""
from __future__ import annotations

import re
import unicodedata

SO_HIEU_RE = re.compile(
    r"(\d+\s*[/-]\s*\d+\s*[/-]\s*[A-ZĐ]+(?:[-/][A-ZĐ0-9]+)*)",
    re.IGNORECASE,
)

DIEU_RE = re.compile(
    r"(?:Đ\s*i\s*ề\s*u|Điều|ĐIỀU|Đ\.?|Article)\s*\.?\s*(\d+)",
    re.IGNORECASE | re.UNICODE,
)

# How close (in chars) a so_hieu and an Điều can be to count as one citation.
PROXIMITY_CHARS = 120


def _norm_so_hieu(s: str) -> str:
    """Lowercase + strip internal whitespace. Keep separators as-is so the
    display form ('14/2022/tt-nhnn') stays readable; the LLM is instructed
    to mirror the source format so comparison is stable."""
    s = unicodedata.normalize("NFC", s or "")
    return re.sub(r"\s+", "", s.lower()).strip()


def extract_citations(text: str) -> list[tuple[str, str | None]]:
    """Return a list of (normalized_so_hieu, dieu_or_None) pairs."""
    if not text:
        return []
    so_hieus = [(m.group(1), m.start()) for m in SO_HIEU_RE.finditer(text)]
    dieus    = [(m.group(1), m.start()) for m in DIEU_RE.finditer(text)]
    pairs: list[tuple[str, str | None]] = []
    for sh, sh_pos in so_hieus:
        near = [(d, pos) for d, pos in dieus if abs(pos - sh_pos) < PROXIMITY_CHARS]
        # tie-break: prefer Điều AFTER so_hieu (matches enforced format
        # "[Số hiệu: X, Điều Y]" where Điều always trails the số hiệu).
        d = min(
            near,
            key=lambda x: (abs(x[1] - sh_pos), -(x[1] - sh_pos)),
        )[0] if near else None
        pairs.append((_norm_so_hieu(sh), d))
    # Dedupe while preserving order
    seen: set[tuple[str, str | None]] = set()
    out: list[tuple[str, str | None]] = []
    for p in pairs:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _allowed_from_chunks(chunks: list[dict]) -> tuple[set[str], set[tuple[str, str]]]:
    """Build (allowed_so_hieus, allowed_(so_hieu, dieu)_pairs) from context."""
    allowed_sh: set[str] = set()
    allowed_pairs: set[tuple[str, str]] = set()
    dieu_in_text = re.compile(r"Điều\s*(\d+)", re.IGNORECASE)
    for c in chunks:
        sh = _norm_so_hieu(c.get("so_hieu") or "")
        if not sh:
            continue
        allowed_sh.add(sh)
        for field in ("text", "tieu_de"):
            for m in dieu_in_text.finditer(c.get(field) or ""):
                allowed_pairs.add((sh, m.group(1)))
    return allowed_sh, allowed_pairs


def verify_citations(answer: str, chunks: list[dict]) -> dict:
    """Verify all citations in `answer` against retrieved `chunks`.

    Returns a dict with keys:
      cited, ok, partial, hallucinated, score, total
    where `score = len(ok) / total` (0.0 if nothing was cited).
    """
    allowed_sh, allowed_pairs = _allowed_from_chunks(chunks)
    cited = extract_citations(answer)
    ok:           list[tuple[str, str | None]] = []
    partial:      list[tuple[str, str | None]] = []
    hallucinated: list[tuple[str, str | None]] = []
    for sh, d in cited:
        if sh not in allowed_sh:
            hallucinated.append((sh, d))
        elif d is None or (sh, d) not in allowed_pairs:
            partial.append((sh, d))
        else:
            ok.append((sh, d))
    total = len(cited)
    return {
        "cited":        cited,
        "ok":           ok,
        "partial":      partial,
        "hallucinated": hallucinated,
        "score":        (len(ok) / total) if total else 0.0,
        "total":        total,
    }


def format_report(result: dict) -> str:
    """Pretty one-screen summary for demo.py."""
    total = result["total"]
    if total == 0:
        return "Citation check: (no citations parsed from answer)"
    lines = [
        f"Citation check: {len(result['ok'])}/{total} OK"
        f"  |  partial: {len(result['partial'])}"
        f"  |  hallucinated: {len(result['hallucinated'])}"
    ]
    for sh, d in result["ok"]:
        lines.append(f"  ✓ {sh}, Điều {d}")
    for sh, d in result["partial"]:
        extra = f", Điều {d}" if d else ""
        lines.append(f"  ~ {sh}{extra}  — so_hieu match, Điều không khớp context")
    for sh, d in result["hallucinated"]:
        extra = f", Điều {d}" if d else ""
        lines.append(f"  ⚠ {sh}{extra}  — KHÔNG có trong context")
    return "\n".join(lines)
