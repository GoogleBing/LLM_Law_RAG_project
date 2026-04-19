"""
=============================================================================
parser.py - Legal Document Parser (Integrated Version)
=============================================================================
Parse van ban phap luat Viet Nam tu file .txt thanh structured dict.

Output attributes (tieng Viet khong dau):
    so_hieu, loai_van_ban, co_quan_ban_hanh, ngay_ban_hanh, tieu_de,
    ten_day_du, nguoi_ky, linh_vuc, ngay_hieu_luc, tinh_trang,
    muc_luc, van_ban_lien_quan
    # noi_dung  (commented out - enable after validation)
=============================================================================
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any


# =============================================================================
# CONSTANTS
# =============================================================================

MARKER_METADATA = "THONG TIN THEM"
MARKER_TOC      = "MUC LUC"
MARKER_RELATED  = "VAN BAN LIEN QUAN"

# Longer keywords first so "THONG TU LIEN TICH" matches before "THONG TU"
KEYWORDS_DOC_TYPES = [
    "THÔNG TƯ LIÊN TỊCH",
    "THÔNG TƯ",
    "NGHỊ ĐỊNH",
    "QUYẾT ĐỊNH",
    "NGHỊ QUYẾT",
    "CHỈ THỊ",
    "KẾ HOẠCH",
    "PHÁP LỆNH",
    "SẮC LỆNH",
    "THÔNG BÁO",
    "BÁO CÁO",
    "CÔNG ƯỚC",
    "ĐIỀU LỆ",
    "LỆNH",
    "LUẬT",
    "CÔNG BỐ",
    "QUY ĐỊNH",
    "KẾT LUẬN",
    "QUY TRÌNH",
    "TỜ TRÌNH",
    "THÔNG TRI"
]

TITLE_STOP_KEYWORDS_NORM = [
    "BO TRUONG", "THU TRUONG", "CHU TICH", "THU TUONG",
    "TONG BI THU", "TONG GIAM DOC", "GIAM DOC",
    "HIEU TRUONG", "VIEN TRUONG", "TRUONG BAN",
    "CHINH PHU", "UY BAN NHAN DAN", "HOI DONG NHAN DAN",
    "BAN CHAP HANH", "CAN CU", "QUYET NGHI",
    "QUYET DINH:", "NGHI QUYET:",
]

METADATA_FIELD_MAP = {
    "Ten":            "ten_day_du",
    "So hieu":        "so_hieu_meta",
    "Loai van ban":   "loai_van_ban_meta",
    "Linh vuc nganh": "linh_vuc",
    "Noi ban hanh":   "noi_ban_hanh_meta",
    "Nguoi ky":       "nguoi_ky",
    "Ngay ban hanh":  "ngay_ban_hanh_meta",
    "Ngay hieu luc":  "ngay_hieu_luc",
    "Tinh trang":     "tinh_trang",
    "So cong bao":    "so_cong_bao",
    "Ngay dang":      "ngay_dang",
}

RELATION_TYPE_MAP = {
    "VAN BAN DUOC CAN CU":             "can_cu",
    "VAN BAN DUOC HUONG DAN":           "huong_dan",
    "VAN BAN DUOC DAN CHIEU":           "dan_chieu",
    "VAN BAN THAY THE":                 "thay_the",
    "VAN BAN LIEN QUAN CUNG NOI DUNG":  "lien_quan",
}


# =============================================================================
# COMPILED REGEX
# =============================================================================

RE_DATE = re.compile(
    r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})", re.IGNORECASE)

# Fixed: [^\s,]+ stops at whitespace or comma — won't eat trailing city/date
RE_DOC_ID = re.compile(r"Số\s*:\s*([^\s,]+)", re.IGNORECASE)

RE_SEPARATOR = re.compile(r"^[\s\*\-\_\=\.]+$")
RE_BULLET    = re.compile(r"^\s*-\s+")
RE_REF_ID    = re.compile(r"\(ID:\s*(\d+)\)")

# Extract effective date from body when metadata field is absent/unknown.
# Matches: "có hiệu lực kể từ ngày DD tháng MM năm YYYY" and variants.
RE_HIEU_LUC_BODY = re.compile(
    r"(?:có hiệu lực|hiệu lực thi hành|hiệu lực kể từ|hiệu lực từ ngày)"
    r"[^\n]{0,60}?"
    r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})",
    re.IGNORECASE | re.UNICODE,
)

# Inline stop-phrase regex used to trim preamble from title candidates.
# When a paragraph merges the title with "Bộ trưởng... Căn cứ..." we cut there.
RE_TITLE_STOP_INLINE = re.compile(
    r"(?:bộ\s*trưởng|thứ\s*trưởng|chủ\s*tịch|thủ\s*tướng|tổng\s*bí\s*thư|"
    r"tổng\s*giám\s*đốc|giám\s*đốc|hiệu\s*trưởng|viện\s*trưởng|trưởng\s*ban|"
    r"căn\s*cứ|quyết\s*nghị|chính\s*phủ\b)",
    re.IGNORECASE | re.UNICODE,
)


# =============================================================================
# TEXT UTILITIES
# =============================================================================

def _normalize_text(s: str) -> str:
    decomp = unicodedata.normalize("NFD", s)
    stripped = "".join(ch for ch in decomp if unicodedata.category(ch) != "Mn")
    return stripped.upper().replace("Đ", "D")


def _compact_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _clean_co_quan(text: str) -> str:
    """Strip trailing separator characters (***..., ---...) from issuer name."""
    return re.sub(r"[\s\*\-\_\=\.]+$", "", text).strip()


# =============================================================================
# STEP 0: SPLIT RAW FILE INTO SECTIONS
# =============================================================================

def _find_section_index(lines: list[str], marker: str) -> int | None:
    for idx, line in enumerate(lines):
        if marker in _normalize_text(line):
            return idx
    return None


def _split_raw_sections(lines: list[str]) -> dict:
    idx_meta    = _find_section_index(lines, MARKER_METADATA)
    idx_toc     = _find_section_index(lines, MARKER_TOC)
    idx_related = _find_section_index(lines, MARKER_RELATED)

    tail_starts = [i for i in [idx_meta, idx_toc, idx_related] if i is not None]
    body_end = min(tail_starts) if tail_starts else len(lines)

    metadata_lines = []
    if idx_meta is not None:
        ends = [i for i in [idx_toc, idx_related, len(lines)] if i is not None and i > idx_meta]
        metadata_lines = lines[idx_meta + 1 : min(ends) if ends else len(lines)]

    toc_lines = []
    if idx_toc is not None:
        ends = [i for i in [idx_related, len(lines)] if i is not None and i > idx_toc]
        toc_lines = lines[idx_toc + 1 : min(ends) if ends else len(lines)]

    related_lines = lines[idx_related + 1 :] if idx_related is not None else []

    return {
        "body_lines":     lines[:body_end],
        "metadata_lines": metadata_lines,
        "toc_lines":      toc_lines,
        "related_lines":  related_lines,
    }


# =============================================================================
# STEP 1: MERGE SINGLE NEWLINES
# =============================================================================

def _merge_single_newlines(text: str) -> str:
    merged = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    return re.sub(r"[ \t]+", " ", merged)


def _split_into_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]


# =============================================================================
# STEP 2: UPPERCASE GROUP ANALYSIS
# =============================================================================

def _is_separator_para(para: str) -> bool:
    return bool(RE_SEPARATOR.match(para))


def _is_uppercase_para(para: str) -> bool:
    words = re.findall(r"[^\W\d_]+", para, re.UNICODE)
    if not words:
        return False
    return sum(1 for w in words if w.isupper()) / len(words) >= 0.8


def _starts_with_stop_keyword(para: str) -> bool:
    norm = _normalize_text(para)
    return any(norm.startswith(kw) for kw in TITLE_STOP_KEYWORDS_NORM)


def _find_uppercase_groups(paragraphs: list[str]) -> list[list[str]]:
    groups: list[list[str]] = []
    current: list[str] = []

    for para in paragraphs:
        if _is_separator_para(para):
            if current:
                groups.append(current)
                current = []
        elif _is_uppercase_para(para):
            if current and _starts_with_stop_keyword(para):
                groups.append(current)
                current = []
            else:
                current.append(para)
        else:
            if current:
                groups.append(current)
                current = []

    if current:
        groups.append(current)
    return groups


def _find_quoc_hieu_group_idx(groups: list[list[str]]) -> int | None:
    for i, group in enumerate(groups[:6]):
        norm = _normalize_text(" ".join(group))
        if ("CONG HOA" in norm and "VIET NAM" in norm) or \
           "VIET NAM DAN CHU CONG HOA" in norm:
            return i
    return None


# =============================================================================
# STEP 2b: EXTRACT TYPE + TITLE
# =============================================================================

def _canonicalize_doc_type(keyword: str) -> str:
    norm = _normalize_text(keyword)
    mapping = [
        ("THONG TU LIEN TICH", "THÔNG TƯ LIÊN TỊCH"),
        ("THONG TU",           "THÔNG TƯ"),
        ("NGHI DINH",          "NGHỊ ĐỊNH"),
        ("QUYET DINH",         "QUYẾT ĐỊNH"),
        ("NGHI QUYET",         "NGHỊ QUYẾT"),
        ("CHI THI",            "CHỈ THỊ"),
        ("KE HOACH",           "KẾ HOẠCH"),
        ("PHAP LENH",          "PHÁP LỆNH"),
        ("SAC LENH",           "SẮC LỆNH"),
        ("THONG BAO",          "THÔNG BÁO"),
        ("BAO CAO",            "BÁO CÁO"),
        ("CONG UOC",           "CÔNG ƯỚC"),
        ("DIEU LE",            "ĐIỀU LỆ"),
        ("LENH",               "LỆNH"),
    ]
    for pattern, canonical in mapping:
        if pattern in norm:
            return canonical
    if norm.startswith("LUAT"):
        return "LUẬT"
    return keyword.upper() if keyword else ""


def _extract_type_and_title(group_paras: list[str]) -> tuple[str, str]:
    if not group_paras:
        return "", ""

    doc_type_idx: int | None = None
    matched_kw: str = ""
    for j, para in enumerate(group_paras):
        norm = _normalize_text(para)
        for kw in KEYWORDS_DOC_TYPES:
            if _normalize_text(kw) in norm:
                doc_type_idx = j
                matched_kw = kw
                break
        if doc_type_idx is not None:
            break

    if doc_type_idx is None:
        return "", _compact_spaces(" ".join(group_paras))

    title_parts = list(group_paras[doc_type_idx + 1:])

    if not title_parts and matched_kw:
        doc_type_para = group_paras[doc_type_idx]
        m = re.search(re.escape(matched_kw), doc_type_para, re.IGNORECASE | re.UNICODE)
        if m:
            rest = doc_type_para[m.end():].strip()
            if rest:
                title_parts = [rest]

    return matched_kw, _compact_spaces(" ".join(title_parts))


# =============================================================================
# STEP 3: EXTRACT DOC_ID
# =============================================================================

def _extract_doc_id(paragraphs: list[str]) -> str:
    for i, para in enumerate(paragraphs):
        m = RE_DOC_ID.search(para)
        if m:
            raw = m.group(1).strip()
            if raw:
                return re.sub(r"\s+", "", raw)
            if i + 1 < len(paragraphs):
                nxt = paragraphs[i + 1].strip()
                if nxt and not RE_DOC_ID.search(nxt):
                    return re.sub(r"\s+", "", nxt)
    return ""


# =============================================================================
# STEP 4: STRUCTURED METADATA
# =============================================================================

def _parse_structured_metadata(metadata_lines: list[str]) -> dict:
    result = {}
    for raw_line in metadata_lines:
        line = raw_line.strip()
        if not line:
            continue
        for key_prefix, field_name in METADATA_FIELD_MAP.items():
            if line.startswith(key_prefix + ":"):
                value = line[len(key_prefix) + 1:].strip()
                if field_name == "linh_vuc":
                    result[field_name] = [v.strip() for v in value.split(",") if v.strip()]
                elif value and value != "Dữ liệu đang cập nhật" \
                        and not re.fullmatch(r"[\*\-\_\=\.]+", value):
                    result[field_name] = value
                break
    return result


# =============================================================================
# STEP 5: TABLE OF CONTENTS
# =============================================================================

def _parse_toc(toc_lines: list[str]) -> list[str]:
    return [l.strip()[2:].strip() for l in toc_lines if l.strip().startswith("- ")]


# =============================================================================
# STEP 6: TYPED RELATIONS
# =============================================================================

def _parse_typed_relations(related_lines: list[str]) -> list[dict]:
    relations = []
    current_type = "lien_quan"

    for raw_line in related_lines:
        stripped = raw_line.strip()
        if not stripped:
            continue

        if stripped.endswith(":") and not stripped.startswith("-"):
            norm = _normalize_text(stripped.rstrip(":"))
            for pattern, rel_type in RELATION_TYPE_MAP.items():
                if pattern in norm:
                    current_type = rel_type
                    break
            continue

        if stripped.startswith("- "):
            ref_text = stripped[2:].strip()
            ref_id = None
            id_match = RE_REF_ID.search(ref_text)
            if id_match:
                ref_id = id_match.group(1)
                ref_text = ref_text[:id_match.start()].strip()

            relations.append({
                "loai": current_type,
                "mo_ta": ref_text,
                "id": ref_id,
            })

    return relations


# =============================================================================
# STEP 7 (OPTIONAL): EXTRACT NOI_DUNG
# =============================================================================

def _extract_noi_dung(body_lines, paragraphs, groups, qh_idx):
    """
    Extract main body content, excluding the header block.
    Enable after Steps 1-6 are validated on the full 16K dataset.
    """
    if qh_idx is None:
        return "\n".join(body_lines)

    title_group_idx = qh_idx + 1
    if title_group_idx >= len(groups):
        return "\n".join(body_lines)

    title_group = groups[title_group_idx]
    last_title_para = title_group[-1] if title_group else ""

    body_start = 0
    for i, para in enumerate(paragraphs):
        if para == last_title_para:
            body_start = i + 1
            break

    # Skip date line and signer preamble
    while body_start < len(paragraphs):
        para = paragraphs[body_start]
        if RE_DATE.search(para) or _is_separator_para(para):
            body_start += 1
        else:
            break

    if body_start >= len(paragraphs):
        return ""

    return "\n\n".join(paragraphs[body_start:])


# =============================================================================
# STEP 2c: FALLBACK TITLE FROM NEXT PARAGRAPH
# =============================================================================

def _try_next_para_title(paragraphs: list[str], doc_type_last_para: str) -> str:
    """
    When the title paragraph immediately follows the doc-type paragraph but was
    merged with preamble text (e.g. 'VỀ VIỆC... BỘ TRƯỞNG... Căn cứ...'), try
    to recover the title by cutting at the first stop phrase.

    Returns the extracted title string, or "" if nothing useful is found.
    """
    if not doc_type_last_para:
        return ""
    try:
        idx = next(i for i, p in enumerate(paragraphs) if p == doc_type_last_para)
    except StopIteration:
        return ""
    if idx + 1 >= len(paragraphs):
        return ""

    candidate = paragraphs[idx + 1]
    if not candidate:
        return ""

    if _is_uppercase_para(candidate) and not _starts_with_stop_keyword(candidate):
        return _compact_spaces(candidate)

    # Mixed-case paragraph: try to cut at the first inline stop phrase.
    m = RE_TITLE_STOP_INLINE.search(candidate)
    if m and m.start() > 0:
        title = candidate[: m.start()].strip()
        if title:
            return _compact_spaces(title)

    return ""


# =============================================================================
# PUBLIC API
# =============================================================================

def parse_legal_document(content: str) -> dict[str, Any]:
    """
    Parse a Vietnamese legal document .txt file into structured fields.

    Returns dict with keys:
        so_hieu, loai_van_ban, co_quan_ban_hanh, ngay_ban_hanh,
        tieu_de, ten_day_du, nguoi_ky, linh_vuc, ngay_hieu_luc,
        tinh_trang, muc_luc, van_ban_lien_quan
    """
    lines = content.replace("\r\n", "\n").split("\n")

    # Step 0: split sections
    sections = _split_raw_sections(lines)

    # Step 1: merge + paragraphs
    merged_body = _merge_single_newlines("\n".join(sections["body_lines"]))
    paragraphs = _split_into_paragraphs(merged_body)

    # Step 2: uppercase groups
    groups = _find_uppercase_groups(paragraphs)
    qh_idx = _find_quoc_hieu_group_idx(groups)

    if qh_idx is not None:
        if qh_idx == 0:
            issuer_paras = []
            for para in groups[0]:
                norm = _normalize_text(para)
                if ("CONG HOA" in norm and "VIET NAM" in norm) or \
                   "VIET NAM DAN CHU CONG HOA" in norm:
                    break
                issuer_paras.append(para)
            co_quan = _clean_co_quan(_compact_spaces(" ".join(issuer_paras)))
        else:
            co_quan = _clean_co_quan(_compact_spaces(" ".join(p for g in groups[:qh_idx] for p in g)))

        title_idx = qh_idx + 1
        if title_idx < len(groups):
            matched_kw, tieu_de = _extract_type_and_title(groups[title_idx])
            loai_vb = _canonicalize_doc_type(matched_kw) if matched_kw else ""
        else:
            tieu_de, loai_vb = "", ""
    else:
        # When qh_idx is None the Quoc-hieu block was not detected as uppercase
        # (e.g. "Độc lập - Tự do - Hạnh phúc" is mixed-case).
        # For single-group documents (CÔNG ƯỚC, ĐIỀU LỆ …) start search from 0.
        search_from = 0 if len(groups) == 1 else 1
        co_quan = _clean_co_quan(_compact_spaces(" ".join(groups[0]))) if groups else ""

        title_idx = None
        for i in range(search_from, len(groups)):
            norm = _normalize_text(" ".join(groups[i]))
            if any(_normalize_text(kw) in norm for kw in KEYWORDS_DOC_TYPES):
                title_idx = i
                break
        if title_idx is None and len(groups) >= 3:
            title_idx = 2  # legacy fallback

        if title_idx is not None:
            matched_kw, tieu_de = _extract_type_and_title(groups[title_idx])
            loai_vb = _canonicalize_doc_type(matched_kw) if matched_kw else ""
            if title_idx == 0:
                co_quan = ""  # no separate issuer; will fall back to metadata
        else:
            tieu_de, loai_vb = "", ""

    # Step 2c: fallback — title in the paragraph immediately after the doc-type
    if not tieu_de and groups:
        title_group = groups[title_idx] if title_idx is not None and title_idx < len(groups) else []
        doc_type_last = title_group[-1] if title_group else ""
        tieu_de = _try_next_para_title(paragraphs, doc_type_last)

    # Step 3: doc_id
    so_hieu = _extract_doc_id(paragraphs)

    # Date
    dm = RE_DATE.search(merged_body)
    ngay_bh = f"{int(dm.group(1)):02d}/{int(dm.group(2)):02d}/{dm.group(3)}" if dm else ""

    # Step 4: metadata
    meta = _parse_structured_metadata(sections["metadata_lines"])

    # Step 5: TOC
    muc_luc = _parse_toc(sections["toc_lines"])

    # Step 6: relations
    relations = _parse_typed_relations(sections["related_lines"])

    # Step 7: noi_dung (disabled)
    noi_dung = _extract_noi_dung(
        sections["body_lines"], paragraphs, groups, qh_idx)

    # Last-resort fallback: use ten_day_du from metadata when tieu_de is empty
    ten_day_du = meta.get("ten_day_du", "")
    if not tieu_de:
        tieu_de = ten_day_du

    # ngay_hieu_luc: metadata first; fall back to body extraction for VBHN/TTHN
    # and similar docs where the field is absent from the metadata block.
    ngay_hieu_luc = meta.get("ngay_hieu_luc", "")
    if not ngay_hieu_luc:
        hm = RE_HIEU_LUC_BODY.search(merged_body)
        if hm:
            ngay_hieu_luc = f"{int(hm.group(1)):02d}/{int(hm.group(2)):02d}/{hm.group(3)}"

    return {
        "so_hieu":          so_hieu or meta.get("so_hieu_meta", ""),
        "loai_van_ban":     loai_vb or meta.get("loai_van_ban_meta", ""),
        "co_quan_ban_hanh": co_quan or meta.get("noi_ban_hanh_meta", ""),
        "ngay_ban_hanh":    ngay_bh or meta.get("ngay_ban_hanh_meta", ""),
        "tieu_de":          tieu_de,
        "ten_day_du":       ten_day_du,
        "nguoi_ky":         meta.get("nguoi_ky", ""),
        "linh_vuc":         meta.get("linh_vuc", []),
        "ngay_hieu_luc":    ngay_hieu_luc,
        "tinh_trang":       meta.get("tinh_trang", ""),
        "muc_luc":          muc_luc,
        "van_ban_lien_quan": relations,
        "noi_dung":       noi_dung,
    }


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    import os
    import sys
    from collections import Counter
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from config import RAW_DOCS_DIR

    sample_dir = RAW_DOCS_DIR
    print("=" * 70)
    print("Legal Document Parser - Integrated Version Test")
    print("=" * 70)

    for fname in sorted(os.listdir(sample_dir)):
        if not fname.endswith(".txt"):
            continue

        with open(os.path.join(sample_dir, fname), "r", encoding="utf-8") as f:
            result = parse_legal_document(f.read())

        print(f"\n{'─'*50}")
        print(f"FILE: {fname}")
        print(f"{'─'*50}")
        print(f"  so_hieu:          {result['so_hieu']}")
        print(f"  loai_van_ban:     {result['loai_van_ban']}")
        print(f"  co_quan_ban_hanh: {result['co_quan_ban_hanh'][:55]}")
        print(f"  ngay_ban_hanh:    {result['ngay_ban_hanh']}")
        print(f"  tieu_de:          {result['tieu_de'][:65]}")
        print(f"  nguoi_ky:         {result['nguoi_ky']}")
        print(f"  linh_vuc:         {result['linh_vuc']}")
        print(f"  tinh_trang:       {result['tinh_trang']}")
        print(f"  muc_luc:          {len(result['muc_luc'])} items")

        rels = result["van_ban_lien_quan"]
        if rels:
            types = Counter(r["loai"] for r in rels)
            print(f"  van_ban_lien_quan: {len(rels)} items → {dict(types)}")
            for r in rels[:2]:
                print(f"    [{r['loai']:10}] {r['mo_ta'][:50]}... (ID:{r['id']})")
        else:
            print(f"  van_ban_lien_quan: 0 items")
