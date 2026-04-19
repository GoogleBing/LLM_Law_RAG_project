"""
check_parse.py - Parse all .txt files, export JSONL + CSV report, report empty fields.

Usage:
    python src/check_parse.py
    python src/check_parse.py --limit 500            # test on first 500 files
    python src/check_parse.py --out custom.jsonl     # custom JSONL path
    python src/check_parse.py --csv custom.csv       # custom CSV report path
    python src/check_parse.py --no-csv               # skip CSV export
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from collections import Counter, defaultdict

# Ensure Vietnamese characters print correctly on Windows terminals
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
except AttributeError:
    pass

# Allow running from project root or from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from config import RAW_DOCS_DIR, PARSED_JSONL
from parser import parse_legal_document

try:
    from tqdm import tqdm
    USE_TQDM = True
except ImportError:
    USE_TQDM = False

# Fields that are "scalar" (str) vs "list"
SCALAR_FIELDS = [
    "so_hieu", "loai_van_ban", "co_quan_ban_hanh",
    "ngay_ban_hanh", "tieu_de", "ten_day_du",
    "nguoi_ky", "ngay_hieu_luc", "tinh_trang",
]
LIST_FIELDS = ["linh_vuc", "muc_luc", "van_ban_lien_quan"]
ALL_FIELDS  = SCALAR_FIELDS + LIST_FIELDS

# Columns written to the CSV (tieu_de truncated for readability)
CSV_COLUMNS = [
    "_file", "so_hieu", "loai_van_ban", "co_quan_ban_hanh",
    "ngay_ban_hanh", "ngay_hieu_luc", "nguoi_ky", "tinh_trang",
    "tieu_de", "linh_vuc",
    "empty_fields", "empty_count",
]


def is_empty(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, list):
        return len(value) == 0
    return False


def parse_args():
    p = argparse.ArgumentParser(description="Parse legal docs → JSONL + CSV report")
    p.add_argument("--docs", default=RAW_DOCS_DIR,
                   help=f"Directory containing .txt files (default: {RAW_DOCS_DIR})")
    p.add_argument("--out", default=PARSED_JSONL,
                   help=f"Output JSONL file path (default: {PARSED_JSONL})")
    p.add_argument("--csv", default=None,
                   help="Output CSV report path (default: same dir as --out, .csv extension)")
    p.add_argument("--no-csv", action="store_true",
                   help="Skip CSV export")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only first N files (for quick testing)")
    return p.parse_args()


def _csv_path_default(jsonl_path: str) -> str:
    base, _ = os.path.splitext(jsonl_path)
    return base + "_report.csv"


def _record_to_csv_row(record: dict) -> dict:
    empty_scalar = [f for f in SCALAR_FIELDS if is_empty(record.get(f))]
    empty_list   = [f for f in LIST_FIELDS   if is_empty(record.get(f))]
    all_empty    = empty_scalar + empty_list
    return {
        "_file":           record.get("_file", ""),
        "so_hieu":         record.get("so_hieu", ""),
        "loai_van_ban":    record.get("loai_van_ban", ""),
        "co_quan_ban_hanh": record.get("co_quan_ban_hanh", "")[:60],
        "ngay_ban_hanh":   record.get("ngay_ban_hanh", ""),
        "ngay_hieu_luc":   record.get("ngay_hieu_luc", ""),
        "nguoi_ky":        record.get("nguoi_ky", ""),
        "tinh_trang":      record.get("tinh_trang", ""),
        "tieu_de":         record.get("tieu_de", "")[:100],
        "linh_vuc":        "; ".join(record.get("linh_vuc") or []),
        "empty_fields":    ", ".join(all_empty),
        "empty_count":     len(all_empty),
    }


def main():
    args = parse_args()

    csv_path = None if args.no_csv else (args.csv or _csv_path_default(args.out))

    txt_files = sorted(f for f in os.listdir(args.docs) if f.endswith(".txt"))
    if args.limit:
        txt_files = txt_files[: args.limit]

    total = len(txt_files)
    print(f"Processing {total} files -> {args.out}")
    if csv_path:
        print(f"CSV report   -> {csv_path}")

    empty_counts      = Counter()           # field → how many docs have it empty
    empty_field_files = defaultdict(list)   # field → filenames with that field empty
    error_files       = []
    written           = 0

    iterator = tqdm(txt_files, desc="Parsing") if USE_TQDM else txt_files

    csv_fout = open(csv_path, "w", encoding="utf-8-sig", newline="") if csv_path else None
    csv_writer = csv.DictWriter(csv_fout, fieldnames=CSV_COLUMNS) if csv_fout else None
    if csv_writer:
        csv_writer.writeheader()

    try:
        with open(args.out, "w", encoding="utf-8") as fout:
            for fname in iterator:
                fpath = os.path.join(args.docs, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        content = f.read()

                    record = parse_legal_document(content)
                    record["_file"] = fname

                    # Track empty fields
                    for field in ALL_FIELDS:
                        if is_empty(record.get(field)):
                            empty_counts[field] += 1
                            empty_field_files[field].append(fname)

                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    if csv_writer:
                        csv_writer.writerow(_record_to_csv_row(record))
                    written += 1

                except Exception:
                    error_files.append(fname)
                    if not USE_TQDM:
                        print(f"  ERROR: {fname}", flush=True)
    finally:
        if csv_fout:
            csv_fout.close()

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PARSE SUMMARY  ({written}/{total} files written, {len(error_files)} errors)")
    print(f"{'='*60}")
    print(f"\n{'Field':<22}  {'Empty':>7}  {'Filled':>7}  {'Empty %':>8}")
    print(f"{'-'*22}  {'-'*7}  {'-'*7}  {'-'*8}")
    for field in ALL_FIELDS:
        n_empty  = empty_counts[field]
        n_filled = written - n_empty
        pct      = 100.0 * n_empty / written if written else 0
        flag     = " !" if pct > 50 else ""
        print(f"  {field:<20}  {n_empty:>7}  {n_filled:>7}  {pct:>7.1f}%{flag}")

    # ── Per-field file listing ─────────────────────────────────────────────────
    fields_with_empties = [f for f in ALL_FIELDS if empty_field_files[f]]
    if fields_with_empties:
        print(f"\n{'='*60}")
        print(f"  EMPTY FIELD DETAILS (files with missing values)")
        print(f"{'='*60}")
        MAX_PER_FIELD = 30
        for field in fields_with_empties:
            fnames = empty_field_files[field]
            print(f"\n  [{field}]  -- {len(fnames)} file(s) empty:")
            for fn in fnames[:MAX_PER_FIELD]:
                print(f"    {fn}")
            if len(fnames) > MAX_PER_FIELD:
                print(f"    ... and {len(fnames) - MAX_PER_FIELD} more")

    # ── Per-file summary (files with ANY empty scalar field) ──────────────────
    files_empties: dict[str, list[str]] = defaultdict(list)
    for field in SCALAR_FIELDS:
        for fn in empty_field_files[field]:
            files_empties[fn].append(field)

    if files_empties:
        sorted_files = sorted(files_empties.items(), key=lambda x: (-len(x[1]), x[0]))
        print(f"\n{'='*60}")
        print(f"  FILES WITH EMPTY ATTRIBUTES ({len(sorted_files)} file(s))")
        print(f"{'='*60}")
        MAX_FILES = 100
        for fn, empty_flds in sorted_files[:MAX_FILES]:
            print(f"  {fn:<50}  [{', '.join(empty_flds)}]")
        if len(sorted_files) > MAX_FILES:
            print(f"  ... and {len(sorted_files) - MAX_FILES} more files")

    if error_files:
        print(f"\n{'='*60}")
        print(f"  FILES WITH ERRORS ({len(error_files)}):")
        for ef in error_files[:20]:
            print(f"    {ef}")
        if len(error_files) > 20:
            print(f"    ... and {len(error_files) - 20} more")

    print(f"\n  JSONL : {args.out}")
    if csv_path:
        print(f"  CSV   : {csv_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
