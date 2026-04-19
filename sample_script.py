import pandas as pd
import sys

# Fix stdout encoding for Windows
sys.stdout.reconfigure(encoding='utf-8')

# Read CSV
df = pd.read_csv('RES ver2.csv', encoding='utf-8-sig')
print(f'Total rows in CSV: {len(df)}')

col = 'Số hiệu VBPL (Trích xuất)'

# Target doc IDs (case-insensitive partial match)
target_ids = [
    '14/2022/TT-NHNN',
    '73/2024/NĐ-CP',
    '07/2024/TT-BNV',
    '07/VBHN-VPQH',
    '595/QĐ-BHXH',
    '1570/TB-BLĐTBXH',
    '5015/TB-LĐTBXH',
    '6150/TB-BLĐTBXH',
    '52/2024/NĐ-CP',
    '17/2024/TT-NHNN',
    '39/2016/TT-NHNN',
]

# Build a case-insensitive filter
target_ids_lower = [t.lower() for t in target_ids]

def matches_any(cell_value):
    if pd.isna(cell_value):
        return False
    val = str(cell_value).lower()
    return any(tid in val for tid in target_ids_lower)

mask = df[col].apply(matches_any)
filtered = df[mask].copy()
print(f'Rows matching filter: {len(filtered)}')

# Breakdown by doc ID
print('\nBreakdown by doc ID:')
for tid in target_ids:
    tid_lower = tid.lower()
    count = filtered[col].apply(
        lambda x: tid_lower in str(x).lower() if not pd.isna(x) else False
    ).sum()
    print(f'  {tid}: {count}')

# Sample 150 rows (or all if fewer)
sample_size = min(150, len(filtered))
sampled = filtered.sample(n=sample_size, random_state=42)
print(f'\nRows sampled: {len(sampled)}')

# Save to output CSV
sampled.to_csv('test_sample.csv', index=False, encoding='utf-8-sig')
print('Saved to test_sample.csv')

# Also show unique doc IDs in filtered set for verification
print('\nUnique doc IDs in filtered set:')
for v in sorted(filtered[col].dropna().unique()):
    print(f'  {repr(v)}')
