"""Generate clean sales data and inject labelled defects.

    python -m bench.corrupt --rows 10000 --seed 7 --rate 0.015 --out bench/data

Writes clean_sales.parquet, dirty_sales.duckdb (table `sales`, every column
VARCHAR so dirty values survive) and labels.csv, one row per injected defect:
(source_row, id, column, defect_type, original_value, injected_value).

Each corrupted row gets exactly one defect. Duplicates are appended after the
original rows, so the later copy (the one a duplicate check flags) is always
the injected one. Everything is a pure function of (rows, seed, rate).
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import List, Tuple

import duckdb
import numpy as np
import pandas as pd

COLUMNS = ["id", "date", "sales", "region", "notes"]
REGIONS = ["North", "South", "East", "West"]
NOTES = ["ok", "promo", "repeat customer", "online"]
DEFECT_TYPES = [
    "type_pollution",
    "null",
    "duplicate",
    "out_of_range",
    "unit_error",
    "future_date",
    "pre_1970_date",
]
# The check each defect type must trip. Detection precision and recall are
# measured against these.
EXPECTED_CHECKS = {
    "type_pollution": ["type:sales"],
    "null": ["not_null:sales", "not_null:region"],
    "duplicate": ["duplicate_rows"],
    "out_of_range": ["min:sales"],
    "unit_error": ["max:sales"],
    "future_date": ["max:date"],
    "pre_1970_date": ["min:date"],
}
START = dt.date(2020, 1, 1)
END = dt.date(2025, 12, 31)
AS_OF = dt.date(2026, 1, 1)

_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def number_words(n: int) -> str:
    """Spell out 0..999, the way a person types a number into the wrong field."""
    parts = []
    if n >= 100:
        parts.append(f"{_ONES[n // 100]} hundred")
        n %= 100
    if n >= 20:
        parts.append(_TENS[n // 10] + (f"-{_ONES[n % 10]}" if n % 10 else ""))
    elif n or not parts:
        parts.append(_ONES[n])
    return " ".join(parts)


def make_clean(rows: int, seed: int) -> pd.DataFrame:
    """Clean data, all text. Sales are N(500, 100) truncated at +-3 sigma, so a
    correct outlier rule reports nothing on it."""
    rng = np.random.default_rng(seed)
    sales = []
    while len(sales) < rows:
        draw = rng.normal(500, 100, size=rows)
        sales.extend(int(round(v)) for v in draw if 200 <= round(v) <= 800)
    sales = sales[:rows]
    span = (END - START).days
    dates = [(START + dt.timedelta(days=int(d))).isoformat() for d in rng.integers(0, span + 1, size=rows)]
    return pd.DataFrame(
        {
            "id": [str(i) for i in range(1, rows + 1)],
            "date": dates,
            "sales": [str(v) for v in sales],
            "region": [REGIONS[i] for i in rng.integers(0, len(REGIONS), size=rows)],
            "notes": [NOTES[i] for i in rng.integers(0, len(NOTES), size=rows)],
        },
        dtype=object,
    )


def corrupt(clean: pd.DataFrame, seed: int, rate: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (dirty, labels). `rate` is the share of rows per defect type."""
    rng = np.random.default_rng(seed + 1)
    dirty = clean.copy()
    n = len(clean)
    per_type = int(round(n * rate))
    labels: List[dict] = []
    if per_type == 0:
        return dirty, pd.DataFrame(columns=["source_row", "id", "column", "defect_type", "original_value", "injected_value"])
    order = rng.permutation(n)
    picks = {t: sorted(int(r) for r in order[i * per_type : (i + 1) * per_type]) for i, t in enumerate(DEFECT_TYPES)}

    def label(row, col, kind, new):
        labels.append(
            {
                "source_row": row,
                "id": clean.at[row, "id"],
                "column": col,
                "defect_type": kind,
                "original_value": clean.at[row, col] if col else "",
                "injected_value": new,
            }
        )

    for row in picks["type_pollution"]:
        new = number_words(int(clean.at[row, "sales"]))
        dirty.at[row, "sales"] = new
        label(row, "sales", "type_pollution", new)
    for i, row in enumerate(picks["null"]):
        col = "sales" if i % 2 == 0 else "region"
        dirty.at[row, col] = None
        label(row, col, "null", "")
    for row in picks["out_of_range"]:
        new = str(-int(clean.at[row, "sales"]))
        dirty.at[row, "sales"] = new
        label(row, "sales", "out_of_range", new)
    for row in picks["unit_error"]:
        new = str(int(clean.at[row, "sales"]) * 1000)
        dirty.at[row, "sales"] = new
        label(row, "sales", "unit_error", new)
    for row in picks["future_date"]:
        new = dt.date(int(rng.integers(2030, 2041)), int(rng.integers(1, 13)), int(rng.integers(1, 29))).isoformat()
        dirty.at[row, "date"] = new
        label(row, "date", "future_date", new)
    for row in picks["pre_1970_date"]:
        new = dt.date(int(rng.integers(1940, 1970)), int(rng.integers(1, 13)), int(rng.integers(1, 29))).isoformat()
        dirty.at[row, "date"] = new
        label(row, "date", "pre_1970_date", new)

    # Duplicates: exact copies of untouched rows, appended at the end.
    extra = []
    for k, row in enumerate(picks["duplicate"]):
        extra.append(dirty.loc[row].copy())
        labels.append(
            {
                "source_row": n + k,
                "id": clean.at[row, "id"],
                "column": "",
                "defect_type": "duplicate",
                "original_value": f"copy of row {row}",
                "injected_value": "",
            }
        )
    if extra:
        dirty = pd.concat([dirty, pd.DataFrame(extra)], ignore_index=True)
    return dirty, pd.DataFrame(labels)


def write_duckdb(df: pd.DataFrame, path: Path, table: str = "sales") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    con = duckdb.connect(str(path))
    try:
        con.execute(f'CREATE TABLE "{table}" (' + ", ".join(f'"{c}" VARCHAR' for c in COLUMNS) + ")")
        con.executemany(
            f'INSERT INTO "{table}" VALUES (' + ", ".join("?" for _ in COLUMNS) + ")",
            [[None if v is None or (isinstance(v, float) and v != v) else str(v) for v in r]
             for r in df[COLUMNS].itertuples(index=False, name=None)],
        )
    finally:
        con.close()


def build(out: Path, rows: int, seed: int, rate: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    out.mkdir(parents=True, exist_ok=True)
    clean = make_clean(rows, seed)
    dirty, labels = corrupt(clean, seed, rate)
    clean.to_parquet(out / "clean_sales.parquet", index=False)
    labels.to_csv(out / "labels.csv", index=False)
    db = out / "dirty_sales.duckdb"
    write_duckdb(dirty, db)
    return clean, dirty, labels, db


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate clean data and inject labelled defects")
    ap.add_argument("--rows", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--rate", type=float, default=0.015, help="share of rows per defect type")
    ap.add_argument("--out", default="bench/data")
    a = ap.parse_args()
    clean, dirty, labels, db = build(Path(a.out), a.rows, a.seed, a.rate)
    print(f"{len(clean)} clean rows, {len(labels)} defects, {len(dirty)} dirty rows -> {db}")


if __name__ == "__main__":
    main()
