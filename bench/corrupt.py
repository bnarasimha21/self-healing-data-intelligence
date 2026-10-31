"""Generate clean sales data and inject labelled defects.

    python -m bench.corrupt --rows 10000 --seed 7 --rate 0.015 --out bench/data

There are two scenarios, each with its own contract:

* text: table `sales`, every column VARCHAR, so dirty text (number words,
  out-of-range dates) survives. Contract bench/contract.yaml.
* typed: table `sales_typed` with BIGINT, DATE, DECIMAL(10,2) and TIMESTAMPTZ
  columns, the way a real warehouse table looks. Type pollution cannot exist
  there. Contract bench/contract_typed.yaml.

Each scenario writes clean_<name>.parquet, dirty_<name>.duckdb and
labels_<name>.csv, one row per injected defect:
(source_row, id, column, defect_type, original_value, injected_value).

Some defect types are described by the contract (a negative sale breaks
`min`, x1000 is a declared unit factor). Others are not: a x10 slip, a wrong
value that is still in range, a date with day and month swapped. The contract
says nothing that separates them from good data, so they measure what the
pipeline cannot see rather than what it was told to look for.

Each corrupted row gets exactly one defect. Duplicates are appended after the
original rows, so the later copy (the one a duplicate check flags) is always
the injected one. Everything is a pure function of (rows, seed, rate).
"""
from __future__ import annotations

import argparse
import datetime as dt
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REGIONS = ["North", "South", "East", "West"]
NOTES = ["ok", "promo", "repeat customer", "online"]
START = dt.date(2020, 1, 1)
END = dt.date(2025, 12, 31)
AS_OF = dt.date(2026, 1, 1)

# Defect types the contract does not describe. No check is expected to flag
# them; detection is measured against every failing check and outlier finding.
UNDESCRIBED = ["x10_error", "in_range_wrong", "day_month_swap"]

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


@dataclass(frozen=True)
class Scenario:
    name: str
    table: str
    contract: Path
    columns: Dict[str, str]  # column -> contract type
    sql_types: Dict[str, str]
    measure: str  # the sales column
    date: str  # the date column
    defect_types: List[str]
    # The checks each described defect type must trip. Detection precision
    # and recall are measured against these; None means undescribed.
    expected_checks: Dict[str, Optional[List[str]]]


TEXT = Scenario(
    name="text",
    table="sales",
    contract=HERE / "contract.yaml",
    columns={"id": "integer", "date": "date", "sales": "number", "region": "string", "notes": "string"},
    sql_types={c: "VARCHAR" for c in ("id", "date", "sales", "region", "notes")},
    measure="sales",
    date="date",
    defect_types=["type_pollution", "null", "duplicate", "out_of_range", "unit_error", "future_date",
                  "pre_1970_date"] + UNDESCRIBED,
    expected_checks={
        "type_pollution": ["type:sales"],
        "null": ["not_null:sales", "not_null:region"],
        "duplicate": ["duplicate_rows"],
        "out_of_range": ["min:sales"],
        "unit_error": ["max:sales"],
        "future_date": ["max:date"],
        "pre_1970_date": ["min:date"],
        **{k: None for k in UNDESCRIBED},
    },
)

TYPED = Scenario(
    name="typed",
    table="sales_typed",
    contract=HERE / "contract_typed.yaml",
    columns={"id": "integer", "day": "date", "amount": "number", "ts": "timestamp", "region": "string"},
    sql_types={"id": "BIGINT", "day": "DATE", "amount": "DECIMAL(10,2)", "ts": "TIMESTAMPTZ", "region": "VARCHAR"},
    measure="amount",
    date="day",
    defect_types=["null", "duplicate", "out_of_range", "unit_error", "future_date", "pre_1970_date"] + UNDESCRIBED,
    expected_checks={
        "null": ["not_null:amount", "not_null:region"],
        "duplicate": ["duplicate_rows"],
        "out_of_range": ["min:amount"],
        "unit_error": ["max:amount"],
        "future_date": ["max:day"],
        "pre_1970_date": ["min:day"],
        **{k: None for k in UNDESCRIBED},
    },
)

SCENARIOS = {s.name: s for s in (TEXT, TYPED)}

# Kept for callers of the text scenario.
COLUMNS = list(TEXT.columns)
DEFECT_TYPES = TEXT.defect_types
EXPECTED_CHECKS = TEXT.expected_checks


def _sales_draws(rng: np.random.Generator, rows: int, cents: bool) -> List[Any]:
    """N(500, 100) truncated at +-3 sigma, so a correct outlier rule reports
    nothing on clean data."""
    out: List[Any] = []
    while len(out) < rows:
        draw = rng.normal(500, 100, size=rows)
        for v in draw:
            v = round(float(v), 2) if cents else int(round(v))
            if 200 <= v <= 800:
                out.append(v)
    return out[:rows]


def make_clean(rows: int, seed: int, scenario: Scenario = TEXT) -> pd.DataFrame:
    """Clean data as Python values: text for the text scenario; int, date,
    Decimal and aware datetime for the typed one."""
    rng = np.random.default_rng(seed)
    span = (END - START).days
    if scenario.name == "text":
        sales = _sales_draws(rng, rows, cents=False)
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
    amounts = _sales_draws(rng, rows, cents=True)
    days = [START + dt.timedelta(days=int(d)) for d in rng.integers(0, span + 1, size=rows)]
    msecs = rng.integers(0, 86400 * 1000, size=rows)
    utc = dt.timezone.utc
    return pd.DataFrame(
        {
            "id": list(range(1, rows + 1)),
            "day": days,
            "amount": [Decimal(f"{v:.2f}") for v in amounts],
            "ts": [dt.datetime.combine(d, dt.time(), utc) + dt.timedelta(milliseconds=int(s))
                   for d, s in zip(days, msecs)],
            "region": [REGIONS[i] for i in rng.integers(0, len(REGIONS), size=rows)],
        },
        dtype=object,
    )


def _as_date(v: Any) -> dt.date:
    return v if isinstance(v, dt.date) else dt.date.fromisoformat(v)


def _swappable(v: Any) -> bool:
    d = _as_date(v)
    return d.day <= 12 and d.day != d.month


def corrupt(clean: pd.DataFrame, seed: int, rate: float, scenario: Scenario = TEXT) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (dirty, labels). `rate` is the share of rows per defect type."""
    rng = np.random.default_rng(seed + 1)
    dirty = clean.copy()
    n = len(clean)
    per_type = int(round(n * rate))
    labels: List[dict] = []
    if per_type == 0:
        return dirty, pd.DataFrame(columns=["source_row", "id", "column", "defect_type", "original_value", "injected_value"])
    text = scenario.name == "text"
    m, dcol = scenario.measure, scenario.date

    def num(v: Any) -> Decimal:
        return Decimal(v) if isinstance(v, str) else v

    def out(v: Decimal) -> Any:
        return str(v) if text else v

    def day_out(d: dt.date) -> Any:
        return d.isoformat() if text else d

    # Rows are handed out from one permutation in defect-type order. A type
    # that needs a particular kind of row (a date whose day and month can be
    # swapped) takes the next rows that qualify.
    order = [int(r) for r in rng.permutation(n)]
    used: set = set()
    eligible: Dict[str, Callable[[int], bool]] = {"day_month_swap": lambda r: _swappable(clean.at[r, dcol])}
    picks: Dict[str, List[int]] = {}
    for kind in scenario.defect_types:
        ok = eligible.get(kind, lambda r: True)
        chosen = []
        for r in order:
            if len(chosen) == per_type:
                break
            if r not in used and ok(r):
                chosen.append(r)
        used.update(chosen)
        picks[kind] = sorted(chosen)

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

    def put(row, col, kind, new):
        dirty.at[row, col] = new
        label(row, col, kind, new)

    for row in picks.get("type_pollution", []):
        put(row, m, "type_pollution", number_words(int(clean.at[row, m])))
    for i, row in enumerate(picks["null"]):
        col = m if i % 2 == 0 else "region"
        dirty.at[row, col] = None
        label(row, col, "null", "")
    for row in picks["out_of_range"]:
        put(row, m, "out_of_range", out(-num(clean.at[row, m])))
    for row in picks["unit_error"]:
        put(row, m, "unit_error", out(num(clean.at[row, m]) * 1000))
    for row in picks["future_date"]:
        d = dt.date(int(rng.integers(2030, 2041)), int(rng.integers(1, 13)), int(rng.integers(1, 29)))
        put(row, dcol, "future_date", day_out(d))
    for row in picks["pre_1970_date"]:
        d = dt.date(int(rng.integers(1940, 1970)), int(rng.integers(1, 13)), int(rng.integers(1, 29)))
        put(row, dcol, "pre_1970_date", day_out(d))
    # Undescribed: a slipped decimal point, still inside the contract's 0..10000.
    for row in picks["x10_error"]:
        put(row, m, "x10_error", out(num(clean.at[row, m]) * 10))
    # Undescribed: a plausible but wrong value, still inside the clean spread.
    for row in picks["in_range_wrong"]:
        v = num(clean.at[row, m])
        step = Decimal(int(rng.integers(50, 151)))
        put(row, m, "in_range_wrong", out(v + step if v + step <= 800 else v - step))
    # Undescribed: day and month read the wrong way round, still a valid date.
    for row in picks["day_month_swap"]:
        d = _as_date(clean.at[row, dcol])
        put(row, dcol, "day_month_swap", day_out(dt.date(d.year, d.day, d.month)))

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


def write_duckdb(df: pd.DataFrame, path: Path, scenario: Scenario = TEXT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    cols = list(scenario.columns)
    text = scenario.name == "text"
    con = duckdb.connect(str(path))
    try:
        con.execute(f'CREATE TABLE "{scenario.table}" ('
                    + ", ".join(f'"{c}" {scenario.sql_types[c]}' for c in cols) + ")")
        con.executemany(
            f'INSERT INTO "{scenario.table}" VALUES (' + ", ".join("?" for _ in cols) + ")",
            [[None if v is None or (isinstance(v, float) and v != v) else (str(v) if text else v) for v in r]
             for r in df[cols].itertuples(index=False, name=None)],
        )
    finally:
        con.close()


def build(out: Path, rows: int, seed: int, rate: float,
          scenario: Scenario = TEXT) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
    out.mkdir(parents=True, exist_ok=True)
    clean = make_clean(rows, seed, scenario)
    dirty, labels = corrupt(clean, seed, rate, scenario)
    clean.astype(str).to_parquet(out / f"clean_{scenario.name}.parquet", index=False)
    labels.astype(str).to_csv(out / f"labels_{scenario.name}.csv", index=False)
    db = out / f"dirty_{scenario.name}.duckdb"
    write_duckdb(dirty, db, scenario)
    return clean, dirty, labels, db


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate clean data and inject labelled defects")
    ap.add_argument("--rows", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--rate", type=float, default=0.015, help="share of rows per defect type")
    ap.add_argument("--out", default="bench/data")
    a = ap.parse_args()
    for scenario in SCENARIOS.values():
        clean, dirty, labels, db = build(Path(a.out), a.rows, a.seed, a.rate, scenario)
        print(f"{scenario.name}: {len(clean)} clean rows, {len(labels)} defects, {len(dirty)} dirty rows -> {db}")


if __name__ == "__main__":
    main()
