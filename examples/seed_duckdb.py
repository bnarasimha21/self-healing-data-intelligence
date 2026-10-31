"""Load a CSV into a DuckDB table with every column as VARCHAR.

Text columns keep dirty values such as "three hundred" exactly as they are,
which is what a raw landing table usually looks like.

    python examples/seed_duckdb.py examples/dirty_sales.csv .shdi/demo.duckdb sales
    shdi run duckdb:.shdi/demo.duckdb#sales --contract contracts/sales.yaml
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import duckdb


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def seed(csv_path: str, db_path: str, table: str = "sales") -> int:
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(db_path)
    try:
        cols = ", ".join(f"{quote_ident(c)} VARCHAR" for c in header)
        con.execute(f"CREATE OR REPLACE TABLE {quote_ident(table)} ({cols})")
        marks = ", ".join("?" for _ in header)
        # Empty cells become NULL, matching how the CSV source reads them.
        con.executemany(
            f"INSERT INTO {quote_ident(table)} VALUES ({marks})",
            [[v if v != "" else None for v in r] for r in body],
        )
        return con.execute(f"SELECT count(*) FROM {quote_ident(table)}").fetchone()[0]
    finally:
        con.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csv")
    ap.add_argument("db")
    ap.add_argument("table", nargs="?", default="sales")
    a = ap.parse_args()
    n = seed(a.csv, a.db, a.table)
    print(f"seeded {n} rows into {a.db}#{a.table}")


if __name__ == "__main__":
    main()
