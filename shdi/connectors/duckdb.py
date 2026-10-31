"""DuckDB source and sink.

The source is opened with read_only=True, so the run cannot modify the source
table even by accident. The sink opens the same file read-write only after the
source connection is closed, and writes only `<table>__shdi_staging` and
`<table>__shdi_quarantine`.
"""
from __future__ import annotations

from typing import List, Optional

import duckdb
import pandas as pd

from .base import (
    QUARANTINE_SUFFIX,
    STAGING_SUFFIX,
    check_target,
    frame_fingerprint,
    quarantine_as_text,
    with_meta,
)


# No file, URL or extension access on any connection we open. The source table
# name is already restricted to identifiers; this is the second lock on the door.
_CONFIG = {"enable_external_access": False}


def connect(path: str, read_only: bool) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(path, read_only=read_only, config=dict(_CONFIG))


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def quote_table(table: str) -> str:
    return ".".join(quote_ident(p) for p in table.split("."))


class DuckDBSource:
    def __init__(self, path: str, table: str):
        self.path = path
        self.table = table
        self.uri = f"duckdb://{path}#{table}"
        self.name = table.split(".")[-1]

    def read(self, limit: Optional[int] = None) -> pd.DataFrame:
        con = connect(self.path, read_only=True)
        try:
            q = f"SELECT * FROM {quote_table(self.table)}"
            try:
                # Base tables: insertion order, which is what a person calls row 1, 2, 3.
                df = con.execute(q + " ORDER BY rowid" + (f" LIMIT {int(limit)}" if limit else "")).df()
            except duckdb.Error:
                # Views have no rowid. Sorting on every column is still deterministic:
                # rows that tie are identical, so their order does not matter.
                df = con.execute(q + " ORDER BY ALL" + (f" LIMIT {int(limit)}" if limit else "")).df()
        finally:
            con.close()
        return df

    def fingerprint(self) -> str:
        return frame_fingerprint(self.read())


_INTEGER_TYPES = {"TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT"}
_WIDE_INTEGERS = {"BIGINT", "HUGEINT", "UBIGINT"}
_FLOAT_TYPES = {"FLOAT", "DOUBLE"}


def _widen(existing: str, wanted: str) -> Optional[str]:
    """The type to change an existing sink column to, or None to leave it.

    Never narrows: a number column arrives as BIGINT in a run whose values
    are all whole and as DOUBLE otherwise, and both fit a DOUBLE column."""
    if wanted == "NULL" or existing == wanted or existing == "VARCHAR":
        return None
    numeric = _INTEGER_TYPES | _FLOAT_TYPES
    if existing in numeric and wanted in numeric:
        if existing == "DOUBLE":
            return None
        if wanted in _FLOAT_TYPES or existing == "FLOAT":
            return "DOUBLE"
        return None if existing in _WIDE_INTEGERS else "BIGINT"
    return "VARCHAR"


def _frame_types(con: duckdb.DuckDBPyConnection) -> List[tuple]:
    """(column, type) for the registered frame. An all-null column has type NULL,
    which DuckDB cannot store later values in, so it is created as VARCHAR."""
    return [(r[0], r[1]) for r in con.execute("DESCRIBE SELECT * FROM shdi_frame").fetchall()]


def _table_types(con: duckdb.DuckDBPyConnection, table: str) -> dict:
    return {r[0]: r[1] for r in con.execute(f"DESCRIBE {quote_table(table)}").fetchall()}


class DuckDBSink:
    def __init__(self, path: str, table: str):
        self.path = path
        self.table = table

    def _write(self, df: pd.DataFrame, run_id: str, suffix: str) -> int:
        target = self.table + suffix
        check_target(target)
        qt = quote_table(target)
        con = connect(self.path, read_only=False)
        try:
            con.register("shdi_frame", df)
            con.execute("BEGIN")
            wanted = _frame_types(con)
            cols = ", ".join(f"{quote_ident(c)} {'VARCHAR' if t == 'NULL' else t}" for c, t in wanted)
            con.execute(f"CREATE TABLE IF NOT EXISTS {qt} ({cols})")
            existing = _table_types(con, target)
            # A column that is entirely null in this run says nothing about its
            # type, so it never changes an existing column.
            all_null = {c for c in df.columns if df[c].isna().all()}
            for c, t in wanted:
                if c not in existing:
                    con.execute(f"ALTER TABLE {qt} ADD COLUMN {quote_ident(c)} {'VARCHAR' if t == 'NULL' else t}")
                    continue
                new = None if c in all_null else _widen(existing[c], t)
                if new:
                    con.execute(f"ALTER TABLE {qt} ALTER COLUMN {quote_ident(c)} TYPE {new}")
            con.execute(f"DELETE FROM {qt} WHERE shdi_run_id = ?", [run_id])
            con.execute(f"INSERT INTO {qt} BY NAME SELECT * FROM shdi_frame")
            con.execute("COMMIT")
        except Exception:
            try:
                con.execute("ROLLBACK")
            except duckdb.Error:
                pass
            raise
        finally:
            con.close()
        return len(df)

    def write_staging(self, df: pd.DataFrame, run_id: str) -> int:
        return self._write(with_meta(df, run_id), run_id, STAGING_SUFFIX)

    def write_quarantine(self, df: pd.DataFrame, run_id: str) -> int:
        if df.empty:
            return 0
        return self._write(with_meta(quarantine_as_text(df), run_id), run_id, QUARANTINE_SUFFIX)

    def _read(self, suffix: str, run_id: Optional[str]) -> pd.DataFrame:
        con = connect(self.path, read_only=True)
        try:
            parts = (self.table + suffix).split(".")
            q_exists = "SELECT count(*) FROM information_schema.tables WHERE table_name = ?"
            args_exists = [parts[-1]]
            if len(parts) > 1:
                q_exists += " AND table_schema = ?"
                args_exists.append(parts[-2])
            exists = con.execute(q_exists, args_exists).fetchone()[0]
            if not exists:
                return pd.DataFrame()
            q = f"SELECT * FROM {quote_table(self.table + suffix)}"
            args: List[str] = []
            if run_id:
                q += " WHERE shdi_run_id = ?"
                args.append(run_id)
            return con.execute(q + " ORDER BY shdi_run_id, shdi_source_row", args).df()
        finally:
            con.close()

    def read_staging(self, run_id: Optional[str] = None) -> pd.DataFrame:
        return self._read(STAGING_SUFFIX, run_id)

    def read_quarantine(self, run_id: Optional[str] = None) -> pd.DataFrame:
        return self._read(QUARANTINE_SUFFIX, run_id)
