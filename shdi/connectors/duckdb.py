"""DuckDB source and sink.

The source is opened with read_only=True, so the run cannot modify the source
table even by accident. The sink opens the same file read-write only after the
source connection is closed, and writes only `<table>__shdi_staging` and
`<table>__shdi_quarantine`.

Rows are read through Arrow, so DECIMAL stays an exact Decimal, HUGEINT keeps
every digit, DATE stays a date and a DOUBLE NaN stays NaN instead of becoming
NULL. The fingerprint hashes the database's own text for every cell plus the
column types, so a change pandas would hide (DECIMAL 1.10 to 1.1, a DATE
turned TIMESTAMP) still marks the source as changed. Every connection runs in
UTC, so TIMESTAMPTZ values read and hash the same on any machine.
"""
from __future__ import annotations

import hashlib
import re
from typing import List, Optional, Tuple

import duckdb
import pandas as pd

from .base import (
    QUARANTINE_SUFFIX,
    STAGING_SUFFIX,
    check_target,
    quarantine_as_text,
    with_meta,
)


def _run(con: duckdb.DuckDBPyConnection, query: str, args: Optional[list] = None) -> list:
    """Execute and fetch, so no unfinished result blocks the next statement."""
    return con.execute(query, args).fetchall() if args is not None else con.execute(query).fetchall()


def connect(path: str, read_only: bool) -> duckdb.DuckDBPyConnection:
    """Open `path` in UTC with file, URL and extension access switched off.

    TimeZone needs the ICU extension, which cannot be loaded once external
    access is off, so it is loaded first; external access is then disabled
    for the rest of the connection's life (DuckDB refuses to turn it back on).
    The source table name is already restricted to identifiers; this is the
    second lock on the door."""
    con = duckdb.connect(path, read_only=read_only)
    try:
        try:
            _run(con, "LOAD icu")
            _run(con, "SET TimeZone = 'UTC'")
        except duckdb.Error:
            # Without ICU, TIMESTAMPTZ is always rendered in UTC anyway.
            pass
        _run(con, "SET enable_external_access = false")
    except BaseException:
        con.close()
        raise
    return con


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def quote_table(table: str) -> str:
    return ".".join(quote_ident(p) for p in table.split("."))


def _arrow_frame(rel: duckdb.DuckDBPyRelation) -> pd.DataFrame:
    return rel.to_arrow_table().to_pandas(types_mapper=pd.ArrowDtype)


def _fingerprint(columns: List[Tuple[str, str]], rows: List[tuple]) -> str:
    h = hashlib.sha256()
    h.update(("\x1f".join(f"{c}:{t}" for c, t in columns) + "\x1e").encode())
    for row in rows:
        h.update(("\x1f".join("\x00" if v is None else v for v in row) + "\x1e").encode())
    return h.hexdigest()


class DuckDBSource:
    def __init__(self, path: str, table: str):
        self.path = path
        self.table = table
        self.uri = f"duckdb://{path}#{table}"
        self.name = table.split(".")[-1]

    def _ordered(self, con: duckdb.DuckDBPyConnection, select: str, limit: Optional[int]) -> duckdb.DuckDBPyRelation:
        q = f"SELECT {select} FROM {quote_table(self.table)}"
        lim = f" LIMIT {int(limit)}" if limit else ""
        try:
            # Base tables: insertion order, which is what a person calls row 1, 2, 3.
            return con.sql(q + " ORDER BY rowid" + lim)
        except duckdb.Error:
            # Views have no rowid. Sorting on every column is still deterministic:
            # rows that tie are identical, so their order does not matter.
            return con.sql(q + " ORDER BY ALL" + lim)

    def read(self, limit: Optional[int] = None) -> pd.DataFrame:
        con = connect(self.path, read_only=True)
        try:
            return _arrow_frame(self._ordered(con, "*", limit))
        finally:
            con.close()

    def snapshot(self) -> Tuple[pd.DataFrame, str]:
        """The rows and their fingerprint, read on one connection."""
        con = connect(self.path, read_only=True)
        try:
            df = _arrow_frame(self._ordered(con, "*", None))
            types = [(r[0], r[1]) for r in con.execute(f"DESCRIBE SELECT * FROM {quote_table(self.table)}").fetchall()]
            if types:
                text = self._ordered(con, ", ".join(f"CAST({quote_ident(c)} AS VARCHAR)" for c, _ in types), None).fetchall()
            else:
                text = []
        finally:
            con.close()
        return df, _fingerprint(types, text)

    def fingerprint(self) -> str:
        return self.snapshot()[1]


_INT_DIGITS = {
    "TINYINT": 3, "SMALLINT": 5, "INTEGER": 10, "BIGINT": 19, "HUGEINT": 39,
    "UTINYINT": 3, "USMALLINT": 5, "UINTEGER": 10, "UBIGINT": 20, "UHUGEINT": 39,
}
_FLOAT_TYPES = {"FLOAT", "DOUBLE"}
_DECIMAL_RE = re.compile(r"DECIMAL\((\d+),\s*(\d+)\)")


def _exact_shape(t: str) -> Optional[Tuple[int, int]]:
    """(whole digits, scale) for integer and DECIMAL types, else None."""
    if t in _INT_DIGITS:
        return _INT_DIGITS[t], 0
    m = _DECIMAL_RE.fullmatch(t)
    if m:
        p, s = int(m.group(1)), int(m.group(2))
        return p - s, s
    return None


def _widen(existing: str, wanted: str) -> Optional[str]:
    """The type to change an existing sink column to, or None to leave it.

    Never narrows. Integers and decimals widen to a DECIMAL that holds both
    exactly (BIGINT and DECIMAL(4,1) give DECIMAL(20,1)); a float meeting an
    integer gives DOUBLE as before; anything else, or a DECIMAL that would need
    more than 38 digits, becomes VARCHAR, which holds every value's text."""
    if wanted == "NULL" or existing == wanted or existing == "VARCHAR":
        return None
    e, w = _exact_shape(existing), _exact_shape(wanted)
    if e and w:
        whole, scale = max(e[0], w[0]), max(e[1], w[1])
        if e[0] >= whole and e[1] >= scale:
            return None
        if existing in _INT_DIGITS and wanted in _INT_DIGITS and whole <= 19:
            return "BIGINT"
        return f"DECIMAL({whole + scale},{scale})" if whole + scale <= 38 else "VARCHAR"
    if existing == "DOUBLE" and (wanted in _FLOAT_TYPES or wanted in _INT_DIGITS):
        return None
    if (existing in _FLOAT_TYPES or existing in _INT_DIGITS) and (wanted in _FLOAT_TYPES or wanted in _INT_DIGITS):
        return "DOUBLE"
    return "VARCHAR"


def _frame_types(con: duckdb.DuckDBPyConnection) -> List[tuple]:
    """(column, type) for the registered frame. An all-null column has type NULL,
    which DuckDB cannot store later values in, so it is created as VARCHAR."""
    return [(r[0], r[1]) for r in con.execute("DESCRIBE SELECT * FROM shdi_frame").fetchall()]


def _table_types(con: duckdb.DuckDBPyConnection, table: str) -> dict:
    return {r[0]: r[1] for r in con.execute(f"DESCRIBE {quote_table(table)}").fetchall()}


def _table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    parts = table.split(".")
    q = "SELECT count(*) FROM information_schema.tables WHERE table_name = ?"
    args = [parts[-1]]
    if len(parts) > 1:
        q += " AND table_schema = ?"
        args.append(parts[-2])
    return bool(con.execute(q, args).fetchall()[0][0])


class DuckDBSink:
    def __init__(self, path: str, table: str):
        self.path = path
        self.table = table

    def _source_types(self, con: duckdb.DuckDBPyConnection) -> dict:
        # Checked first: a failed DESCRIBE would abort the open transaction.
        if not _table_exists(con, self.table):
            return {}
        return _table_types(con, self.table)

    def _keep_source_type(self, con: duckdb.DuckDBPyConnection, col: str, frame_type: str, src: str) -> bool:
        """True when a staging column can take its source type back (HUGEINT,
        UUID, ENUM come through Arrow as DECIMAL(38,0) or VARCHAR) without any
        value changing. A VARCHAR source column that the contract cast to a type
        keeps the cast."""
        if src == frame_type or src == "VARCHAR":
            return False
        c = quote_ident(col)
        bad = con.execute(
            f"SELECT count(*) FROM shdi_frame WHERE {c} IS NOT NULL AND "
            f"(TRY_CAST({c} AS {src}) IS NULL OR CAST(TRY_CAST({c} AS {src}) AS VARCHAR) <> CAST({c} AS VARCHAR))"
        ).fetchall()[0][0]
        return bad == 0

    def _write_in(self, con: duckdb.DuckDBPyConnection, df: Optional[pd.DataFrame], run_id: str, suffix: str,
                  src_types: dict) -> None:
        """Replace the run's rows in one table, inside the caller's transaction.
        `df` None only deletes them."""
        target = self.table + suffix
        check_target(target)
        qt = quote_table(target)
        if df is None:
            if _table_exists(con, target):
                _run(con, f"DELETE FROM {qt} WHERE shdi_run_id = ?", [run_id])
            return
        con.register("shdi_frame", df)
        try:
            wanted = []
            for c, t in _frame_types(con):
                if c in src_types and (t == "NULL" or self._keep_source_type(con, c, t, src_types[c])):
                    t = src_types[c]
                wanted.append((c, "VARCHAR" if t == "NULL" else t, t == "NULL"))
            cols = ", ".join(f"{quote_ident(c)} {t}" for c, t, _ in wanted)
            _run(con, f"CREATE TABLE IF NOT EXISTS {qt} ({cols})")
            existing = _table_types(con, target)
            # A column that is entirely null in this run says nothing about its
            # type, so it never changes an existing column.
            all_null = {c for c in df.columns if df[c].isna().all()}
            for c, t, untyped in wanted:
                if c not in existing:
                    _run(con, f"ALTER TABLE {qt} ADD COLUMN {quote_ident(c)} {t}")
                    continue
                new = None if (c in all_null or untyped) else _widen(existing[c], t)
                new = self._exact_or_text(con, qt, c, existing[c], t, new)
                if new:
                    _run(con, f"ALTER TABLE {qt} ALTER COLUMN {quote_ident(c)} TYPE {new}")
            _run(con, f"DELETE FROM {qt} WHERE shdi_run_id = ?", [run_id])
            _run(con, f"INSERT INTO {qt} BY NAME SELECT * FROM shdi_frame")
        finally:
            con.unregister("shdi_frame")

    @staticmethod
    def _exact_or_text(con: duckdb.DuckDBPyConnection, qt: str, col: str, existing: str, wanted: str,
                       new: Optional[str]) -> Optional[str]:
        """A DOUBLE holds every integer only up to 2**53. Where integers meet
        floats (either side), a larger integer in the table or the frame makes
        the column VARCHAR instead, so no staged value is rounded."""
        ints_in_table = existing in _INT_DIGITS and (new == "DOUBLE")
        ints_in_frame = wanted in _INT_DIGITS and (existing in _FLOAT_TYPES or new == "DOUBLE")
        if not (ints_in_table or ints_in_frame):
            return new
        c = quote_ident(col)
        big = 0
        if ints_in_table:
            big += con.execute(f"SELECT count(*) FROM {qt} WHERE abs({c}::HUGEINT) > 9007199254740992").fetchall()[0][0]
        if ints_in_frame:
            big += con.execute(f"SELECT count(*) FROM shdi_frame WHERE abs({c}::HUGEINT) > 9007199254740992").fetchall()[0][0]
        return "VARCHAR" if big else new

    def _transaction(self, steps) -> None:
        con = connect(self.path, read_only=False)
        try:
            _run(con, "BEGIN")
            src_types = self._source_types(con)
            for df, run_id, suffix, keep_types in steps:
                self._write_in(con, df, run_id, suffix, src_types if keep_types else {})
            _run(con, "COMMIT")
        except Exception:
            try:
                _run(con, "ROLLBACK")
            except duckdb.Error:
                pass
            raise
        finally:
            con.close()

    def write_staging(self, df: pd.DataFrame, run_id: str) -> int:
        self._transaction([(with_meta(df, run_id), run_id, STAGING_SUFFIX, True)])
        return len(df)

    def write_quarantine(self, df: pd.DataFrame, run_id: str) -> int:
        if df.empty:
            return 0
        self._transaction([(with_meta(quarantine_as_text(df), run_id), run_id, QUARANTINE_SUFFIX, False)])
        return len(df)

    def write_run(self, staging: pd.DataFrame, quarantine: pd.DataFrame, run_id: str) -> tuple:
        """Replace the run's rows in staging and quarantine in one transaction,
        so a failure leaves neither table changed. An empty quarantine still
        removes rows an earlier attempt of the run left there."""
        q = None if quarantine.empty else with_meta(quarantine_as_text(quarantine), run_id)
        self._transaction([
            (with_meta(staging, run_id), run_id, STAGING_SUFFIX, True),
            (q, run_id, QUARANTINE_SUFFIX, False),
        ])
        return len(staging), len(quarantine)

    def delete_run(self, run_id: str) -> None:
        self._transaction([(None, run_id, STAGING_SUFFIX, False), (None, run_id, QUARANTINE_SUFFIX, False)])

    def _read(self, suffix: str, run_id: Optional[str]) -> pd.DataFrame:
        con = connect(self.path, read_only=True)
        try:
            if not _table_exists(con, self.table + suffix):
                return pd.DataFrame()
            q = f"SELECT * FROM {quote_table(self.table + suffix)}"
            if run_id:
                q += " WHERE shdi_run_id = $run_id"
            rel = con.sql(q + " ORDER BY shdi_run_id, shdi_source_row", params={"run_id": run_id} if run_id else None)
            return _arrow_frame(rel)
        finally:
            con.close()

    def read_staging(self, run_id: Optional[str] = None) -> pd.DataFrame:
        return self._read(STAGING_SUFFIX, run_id)

    def read_quarantine(self, run_id: Optional[str] = None) -> pd.DataFrame:
        return self._read(QUARANTINE_SUFFIX, run_id)
