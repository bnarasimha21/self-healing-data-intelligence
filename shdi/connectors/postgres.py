"""Postgres source and sink (psycopg 3).

Connection strings come only from the environment: SHDI_PG_DSN for reading
(use a role that has SELECT on the source and nothing else) and
SHDI_PG_SINK_DSN for writing staging and quarantine tables (defaults to
SHDI_PG_DSN). The source connection is also put in read-only mode, so even a
role with write grants cannot change the source through it.
"""
from __future__ import annotations

import datetime as dt
import decimal
import hashlib
import json
import os
import uuid
from typing import List, Optional, Tuple

import pandas as pd
import psycopg
from psycopg import sql

from ..values import is_null
from .base import (
    QUARANTINE_SUFFIX,
    STAGING_SUFFIX,
    check_target,
    quarantine_as_text,
    records,
    with_meta,
)


def _dsn(var: str, fallback: Optional[str] = None) -> str:
    dsn = os.environ.get(var) or (os.environ.get(fallback) if fallback else None)
    if not dsn:
        raise RuntimeError(f"{var} is not set; Postgres connection details come only from the environment")
    return dsn


def connect(var: str, fallback: Optional[str] = None) -> psycopg.Connection:
    """Open a connection without ever letting the DSN reach an error message.

    psycopg echoes parts of a malformed DSN (password included) in its errors,
    and those errors would otherwise land in MCP tool results, the checkpoint
    and trace spans.
    """
    dsn = _dsn(var, fallback)
    try:
        return psycopg.connect(dsn)
    except psycopg.Error as e:
        detail = e.sqlstate or type(e).__name__
        raise RuntimeError(f"could not connect to Postgres with the DSN in {var} ({detail})") from None


# Postgres truncates identifiers longer than this, so two long names would
# silently collide on the same table.
MAX_IDENT_BYTES = 63


def _parts(table: str) -> List[str]:
    parts = table.split(".")
    if len(parts) > 2:
        raise ValueError(f"table must be <schema>.<table> or <table>, not {table!r}")
    for part in parts:
        if len(part.encode("utf-8")) > MAX_IDENT_BYTES:
            raise ValueError(f"name {part!r} is longer than {MAX_IDENT_BYTES} bytes, which Postgres would truncate")
    return parts


def _ident(table: str) -> sql.Composable:
    return sql.Identifier(*_parts(table))


# Types with a default btree ordering. Anything else (json, xml, point, ...) is
# sorted by its text form, which is still deterministic.
_ORDERABLE = {
    "bool", "int2", "int4", "int8", "float4", "float8", "numeric", "text", "varchar", "bpchar",
    "name", "date", "time", "timetz", "timestamp", "timestamptz", "interval", "uuid", "bytea",
}


def _order_key(con: psycopg.Connection, col: str, type_oid: int) -> sql.Composable:
    info = con.adapters.types.get(type_oid)
    if info is not None and info.name in _ORDERABLE:
        return sql.Identifier(col)
    return sql.SQL("{}::text").format(sql.Identifier(col))


class PostgresSource:
    def __init__(self, table: str):
        self.table = table
        self.uri = f"postgres:{table}"
        self.name = table.split(".")[-1]

    def _read(self, limit: Optional[int], with_text: bool):
        with connect("SHDI_PG_DSN") as con:
            con.read_only = True
            # One snapshot for the rows and their text, rendered in UTC so the
            # fingerprint of a timestamptz does not depend on the session.
            con.isolation_level = psycopg.IsolationLevel.REPEATABLE_READ
            with con.transaction(), con.cursor() as cur:
                cur.execute("SET LOCAL TIME ZONE 'UTC'")
                cur.execute(sql.SQL("SELECT * FROM {} LIMIT 0").format(_ident(self.table)))
                desc = list(cur.description)
                # Order by every column: deterministic, and rows that tie are identical.
                order = sql.SQL(", ").join(_order_key(con, d.name, d.type_code) for d in desc)
                lim = sql.SQL(" LIMIT {}").format(sql.Literal(int(limit))) if limit else sql.SQL("")
                cur.execute(sql.SQL("SELECT * FROM {} ORDER BY {}").format(_ident(self.table), order) + lim)
                cols = [d.name for d in cur.description]
                rows = cur.fetchall()
                fp = None
                if with_text:
                    texts = sql.SQL(", ").join(sql.SQL("{}::text").format(sql.Identifier(d.name)) for d in desc)
                    cur.execute(sql.SQL("SELECT {} FROM {} ORDER BY {}").format(texts, _ident(self.table), order))
                    h = hashlib.sha256()
                    names = []
                    for d in desc:
                        info = con.adapters.types.get(d.type_code)
                        names.append(f"{d.name}:{info.name if info else d.type_code}")
                    h.update(("\x1f".join(names) + "\x1e").encode())
                    for row in cur:
                        h.update(("\x1f".join("\x00" if v is None else v for v in row) + "\x1e").encode())
                    fp = h.hexdigest()
        return pd.DataFrame(rows, columns=cols, dtype=object), fp

    def read(self, limit: Optional[int] = None) -> pd.DataFrame:
        return self._read(limit, False)[0]

    def snapshot(self) -> Tuple[pd.DataFrame, str]:
        return self._read(None, True)

    def fingerprint(self) -> str:
        return self.snapshot()[1]


def _aware(v) -> bool:
    return isinstance(v, dt.datetime) and v.tzinfo is not None


def _pg_type(series: pd.Series) -> str:
    """The Postgres type for a frame column. An instant with an offset is
    timestamptz (a plain timestamp would shift it by the session TimeZone) and
    a Decimal is numeric (double precision would round it)."""
    d = series.dtype
    non_null = [v for v in series.tolist() if not is_null(v)]
    if pd.api.types.is_bool_dtype(d):
        return "boolean"
    if pd.api.types.is_integer_dtype(d) and all(-(2**63) <= int(v) < 2**63 for v in non_null):
        return "bigint"
    if pd.api.types.is_float_dtype(d):
        return "double precision"
    if pd.api.types.is_datetime64_any_dtype(d) and any(isinstance(v, dt.datetime) for v in non_null):
        return "timestamptz" if any(_aware(v) for v in non_null) else "timestamp"
    if non_null and all(isinstance(v, dt.date) and not isinstance(v, dt.datetime) for v in non_null):
        return "date"
    if non_null and all(isinstance(v, dt.datetime) for v in non_null):
        if all(_aware(v) for v in non_null):
            return "timestamptz"
        return "timestamp" if not any(_aware(v) for v in non_null) else "text"
    if non_null and all(isinstance(v, (int, decimal.Decimal)) and not isinstance(v, bool) for v in non_null) and (
        any(isinstance(v, decimal.Decimal) for v in non_null) or any(not -(2**63) <= v < 2**63 for v in non_null)
    ):
        return "numeric"
    # An object column of Python values (a frame read back from Postgres) keeps
    # its type, so writing it again does not widen the sink column to text.
    if non_null and all(isinstance(v, bool) for v in non_null):
        return "boolean"
    if non_null and all(isinstance(v, int) and not isinstance(v, bool) for v in non_null):
        return "bigint"
    if non_null and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in non_null):
        return "double precision"
    return "text"


# information_schema.columns.data_type for each type _pg_type produces.
_INFO_NAME = {
    "boolean": "boolean",
    "bigint": "bigint",
    "double precision": "double precision",
    "timestamp": "timestamp without time zone",
    "timestamptz": "timestamp with time zone",
    "date": "date",
    "numeric": "numeric",
    "text": "text",
}


_PG_INTEGERS = {"smallint", "integer", "bigint"}
_PG_NUMERIC = _PG_INTEGERS | {"real", "double precision", "numeric"}


def _widen(existing: str, wanted: str) -> Optional[str]:
    """The type to change an existing sink column to, or None to leave it.

    Never narrows: a number column arrives as bigint in a run whose values are
    all whole and as double precision otherwise, and both fit a double column."""
    if existing == _INFO_NAME[wanted] or existing == "text":
        return None
    if existing in _PG_NUMERIC and wanted == "numeric":
        # numeric holds every integer and every double exactly.
        return "numeric"
    if existing in _PG_NUMERIC and wanted in ("bigint", "double precision"):
        if existing in ("double precision", "numeric"):
            return None
        if wanted == "double precision" or existing == "real":
            return "double precision"
        return None if existing == "bigint" else "bigint"
    return "text"


def _sync_columns(con: psycopg.Connection, table: str, types: dict, all_null: set) -> None:
    """Make an existing sink table able to take this frame: add missing columns,
    widen columns whose type no longer fits (bigint to double, anything else to text)."""
    parts = _parts(table)
    schema = parts[0] if len(parts) == 2 else None
    q = "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = %s"
    args = [parts[-1]]
    if schema:
        q += " AND table_schema = %s"
        args.append(schema)
    else:
        q += " AND table_schema = current_schema()"
    existing = dict(con.execute(q, args).fetchall())
    ident = _ident(table)
    for col, wanted in types.items():
        if col not in existing:
            con.execute(sql.SQL("ALTER TABLE {} ADD COLUMN {} {}").format(ident, sql.Identifier(col), sql.SQL(wanted)))
            continue
        # A column that is entirely null in this run says nothing about its type.
        new = None if col in all_null else _widen(existing[col], wanted)
        if new:
            con.execute(
                sql.SQL("ALTER TABLE {} ALTER COLUMN {} TYPE {} USING {}::{}").format(
                    ident, sql.Identifier(col), sql.SQL(new), sql.Identifier(col), sql.SQL(new)
                )
            )


_PLAIN = (str, bool, int, float, decimal.Decimal, dt.date, dt.time, dt.timedelta, uuid.UUID, bytes)


def _sink_value(v):
    """A value psycopg can bind. Source columns outside the contract keep their
    Python value (a dict from a json column, say), which has no adapter."""
    if v is None or isinstance(v, _PLAIN):
        return v
    if isinstance(v, (dict, list)):
        return json.dumps(v, default=str)
    return str(v)


class PostgresSink:
    def __init__(self, table: str):
        self.table = table

    def _write_in(self, con: psycopg.Connection, df: Optional[pd.DataFrame], run_id: str, suffix: str) -> None:
        """Replace the run's rows in one table inside the caller's transaction;
        `df` None only deletes them."""
        target = self.table + suffix
        check_target(target)
        ident = _ident(target)
        if df is None:
            if con.execute("SELECT to_regclass(%s)", [ident.as_string(con)]).fetchone()[0] is not None:
                con.execute(sql.SQL("DELETE FROM {} WHERE shdi_run_id = %s").format(ident), [run_id])
            return
        types = {c: _pg_type(df[c]) for c in df.columns}
        types["shdi_run_id"] = "text"
        types["shdi_applied_at"] = "timestamptz"
        types["shdi_source_row"] = "bigint"
        cols = list(df.columns)
        con.execute(
            sql.SQL("CREATE TABLE IF NOT EXISTS {} ({})").format(
                ident,
                sql.SQL(", ").join(sql.SQL("{} {}").format(sql.Identifier(c), sql.SQL(types[c])) for c in cols),
            )
        )
        _sync_columns(con, target, types, {c for c in cols if df[c].isna().all()})
        con.execute(sql.SQL("DELETE FROM {} WHERE shdi_run_id = %s").format(ident), [run_id])
        insert = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
            ident,
            sql.SQL(", ").join(sql.Identifier(c) for c in cols),
            sql.SQL(", ").join(sql.Placeholder() for _ in cols),
        )
        with con.cursor() as cur:
            cur.executemany(insert, [[_sink_value(r[c]) for c in cols] for r in records(df)])

    def _transaction(self, steps) -> None:
        with connect("SHDI_PG_SINK_DSN", "SHDI_PG_DSN") as con:
            with con.transaction():
                for df, run_id, suffix in steps:
                    self._write_in(con, df, run_id, suffix)

    def write_staging(self, df: pd.DataFrame, run_id: str) -> int:
        self._transaction([(with_meta(df, run_id), run_id, STAGING_SUFFIX)])
        return len(df)

    def write_quarantine(self, df: pd.DataFrame, run_id: str) -> int:
        if df.empty:
            return 0
        self._transaction([(with_meta(quarantine_as_text(df), run_id), run_id, QUARANTINE_SUFFIX)])
        return len(df)

    def write_run(self, staging: pd.DataFrame, quarantine: pd.DataFrame, run_id: str) -> tuple:
        """Both tables in one transaction; an empty quarantine still clears
        rows an earlier attempt of the run left there."""
        q = None if quarantine.empty else with_meta(quarantine_as_text(quarantine), run_id)
        self._transaction([(with_meta(staging, run_id), run_id, STAGING_SUFFIX), (q, run_id, QUARANTINE_SUFFIX)])
        return len(staging), len(quarantine)

    def delete_run(self, run_id: str) -> None:
        self._transaction([(None, run_id, STAGING_SUFFIX), (None, run_id, QUARANTINE_SUFFIX)])

    def _read(self, suffix: str, run_id: Optional[str]) -> pd.DataFrame:
        target = self.table + suffix
        ident = _ident(target)
        with connect("SHDI_PG_SINK_DSN", "SHDI_PG_DSN") as con:
            con.read_only = True
            exists = con.execute("SELECT to_regclass(%s)", [ident.as_string(con)]).fetchone()[0]
            if exists is None:
                return pd.DataFrame()
            q = sql.SQL("SELECT * FROM {}").format(ident)
            args: List[str] = []
            if run_id:
                q = q + sql.SQL(" WHERE shdi_run_id = %s")
                args.append(run_id)
            cur = con.execute(q + sql.SQL(" ORDER BY shdi_run_id, shdi_source_row"), args)
            cols = [d.name for d in cur.description]
            return pd.DataFrame(cur.fetchall(), columns=cols, dtype=object)

    def read_staging(self, run_id: Optional[str] = None) -> pd.DataFrame:
        return self._read(STAGING_SUFFIX, run_id)

    def read_quarantine(self, run_id: Optional[str] = None) -> pd.DataFrame:
        return self._read(QUARANTINE_SUFFIX, run_id)
