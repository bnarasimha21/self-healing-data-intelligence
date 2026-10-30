"""Postgres source and sink (psycopg 3).

Connection strings come only from the environment: SHDI_PG_DSN for reading
(use a role that has SELECT on the source and nothing else) and
SHDI_PG_SINK_DSN for writing staging and quarantine tables (defaults to
SHDI_PG_DSN). The source connection is also put in read-only mode, so even a
role with write grants cannot change the source through it.
"""
from __future__ import annotations

import datetime as dt
import os
from typing import List, Optional

import pandas as pd
import psycopg
from psycopg import sql

from .base import (
    QUARANTINE_SUFFIX,
    STAGING_SUFFIX,
    check_target,
    frame_fingerprint,
    quarantine_as_text,
    records,
    with_meta,
)


def _dsn(var: str, fallback: Optional[str] = None) -> str:
    dsn = os.environ.get(var) or (os.environ.get(fallback) if fallback else None)
    if not dsn:
        raise RuntimeError(f"{var} is not set; Postgres connection details come only from the environment")
    return dsn


def _ident(table: str) -> sql.Composable:
    parts = table.split(".")
    if len(parts) > 2:
        raise ValueError(f"table must be <schema>.<table> or <table>, not {table!r}")
    return sql.Identifier(*parts)


class PostgresSource:
    def __init__(self, table: str):
        self.table = table
        self.uri = f"postgres:{table}"
        self.name = table.split(".")[-1]

    def read(self, limit: Optional[int] = None) -> pd.DataFrame:
        with psycopg.connect(_dsn("SHDI_PG_DSN")) as con:
            con.read_only = True
            with con.cursor() as cur:
                cur.execute(sql.SQL("SELECT * FROM {} LIMIT 0").format(_ident(self.table)))
                ncols = len(cur.description)
                # Order by every column: deterministic, and rows that tie are identical.
                q = sql.SQL("SELECT * FROM {} ORDER BY {}").format(
                    _ident(self.table), sql.SQL(", ").join(sql.Literal(i + 1) for i in range(ncols))
                )
                if limit:
                    q = q + sql.SQL(" LIMIT {}").format(sql.Literal(int(limit)))
                cur.execute(q)
                cols = [d.name for d in cur.description]
                rows = cur.fetchall()
        df = pd.DataFrame(rows, columns=cols, dtype=object)
        return df

    def fingerprint(self) -> str:
        return frame_fingerprint(self.read())


def _pg_type(series: pd.Series) -> str:
    d = series.dtype
    if pd.api.types.is_bool_dtype(d):
        return "boolean"
    if pd.api.types.is_integer_dtype(d):
        return "bigint"
    if pd.api.types.is_float_dtype(d):
        return "double precision"
    if pd.api.types.is_datetime64_any_dtype(d):
        return "timestamp"
    non_null = [v for v in series.tolist() if v is not None and not (isinstance(v, float) and v != v) and v is not pd.NA]
    if non_null and all(isinstance(v, dt.date) and not isinstance(v, dt.datetime) for v in non_null):
        return "date"
    if non_null and all(isinstance(v, dt.datetime) for v in non_null):
        return "timestamp"
    return "text"


class PostgresSink:
    def __init__(self, table: str):
        self.table = table

    def _write(self, df: pd.DataFrame, run_id: str, suffix: str) -> int:
        target = self.table + suffix
        check_target(target)
        ident = _ident(target)
        types = {c: _pg_type(df[c]) for c in df.columns}
        types["shdi_run_id"] = "text"
        types["shdi_applied_at"] = "timestamptz"
        types["shdi_source_row"] = "bigint"
        cols = list(df.columns)
        with psycopg.connect(_dsn("SHDI_PG_SINK_DSN", "SHDI_PG_DSN")) as con:
            with con.transaction():
                con.execute(
                    sql.SQL("CREATE TABLE IF NOT EXISTS {} ({})").format(
                        ident,
                        sql.SQL(", ").join(sql.SQL("{} {}").format(sql.Identifier(c), sql.SQL(types[c])) for c in cols),
                    )
                )
                con.execute(sql.SQL("DELETE FROM {} WHERE shdi_run_id = %s").format(ident), [run_id])
                insert = sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
                    ident,
                    sql.SQL(", ").join(sql.Identifier(c) for c in cols),
                    sql.SQL(", ").join(sql.Placeholder() for _ in cols),
                )
                with con.cursor() as cur:
                    cur.executemany(insert, [[r[c] for c in cols] for r in records(df)])
        return len(df)

    def write_staging(self, df: pd.DataFrame, run_id: str) -> int:
        return self._write(with_meta(df, run_id), run_id, STAGING_SUFFIX)

    def write_quarantine(self, df: pd.DataFrame, run_id: str) -> int:
        if df.empty:
            return 0
        return self._write(with_meta(quarantine_as_text(df), run_id), run_id, QUARANTINE_SUFFIX)

    def _read(self, suffix: str, run_id: Optional[str]) -> pd.DataFrame:
        target = self.table + suffix
        with psycopg.connect(_dsn("SHDI_PG_SINK_DSN", "SHDI_PG_DSN")) as con:
            con.read_only = True
            exists = con.execute("SELECT to_regclass(%s)", [target]).fetchone()[0]
            if exists is None:
                return pd.DataFrame()
            q = sql.SQL("SELECT * FROM {}").format(_ident(target))
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
