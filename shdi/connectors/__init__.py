"""Open sources and sinks from a URI.

    file.csv  or  csv:path/file.csv
    duckdb:///abs/path/file.duckdb#table    duckdb:rel/path.duckdb#schema.table
    postgres:schema.table                   (DSN from SHDI_PG_DSN, never the URI)
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple

from ..config import sink_dir
from .base import Sink, Source, frame_fingerprint, to_csv_bytes  # noqa: F401


class SourceURIError(ValueError):
    pass


# A table is `table` or `schema.table`, each part a plain identifier. Anything else
# (a quoted path like "/etc/x.csv", a URL, a table function) is refused, because
# DuckDB would happily read a file named in the table position.
_TABLE_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?", re.ASCII)


def check_table_name(table: str) -> str:
    if not _TABLE_RE.fullmatch(table):
        raise SourceURIError(
            f"table {table!r} is not a plain identifier; use <table> or <schema>.<table> "
            "with letters, digits and underscores"
        )
    return table


def parse_uri(uri: str) -> Tuple[str, str, Optional[str]]:
    """Return (kind, location, table)."""
    u = uri.strip()
    if u.startswith(("postgres://", "postgresql://")) or (u.startswith("postgres") and "@" in u):
        raise SourceURIError(
            "Postgres connection details are read from SHDI_PG_DSN, never from the URI. "
            "Use postgres:<schema>.<table>."
        )
    if u.startswith("postgres:"):
        table = u[len("postgres:"):]
        if not table:
            raise SourceURIError("postgres URI needs a table: postgres:<schema>.<table>")
        return "postgres", "", check_table_name(table)
    if u.startswith("duckdb:"):
        rest = u[len("duckdb:"):]
        if rest.startswith("//"):
            rest = rest[2:]
        if "#" not in rest:
            raise SourceURIError("duckdb URI needs a table: duckdb:///path/file.duckdb#table")
        path, table = rest.split("#", 1)
        if not path or not table or path == ":memory:":
            raise SourceURIError(
                "duckdb URI needs a file path and a table; in-memory databases cannot be "
                "re-opened after a pause, so they are not accepted as sources"
            )
        return "duckdb", str(Path(path).expanduser().resolve()), check_table_name(table)
    if u.startswith("csv:"):
        u = u[len("csv:"):]
    if u.lower().endswith(".csv"):
        return "csv", str(Path(u).expanduser().resolve()), None
    raise SourceURIError(f"unrecognised source {uri!r}; expected a .csv path, duckdb:...#table or postgres:schema.table")


def canonical_uri(uri: str) -> str:
    kind, loc, table = parse_uri(uri)
    if kind == "csv":
        return f"csv:{loc}"
    if kind == "duckdb":
        return f"duckdb://{loc}#{table}"
    return f"postgres:{table}"


def open_source(uri: str) -> Source:
    kind, loc, table = parse_uri(uri)
    if kind == "csv":
        from .csv import CsvSource

        return CsvSource(loc)
    if kind == "duckdb":
        from .duckdb import DuckDBSource

        return DuckDBSource(loc, table)
    from .postgres import PostgresSource

    return PostgresSource(table)


def open_sink(uri: str, directory: Optional[str] = None) -> Sink:
    kind, loc, table = parse_uri(uri)
    if kind == "csv":
        from .csv import CsvSink

        return CsvSink(Path(directory) if directory else sink_dir(), Path(loc).stem)
    if kind == "duckdb":
        from .duckdb import DuckDBSink

        return DuckDBSink(loc, table)
    from .postgres import PostgresSink

    return PostgresSink(table)
