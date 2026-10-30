"""Issue 3: Postgres source, staging and quarantine tables, end to end.

Skipped unless SHDI_PG_DSN (a SELECT-only role) is set. SHDI_PG_SINK_DSN is the
role that writes staging and quarantine; it also seeds the source here. See
docker-compose.test.yml for a local server with both roles.
"""
from __future__ import annotations

import csv
import os
import uuid

import pytest

from conftest import FIXTURES
from db_e2e import rerun_same_id_replaces_not_duplicates, run_and_check

pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif(not os.environ.get("SHDI_PG_DSN"), reason="SHDI_PG_DSN not set"),
]

psycopg = pytest.importorskip("psycopg")
from psycopg import sql  # noqa: E402


def writer_dsn() -> str:
    return os.environ.get("SHDI_PG_SINK_DSN") or os.environ["SHDI_PG_DSN"]


@pytest.fixture
def pg_source():
    """A fresh schema with the dirty fixture as an all-text table."""
    schema = f"shdi_t_{uuid.uuid4().hex[:10]}"
    with open(FIXTURES / "dirty_sales.csv", newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    reader_role = psycopg.conninfo.conninfo_to_dict(os.environ["SHDI_PG_DSN"]).get("user")
    with psycopg.connect(writer_dsn(), autocommit=True) as con:
        con.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        table = sql.Identifier(schema, "sales")
        con.execute(
            sql.SQL("CREATE TABLE {} ({})").format(
                table, sql.SQL(", ").join(sql.SQL("{} text").format(sql.Identifier(c)) for c in header)
            )
        )
        with con.cursor() as cur:
            cur.executemany(
                sql.SQL("INSERT INTO {} VALUES ({})").format(table, sql.SQL(", ").join(sql.Placeholder() for _ in header)),
                [[v if v != "" else None for v in r] for r in body],
            )
        if reader_role:
            con.execute(sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(sql.Identifier(schema), sql.Identifier(reader_role)))
            con.execute(sql.SQL("GRANT SELECT ON {} TO {}").format(table, sql.Identifier(reader_role)))
    yield schema
    with psycopg.connect(writer_dsn(), autocommit=True) as con:
        con.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def count(schema: str) -> int:
    with psycopg.connect(os.environ["SHDI_PG_DSN"]) as con:
        return con.execute(sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(schema, "sales"))).fetchone()[0]


def test_postgres_end_to_end(pg_source, dirty_csv, tmp_path):
    uri = f"postgres:{pg_source}.sales"
    run_and_check(uri, dirty_csv, tmp_path, lambda: count(pg_source))
    rerun_same_id_replaces_not_duplicates(uri, tmp_path)
    with psycopg.connect(writer_dsn()) as con:
        tables = {
            r[0]
            for r in con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = %s", [pg_source]
            ).fetchall()
        }
        types = dict(
            con.execute(
                "SELECT column_name, data_type FROM information_schema.columns WHERE table_schema = %s AND table_name = %s",
                [pg_source, "sales__shdi_staging"],
            ).fetchall()
        )
    assert tables == {"sales", "sales__shdi_staging", "sales__shdi_quarantine"}
    # Approved casts give the staging table real types.
    assert types["id"] == "bigint" and types["date"] == "date" and types["sales"] == "bigint"


def test_reader_role_cannot_modify_the_source(pg_source):
    if psycopg.conninfo.conninfo_to_dict(os.environ["SHDI_PG_DSN"]).get("user") == psycopg.conninfo.conninfo_to_dict(
        writer_dsn()
    ).get("user"):
        pytest.skip("SHDI_PG_DSN and SHDI_PG_SINK_DSN use the same role")
    with psycopg.connect(os.environ["SHDI_PG_DSN"]) as con:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            con.execute(sql.SQL("DELETE FROM {}").format(sql.Identifier(pg_source, "sales")))


def test_dsn_is_never_taken_from_the_uri(monkeypatch):
    from shdi.connectors import SourceURIError, parse_uri

    with pytest.raises(SourceURIError):
        parse_uri("postgres://shdi_reader:reader@localhost:55433/shdi")
    monkeypatch.delenv("SHDI_PG_DSN")
    from shdi.connectors import open_source

    with pytest.raises(RuntimeError, match="SHDI_PG_DSN is not set"):
        open_source("postgres:public.sales").read()
