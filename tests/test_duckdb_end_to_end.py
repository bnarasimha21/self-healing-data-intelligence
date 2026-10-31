"""Issue 3: DuckDB source, staging and quarantine tables, end to end."""
from __future__ import annotations

import sys

import duckdb
import pytest

from shdi import workflow
from shdi.connectors import SourceURIError, open_sink, parse_uri
from shdi.connectors.base import check_target
from shdi.graph import build_graph, make_checkpointer

from conftest import AS_OF, CONTRACT, FIXTURES, ROOT
from db_e2e import rerun_same_id_replaces_not_duplicates, run_and_check

sys.path.insert(0, str(ROOT / "examples"))
from seed_duckdb import seed  # noqa: E402


@pytest.fixture
def duck(tmp_path):
    path = tmp_path / "landing.duckdb"
    assert seed(str(FIXTURES / "dirty_sales.csv"), str(path), "sales") == 12
    return path


def count(path, table="sales"):
    con = duckdb.connect(str(path), read_only=True)
    try:
        return con.execute(f'SELECT count(*) FROM "{table}"').fetchone()[0]
    finally:
        con.close()


def test_duckdb_end_to_end(duck, dirty_csv, tmp_path):
    uri = f"duckdb://{duck}#sales"
    run_and_check(uri, dirty_csv, tmp_path, lambda: count(duck))
    rerun_same_id_replaces_not_duplicates(uri, tmp_path)
    con = duckdb.connect(str(duck), read_only=True)
    tables = {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    con.close()
    assert tables == {"sales", "sales__shdi_staging", "sales__shdi_quarantine"}


def test_duckdb_typed_source_and_relative_uri(tmp_path, monkeypatch):
    """A typed table (INTEGER, DATE, DOUBLE) works too, via a relative URI."""
    monkeypatch.chdir(tmp_path)
    con = duckdb.connect("typed.duckdb")
    con.execute("CREATE TABLE s (id INTEGER, date DATE, sales DOUBLE, region VARCHAR, notes VARCHAR)")
    con.execute(
        "INSERT INTO s VALUES (1, '2024-01-05', 100, 'North', 'ok'), (2, '2024-01-06', 400000, 'South', 'ok'),"
        " (3, '2024-01-07', 120.5, NULL, 'ok'), (4, '2024-01-08', 90, 'East', NULL), (5, '2024-01-09', 110, 'West', 'x'),"
        " (6, '2024-01-10', 95, 'West', 'x')"
    )
    con.close()
    graph = build_graph(make_checkpointer(str(tmp_path / "cp.sqlite")))
    st = workflow.start_run("duckdb:typed.duckdb#s", CONTRACT, as_of=AS_OF, graph=graph, run_id="t1", auto_approve=True)
    assert st["outcome"] == "passed", st["summary"]
    staged = open_sink("duckdb:typed.duckdb#s").read_staging("t1")
    by_id = {int(r["id"]): r for _, r in staged.iterrows()}
    assert by_id[2]["sales"] == 400  # unit fix
    assert by_id[3]["sales"] == 120.5
    assert by_id[3]["region"] == "Unknown"
    con = duckdb.connect("typed.duckdb", read_only=True)
    assert con.execute("SELECT sales FROM s WHERE id = 2").fetchone()[0] == 400000
    con.close()


def test_sinks_refuse_to_write_anything_but_staging_or_quarantine():
    with pytest.raises(PermissionError):
        check_target("sales")
    check_target("sales__shdi_staging")
    check_target("main.sales__shdi_quarantine")


def test_in_memory_and_credentialed_uris_are_rejected():
    with pytest.raises(SourceURIError):
        parse_uri("duckdb://:memory:#sales")
    with pytest.raises(SourceURIError):
        parse_uri("duckdb:///tmp/x.duckdb")
    with pytest.raises(SourceURIError, match="SHDI_PG_DSN"):
        parse_uri("postgresql://u:secret@h/db")
