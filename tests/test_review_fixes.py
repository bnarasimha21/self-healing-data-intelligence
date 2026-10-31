"""Regression tests for the second, independent adversarial review.

Each test is named after the finding it covers (F1 to F11) and uses the input
the review gave. They go through the real graph (workflow.start_run/decide)
and the real sinks; the Postgres ones need SHDI_PG_DSN and SHDI_PG_SINK_DSN.
"""
from __future__ import annotations

import datetime as dt
import os
import subprocess
import sys
import threading
import uuid
from decimal import Decimal
from urllib.parse import quote

import duckdb
import pandas as pd
import pytest

from shdi import workflow
from shdi.audit import AuditLog
from shdi.connectors import open_sink, open_source
from shdi.connectors.csv import CsvSink
from shdi.connectors.duckdb import DuckDBSink
from shdi.graph import build_graph, make_checkpointer
from shdi.repairs import CELL_RULES, same_value
from shdi.values import parse_date

from conftest import AS_OF, ROOT

PY = sys.executable
pg = pytest.mark.skipif(not os.environ.get("SHDI_PG_DSN"), reason="SHDI_PG_DSN not set")


def graph(tmp_path, name="cp"):
    return build_graph(make_checkpointer(str(tmp_path / f"{name}.sqlite")))


def make_duckdb(path, ddl, rows, table="t"):
    con = duckdb.connect(str(path))
    try:
        con.execute(f"CREATE TABLE {table} ({ddl})")
        if rows:
            con.executemany(f"INSERT INTO {table} VALUES ({', '.join('?' for _ in rows[0])})", rows)
    finally:
        con.close()


def contract(**columns):
    return {"name": "t", "columns": {"id": {"type": "integer", "nullable": False}, **columns}}


def assert_staging_matches_audit(uri, run_id, contract_dict):
    """Every cell an applied proposal touched holds, in staging, exactly the
    value its audit record says was written."""
    staged = open_sink(uri).read_staging(run_id)
    by_row = {int(r["shdi_source_row"]): r for r in staged.to_dict("records")}
    types = {n: c["type"] for n, c in contract_dict["columns"].items()}
    checked = 0
    for e in AuditLog().events(run_id, "applied"):
        p = e["proposal"]
        if p["rule"] not in CELL_RULES + ("cast", "add_column"):
            continue
        for r, new in zip(p["rows"], p["new_values"]):
            got = by_row[r][p["column"]]
            assert same_value(got, new, types.get(p["column"])), (p["rule"], r, got, new)
            checked += 1
    assert checked
    return staged


# F1: a mixed int/fraction column used to become float64.


def test_f1_big_integer_next_to_a_fraction_stays_exact(tmp_path):
    db = tmp_path / "s.duckdb"
    make_duckdb(db, "id VARCHAR, amt VARCHAR", [["1", "123456789012345678"], ["2", "1.5"], ["3", "1.50"]])
    c = contract(amt={"type": "number"})
    uri = f"duckdb://{db}#t"
    st = workflow.start_run(uri, c, as_of=AS_OF, graph=graph(tmp_path), run_id="f1")
    cast = next(p for p in st["proposals"] if p["rule"] == "cast" and p["column"] == "amt")
    # Every row is listed with its old and new text; "1.50" becomes "1.5", so
    # the cast changes a value and is not low risk.
    assert cast["rows"] == [0, 1, 2]
    assert cast["old_values"] == ["123456789012345678", "1.5", "1.50"]
    assert cast["new_values"] == ["123456789012345678", "1.5", "1.5"]
    assert cast["risk"] == "needs_approval"
    out = workflow.decide("f1", approve_all=True, reviewer="ann", graph=graph(tmp_path))
    assert out["status"] == "completed"
    staged = assert_staging_matches_audit(uri, "f1", c)
    assert Decimal(str(staged.sort_values("id")["amt"].iloc[0])) == Decimal("123456789012345678")


def test_f1_cast_that_changes_nothing_is_low_risk_and_still_lists_rows(tmp_path):
    db = tmp_path / "s.duckdb"
    make_duckdb(db, "id VARCHAR, amt VARCHAR", [["1", "123456789012345678"], ["2", "1.5"]])
    st = workflow.start_run(f"duckdb://{db}#t", contract(amt={"type": "number"}), as_of=AS_OF,
                            graph=graph(tmp_path), run_id="f1b")
    cast = next(p for p in st["proposals"] if p["rule"] == "cast" and p["column"] == "amt")
    assert cast["risk"] == "low" and cast["old_values"] == cast["new_values"] == ["123456789012345678", "1.5"]


# F2: astype truncated a clamp of 99.5 into a BIGINT column to 99.


def test_f2_clamp_into_a_bigint_column_writes_the_audited_value(tmp_path):
    db = tmp_path / "s.duckdb"
    make_duckdb(db, "id BIGINT, amt BIGINT", [[1, 150], [2, 50], [3, 99]])
    c = contract(amt={"type": "number", "max": 99.5})
    uri = f"duckdb://{db}#t"
    st = workflow.start_run(uri, c, as_of=AS_OF, graph=graph(tmp_path), run_id="f2", auto_approve=True)
    assert st["status"] == "completed"
    clamp = [e for e in AuditLog().events("f2", "applied") if e["proposal"]["rule"] == "clamp"]
    assert [e["proposal"]["new_values"] for e in clamp] == [[99.5]]
    staged = assert_staging_matches_audit(uri, "f2", c)
    assert Decimal(str(staged.sort_values("id")["amt"].iloc[0])) == Decimal("99.5")


# F3: .df() turned DECIMAL into DOUBLE, HUGEINT into float, DATE into
# TIMESTAMP and NaN into NULL.


def test_f3_typed_duckdb_values_survive_into_staging_exactly(tmp_path):
    db = tmp_path / "s.duckdb"
    big = 2**70 + 1
    make_duckdb(
        db, "id BIGINT, price DECIMAL(20,4), huge HUGEINT, d DATE, x DOUBLE",
        [[1, Decimal("12345678901234.5678"), big, dt.date(2024, 2, 29), float("nan")],
         [2, Decimal("0.0001"), -big, dt.date(1999, 12, 31), 1.5]],
    )
    c = contract(price={"type": "number"}, d={"type": "date"})
    uri = f"duckdb://{db}#t"
    st = workflow.start_run(uri, c, as_of=AS_OF, graph=graph(tmp_path), run_id="f3", auto_approve=True)
    assert st["status"] == "completed" and st["outcome"] == "passed", st
    con = duckdb.connect(str(db), read_only=True)
    try:
        types = {r[0]: r[1] for r in con.execute("DESCRIBE t__shdi_staging").fetchall()}
        rows = con.execute("SELECT price, CAST(huge AS VARCHAR), d, isnan(x), x IS NULL "
                           "FROM t__shdi_staging ORDER BY id").fetchall()
    finally:
        con.close()
    assert types["price"] == "DECIMAL(20,4)" and types["huge"] == "HUGEINT"
    assert types["d"] == "DATE" and types["x"] == "DOUBLE"
    assert rows[0] == (Decimal("12345678901234.5678"), str(big), dt.date(2024, 2, 29), True, False)
    assert rows[1] == (Decimal("0.0001"), str(-big), dt.date(1999, 12, 31), False, False)


@pytest.mark.parametrize("ddl,before,after", [
    ("v DECIMAL(20,4)", Decimal("12345678901234.5678"), Decimal("12345678901234.5679")),
    ("v HUGEINT", 2**70, 2**70 + 1),
])
def test_f3_fingerprint_sees_changes_a_double_cannot_hold(tmp_path, ddl, before, after):
    db = tmp_path / "s.duckdb"
    make_duckdb(db, f"id BIGINT, {ddl}", [[1, before]])
    src = open_source(f"duckdb://{db}#t")
    fp1 = src.fingerprint()
    con = duckdb.connect(str(db))
    con.execute("UPDATE t SET v = ?", [after])
    con.close()
    assert float(before) == float(after)  # the old float path could not tell them apart
    assert src.fingerprint() != fp1


# F4: staging kept rows from a failed attempt after the source changed.


def test_f4_failed_quarantine_write_then_stale_source_leaves_no_rows(tmp_path, monkeypatch):
    db = tmp_path / "s.duckdb"
    make_duckdb(db, "id VARCHAR, amt VARCHAR", [["1", "5"], ["2", "abc"], ["3", "7"]])
    uri = f"duckdb://{db}#t"
    c = contract(amt={"type": "number", "nullable": False})
    st = workflow.start_run(uri, c, as_of=AS_OF, graph=graph(tmp_path), run_id="f4")
    assert st["status"] == "awaiting_approval"

    real = DuckDBSink._write_in

    def failing(self, con, df, run_id, suffix, src_types):
        if suffix.endswith("quarantine") and df is not None:
            raise RuntimeError("disk full")
        return real(self, con, df, run_id, suffix, src_types)

    monkeypatch.setattr(DuckDBSink, "_write_in", failing)
    with pytest.raises(Exception, match="disk full"):
        workflow.decide("f4", approve_all=True, reviewer="ann", graph=graph(tmp_path))
    monkeypatch.setattr(DuckDBSink, "_write_in", real)
    sink = open_sink(uri)
    # Staging and quarantine are written in one transaction: nothing landed.
    assert sink.read_staging("f4").empty and sink.read_quarantine("f4").empty

    # Even rows an older, non-transactional attempt left behind are removed
    # once the source has changed.
    sink.write_staging(pd.DataFrame({"id": ["1"], "amt": ["5"]}), "f4")
    sink.write_quarantine(pd.DataFrame({"id": ["2"], "amt": ["abc"]}), "f4")
    con = duckdb.connect(str(db))
    con.execute("UPDATE t SET amt = '6' WHERE id = '1'")
    con.close()
    out = workflow.decide("f4", reviewer="ann", graph=graph(tmp_path))
    assert out["outcome"] == "stale_source"
    assert sink.read_staging("f4").empty and sink.read_quarantine("f4").empty


def test_f4_retry_replaces_both_tables(tmp_path):
    sink = DuckDBSink(str(tmp_path / "x.duckdb"), "t")
    sink.write_run(pd.DataFrame({"id": [1, 2]}), pd.DataFrame({"id": [3]}), "r")
    sink.write_run(pd.DataFrame({"id": [1]}), pd.DataFrame(), "r")
    assert list(sink.read_staging("r")["id"]) == [1]
    assert sink.read_quarantine("r").empty


# F5: a rescale in an integer column could produce a fraction.


def test_f5_integer_rescale_only_when_the_result_is_whole(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("id,qty\n1,1500\n2,2000\n")
    c = contract(qty={"type": "integer", "min": 0, "max": 1000, "unit_factors": [1000]})
    st = workflow.start_run(str(p), c, as_of=AS_OF, graph=graph(tmp_path), run_id="f5")
    fixes = {p_["rows"][0]: p_ for p_ in st["proposals"] if p_["column"] == "qty" and p_["rule"] != "cast"}
    assert fixes[0]["rule"] == "clamp" and fixes[0]["new_values"] == [1000]
    assert fixes[1]["rule"] == "rescale" and fixes[1]["new_values"] == [2]


# F6: two writers to the CSV sink lost each other's rows.


def test_f6_concurrent_csv_sink_writers_keep_every_run(tmp_path):
    frame = pd.DataFrame({"id": [str(i) for i in range(300)], "v": ["x" * 20] * 300})
    quarantine = frame.head(3).assign(shdi_source_row=[0, 1, 2], shdi_reason="r")
    errors = []

    def writer(prefix):
        try:
            for k in range(15):
                CsvSink(tmp_path, "s").write_run(frame, quarantine, f"{prefix}{k}")
        except Exception as e:  # pragma: no cover, reported below
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(p,)) for p in ("a", "b", "c")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    sink = CsvSink(tmp_path, "s")
    staged, quarantined = sink.read_staging(), sink.read_quarantine()
    want = {f"{p}{k}" for p in "abc" for k in range(15)}
    assert set(staged["shdi_run_id"]) == want and len(staged) == 300 * len(want)
    assert set(quarantined["shdi_run_id"]) == want and len(quarantined) == 3 * len(want)


# F8: DuckDB TIMESTAMPTZ text and parse_date followed the process timezone.

_TZ_PROBE = """
import datetime as dt, sys
import duckdb
from shdi.connectors.duckdb import DuckDBSource
from shdi.values import parse_date
path = sys.argv[1]
print(DuckDBSource(path, "t").fingerprint())
print(DuckDBSource(path, "t").read()["ts"].astype(str).tolist())
aware = dt.datetime(2024, 1, 1, 20, 0, tzinfo=dt.timezone(dt.timedelta(hours=-5)))
print(parse_date(aware).value, parse_date("2024-01-01T20:00:00-05:00").value)
"""


def test_f8_fingerprint_and_dates_do_not_depend_on_the_process_timezone(tmp_path):
    db = tmp_path / "s.duckdb"
    con = duckdb.connect(str(db))
    con.execute("SET TimeZone = 'UTC'")
    con.execute("CREATE TABLE t (id BIGINT, ts TIMESTAMPTZ)")
    con.execute("INSERT INTO t VALUES (1, '2024-01-01 23:30:00+00'), (2, '2024-06-30 00:15:00+00')")
    con.close()
    outs = {}
    for tz in ("UTC", "Asia/Kolkata", "America/Los_Angeles"):
        env = {**os.environ, "TZ": tz, "PYTHONPATH": str(ROOT)}
        r = subprocess.run([PY, "-c", _TZ_PROBE, str(db)], capture_output=True, text=True, env=env, cwd=ROOT)
        assert r.returncode == 0, r.stderr
        outs[tz] = r.stdout
    assert len(set(outs.values())) == 1, outs
    assert "2024-01-02 2024-01-02" in outs["UTC"]


def test_f8_parse_date_uses_the_utc_date():
    aware = dt.datetime(2024, 1, 1, 22, 0, tzinfo=dt.timezone(dt.timedelta(hours=-5)))
    assert parse_date(aware).value == dt.date(2024, 1, 2)


# F10: casts and add_column must name their rows and old/new values.


def test_f10_cast_and_add_column_audit_records_name_rows_and_values(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("id,amt\n1,5\n2,07\n3,\n")
    c = {**contract(amt={"type": "integer"}, channel={"type": "string", "default": "web"}), "primary_key": ["id"]}
    st = workflow.start_run(str(p), c, as_of=AS_OF, graph=graph(tmp_path), run_id="f10", auto_approve=True)
    assert st["status"] == "completed"
    applied = {e["proposal"]["rule"]: e for e in AuditLog().events("f10", "applied")}
    cast, add = applied["cast"], applied["add_column"]
    assert cast["row_keys"] == [{"_row": 0, "id": 1}, {"_row": 1, "id": 2}]
    assert cast["old_values"] == ["5", "07"] and cast["new_values"] == ["5", "7"]
    assert add["row_keys"] == [{"_row": r, "id": r + 1} for r in range(3)]
    assert add["old_values"] == [None] * 3 and add["new_values"] == ["web"] * 3


# F11: CSV rows with an extra field and duplicate header names.


@pytest.mark.parametrize("body", ["1,5,surprise\n2,6\n", "1,5\n2,6,surprise\n"])
def test_f11_csv_row_with_an_extra_field_is_a_clean_error(tmp_path, body):
    # An extra field on the first data row used to turn the id column into
    # the index and shift every value one column left.
    p = tmp_path / "s.csv"
    p.write_text("id,amt\n" + body)
    with pytest.raises(ValueError, match="fields"):
        open_source(str(p)).read()
    with pytest.raises(ValueError, match="fields"):
        workflow.start_run(str(p), contract(amt={"type": "integer"}), as_of=AS_OF, graph=graph(tmp_path), run_id="f11")


def test_f11_csv_duplicate_header_names_are_rejected(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("id,amt,amt\n1,5,6\n")
    with pytest.raises(ValueError, match="duplicate"):
        open_source(str(p)).read()


# F7: Postgres must store aware datetimes as timestamptz and Decimal as
# numeric, whatever the session TimeZone.


def _with_timezone(dsn: str, tz: str) -> str:
    sep = "&" if "?" in dsn else "?"
    return f"{dsn}{sep}options={quote(f'-c TimeZone={tz}')}"


@pg
def test_f7_postgres_sink_types_and_values_in_a_non_utc_session(monkeypatch):
    import psycopg
    from psycopg import sql

    from shdi.connectors.postgres import PostgresSink, PostgresSource

    writer = os.environ.get("SHDI_PG_SINK_DSN") or os.environ["SHDI_PG_DSN"]
    reader = psycopg.conninfo.conninfo_to_dict(os.environ["SHDI_PG_DSN"]).get("user")
    schema = f"shdi_f7_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(writer, autocommit=True) as con:
        con.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        con.execute(sql.SQL("CREATE TABLE {}.src (id int, ts timestamptz)").format(sql.Identifier(schema)))
        con.execute(sql.SQL("INSERT INTO {}.src VALUES (1, '2024-01-01 23:30:00+00')").format(sql.Identifier(schema)))
        con.execute(sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(sql.Identifier(schema), sql.Identifier(reader)))
        con.execute(sql.SQL("GRANT SELECT ON {}.src TO {}").format(sql.Identifier(schema), sql.Identifier(reader)))
    try:
        fp_utc = PostgresSource(f"{schema}.src").fingerprint()
        monkeypatch.setenv("SHDI_PG_DSN", _with_timezone(os.environ["SHDI_PG_DSN"], "Asia/Kolkata"))
        monkeypatch.setenv("SHDI_PG_SINK_DSN", _with_timezone(writer, "Asia/Kolkata"))
        assert PostgresSource(f"{schema}.src").fingerprint() == fp_utc

        aware = dt.datetime(2024, 1, 1, 23, 30, tzinfo=dt.timezone.utc)
        exact = Decimal("12345678901234567890.123456789")
        sink = PostgresSink(f"{schema}.sales")
        sink.write_staging(pd.DataFrame({"id": [1], "ts": [aware], "amt": [exact], "big": [2**70]}), "r1")
        with psycopg.connect(writer) as con:
            con.execute("SET TIME ZONE 'UTC'")
            types = dict(con.execute(
                "SELECT column_name, data_type FROM information_schema.columns WHERE table_schema = %s "
                "AND table_name = 'sales__shdi_staging'", [schema]).fetchall())
            row = con.execute(sql.SQL("SELECT ts, amt, big::text FROM {}.sales__shdi_staging").format(
                sql.Identifier(schema))).fetchone()
        assert types["ts"] == "timestamp with time zone"
        assert types["amt"] == "numeric" and types["big"] == "numeric"
        assert row[0] == aware and row[1] == exact and row[2] == str(2**70)
    finally:
        with psycopg.connect(writer, autocommit=True) as con:
            con.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


@pg
def test_f4_postgres_write_run_is_one_transaction(monkeypatch):
    import psycopg
    from psycopg import sql

    from shdi.connectors.postgres import PostgresSink

    writer = os.environ.get("SHDI_PG_SINK_DSN") or os.environ["SHDI_PG_DSN"]
    schema = f"shdi_f4_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(writer, autocommit=True) as con:
        con.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        sink = PostgresSink(f"{schema}.sales")
        sink.write_run(pd.DataFrame({"id": [1, 2]}), pd.DataFrame({"id": [3]}), "r")
        real = PostgresSink._write_in

        def failing(self, con, df, run_id, suffix):
            if suffix.endswith("quarantine") and df is not None:
                raise RuntimeError("disk full")
            return real(self, con, df, run_id, suffix)

        monkeypatch.setattr(PostgresSink, "_write_in", failing)
        with pytest.raises(RuntimeError, match="disk full"):
            sink.write_run(pd.DataFrame({"id": [9]}), pd.DataFrame({"id": [8]}), "r")
        monkeypatch.setattr(PostgresSink, "_write_in", real)
        # The failed attempt changed neither table.
        assert sorted(sink.read_staging("r")["id"]) == [1, 2]
        assert list(sink.read_quarantine("r")["id"]) == ["3"]  # quarantine keeps raw text
        sink.delete_run("r")
        assert sink.read_staging("r").empty and sink.read_quarantine("r").empty
    finally:
        with psycopg.connect(writer, autocommit=True) as con:
            con.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))

