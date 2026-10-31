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



# Second review round. R1: integer text used to go through a float.


@pytest.mark.parametrize("text,want", [
    ("1000000000000000000.5", None),
    ("9007199254740993.5", None),
    ("9007199254740993.0", 9007199254740993),
    ("1e5", 100000),
    ("0e30", 0),
    ("1e999999999", None),
])
def test_r1_integer_text_is_parsed_exactly(text, want):
    from shdi.values import INVALID, parse_integer

    got = parse_integer(text).value
    assert (got is INVALID) if want is None else (got == want and isinstance(got, int))


def test_r1_number_text_keeps_huge_and_tiny_values_exact():
    from shdi.values import INVALID, parse_number

    assert parse_number("1e400").value == Decimal("1e400")
    assert parse_number("1e-400").value == Decimal("1e-400")
    assert parse_number("1e5").value == 100000 and isinstance(parse_number("1e5").value, int)
    assert parse_number("1e999999999").value is INVALID


def test_r1_fractional_text_fails_the_integer_type_check(tmp_path):
    path = tmp_path / "s.csv"
    path.write_text("id,n\n1,1000000000000000000.5\n2,7\n")
    from shdi.checks import run_checks
    from shdi.contract import contract_from_dict

    c = contract_from_dict(contract(n={"type": "integer"}))
    checks = {x.check_id: x for x in run_checks(open_source(str(path)).read(), c, AS_OF)}
    assert not checks["type:n"].passed


# R2: a column widened to DOUBLE rounded integers past 2**53 already staged.


@pytest.mark.parametrize("first,second", [
    ([9007199254740993], [1.5]),
    ([1.5], [9007199254740993]),
])
def test_r2_duckdb_sink_never_rounds_earlier_or_new_integers(tmp_path, first, second):
    db = tmp_path / "w.duckdb"
    make_duckdb(db, "x INTEGER", [])
    sink = DuckDBSink(str(db), "t")
    for run, values in (("a", first), ("b", second)):
        df = pd.DataFrame({"x": pd.Series(values)}, index=range(len(values)))
        sink.write_run(df, pd.DataFrame(), run)
    got = {r: v for r, v in ((r, sink.read_staging(r)["x"].iloc[0]) for r in ("a", "b"))}
    assert same_value(got["a"], first[0], "number") and same_value(got["b"], second[0], "number"), got


# R5: a fix in a timestamptz column used to store a naive datetime.


def test_r5_timestamp_fix_in_an_aware_column_stays_aware():
    from shdi.checks import run_checks
    from shdi.contract import contract_from_dict
    from shdi.repairs import apply_proposals, propose_repairs

    aware = dt.datetime(2024, 1, 1, 5, tzinfo=dt.timezone.utc)
    df = pd.DataFrame({"id": [1, 2], "ts": pd.Series([aware, None], dtype=object)})
    c = contract_from_dict(contract(ts={"type": "timestamp", "nullable": False, "default": "2024-02-02T00:00:00Z"}))
    props = [p.to_dict() for p in propose_repairs(df, c, run_checks(df, c, AS_OF), AS_OF)]
    out = apply_proposals(df, c, props, [p["id"] for p in props]).output
    assert all(isinstance(v, dt.datetime) and v.tzinfo is not None for v in out["ts"]), out["ts"].tolist()


# R6: CSV write_run replaced staging before the quarantine step could fail.


def test_r6_csv_write_run_is_all_or_nothing_and_leaves_no_temp_files(tmp_path, monkeypatch):
    import shdi.connectors.csv as csvmod

    sink = CsvSink(tmp_path, "s")
    q = pd.DataFrame({"id": ["9"], "shdi_source_row": [0]})
    sink.write_run(pd.DataFrame({"id": ["1"]}), q, "R1")
    before = {p.name: p.read_bytes() for p in tmp_path.glob("*.csv")}

    real, calls = csvmod.to_csv_bytes, []

    def fail_second(df):
        calls.append(1)
        if len(calls) == 2:
            raise OSError("disk full")
        return real(df)

    monkeypatch.setattr(csvmod, "to_csv_bytes", fail_second)
    with pytest.raises(OSError, match="disk full"):
        sink.write_run(pd.DataFrame({"id": ["2"]}), q, "R2")
    assert {p.name: p.read_bytes() for p in tmp_path.glob("*.csv")} == before
    assert not list(tmp_path.glob("*.tmp")) and not list(tmp_path.glob(".*.tmp"))


# R7: header checks were bypassed by a BOM, a trailing comma and other encodings.


@pytest.mark.parametrize("raw,match", [
    ("﻿id,id\n1,2\n".encode("utf-8"), "duplicate|repeats"),
    (b"a,b\n1,2,\n", "fields"),
    ("id,name\n1,café\n".encode("latin-1"), "UTF-8"),
])
def test_r7_csv_header_and_encoding_problems_are_clean_errors(tmp_path, raw, match):
    path = tmp_path / "s.csv"
    path.write_bytes(raw)
    with pytest.raises(ValueError, match=match):
        open_source(str(path)).read()


def test_r7_bom_is_not_part_of_the_first_column_name(tmp_path):
    path = tmp_path / "s.csv"
    path.write_bytes("﻿id,v\n1,2\n".encode("utf-8"))
    assert list(open_source(str(path)).read().columns) == ["id", "v"]


# R3, R4: Postgres numeric columns met floats through float8 (15 digits) or text.


@pg
def test_r3_r4_postgres_numeric_columns_take_floats_exactly(monkeypatch):
    import psycopg
    from psycopg import sql

    from shdi.connectors.postgres import PostgresSink

    writer = os.environ.get("SHDI_PG_SINK_DSN") or os.environ["SHDI_PG_DSN"]
    schema = f"shdi_r3_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(writer, autocommit=True) as con:
        con.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        sink = PostgresSink(f"{schema}.m")
        # A clamp to a float inside a Decimal column (R3), then a float that
        # needs 17 digits into the numeric column (R4).
        sink.write_staging(pd.DataFrame({"amt": pd.Series([Decimal("5.25"), 99.5], dtype=object)}), "a")
        sink.write_staging(pd.DataFrame({"amt": [0.30000000000000004]}), "b")
        # A double column holding 17-digit values, then a big integer (R2 in Postgres).
        sink.write_staging(pd.DataFrame({"d": [0.30000000000000004]}), "c")
        sink.write_staging(pd.DataFrame({"d": pd.Series([9007199254740993], dtype=object)}), "d")
        with psycopg.connect(writer) as con:
            types = dict(con.execute(
                "SELECT column_name, data_type FROM information_schema.columns WHERE table_schema = %s "
                "AND table_name = 'm__shdi_staging'", [schema]).fetchall())
            rows = dict(con.execute(sql.SQL(
                "SELECT shdi_run_id, coalesce(amt::text, '') || '|' || coalesce(d::text, '') "
                "FROM {}.m__shdi_staging ORDER BY shdi_run_id, shdi_source_row").format(sql.Identifier(schema))).fetchall())
            amt_a = [r[0] for r in con.execute(sql.SQL(
                "SELECT amt::text FROM {}.m__shdi_staging WHERE shdi_run_id = 'a' ORDER BY shdi_source_row").format(
                    sql.Identifier(schema))).fetchall()]
        assert types["amt"] == "numeric" and types["d"] == "numeric", types
        assert [Decimal(x) for x in amt_a] == [Decimal("5.25"), Decimal("99.5")]
        assert rows["b"] == "0.30000000000000004|"
        assert rows["c"] == "|0.30000000000000004"
        assert rows["d"] == "|9007199254740993"
    finally:
        with psycopg.connect(writer, autocommit=True) as con:
            con.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
