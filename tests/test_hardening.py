"""Regression tests for the defects found in the adversarial review of issues 1 to 6.

Each test names the input that used to break. They drive the real code paths:
the graph through workflow.start_run/decide, the CLI as a subprocess, and the
real DuckDB and Postgres sinks.
"""
from __future__ import annotations

import datetime as dt
import os
import subprocess
import sys
import threading
import uuid

import duckdb
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from shdi import workflow
from shdi.audit import AuditLog
from shdi.connectors import SourceURIError, open_sink, open_source, parse_uri
from shdi.connectors.duckdb import DuckDBSink
from shdi.contract import Contract, load_contract
from shdi.graph import build_graph, make_checkpointer
from shdi.values import INVALID, parse

from conftest import AS_OF, CONTRACT, ROOT

PY = sys.executable


def fresh_graph(db):
    return build_graph(make_checkpointer(str(db)))


def contract_with(**columns) -> dict:
    base = {"name": "t", "columns": {"id": {"type": "integer", "nullable": False}}}
    base["columns"].update(columns)
    return base


# Parsing


@pytest.mark.parametrize("text", ["2024-05", "May 2024", "2024", "17 May"])
def test_partial_dates_are_invalid_not_silently_completed(text):
    # dateutil used to fill the missing day or year from today's date.
    assert parse(text, "date").value is INVALID


def test_timezone_timestamps_become_naive_utc():
    assert parse("2024-05-17T10:00:00+05:30", "timestamp").value == dt.datetime(2024, 5, 17, 4, 30)
    # A date column given a timestamp keeps the date part.
    assert parse("2024-05-17T10:00:00", "date").value == dt.date(2024, 5, 17)


def test_int64_overflow_and_non_ascii_digits():
    assert parse("9223372036854775808", "integer").value is INVALID
    assert parse("9223372036854775807", "integer").value == 2**63 - 1
    assert parse("9223372036854775808", "number").value == pytest.approx(9.223372036854776e18)
    # Arabic-Indic digits used to pass \\d and crash int().
    assert parse("١٢", "integer").value is INVALID
    assert parse("١٢", "number").value is INVALID


def test_overflowing_integer_goes_through_the_pipeline_without_a_crash(tmp_path):
    src = tmp_path / "qty.csv"
    src.write_text("id,qty\n1,5\n2,99999999999999999999\n")
    contract = contract_with(qty={"type": "integer"})
    st = workflow.start_run(str(src), contract, as_of=AS_OF, run_id="big", auto_approve=True)
    assert st["status"] == "completed"
    failing = {c["check_id"]: c["failing_rows"] for c in st["checks_before"] if not c["passed"]}
    assert failing.get("type:qty") == [1]


# Contract validation


@pytest.mark.parametrize(
    "column, message",
    [
        ({"type": "number", "unit_factors": [0]}, "unit_factors"),
        ({"type": "number", "unit_factors": [1]}, "unit_factors"),
        ({"type": "string", "min": 3}, "min"),
        ({"type": "boolean", "max": 1}, "max"),
        ({"type": "integer", "min": 1.5}, "whole"),
        ({"type": "integer", "default": "abc"}, "default"),
        ({"type": "string", "default": "Elsewhere", "accepted_values": ["North"]}, "accepted_values"),
    ],
)
def test_contract_rejects_settings_that_crash_later(column, message):
    with pytest.raises(ValidationError, match=message):
        Contract.model_validate(contract_with(x=column))


def test_yaml_date_bounds_are_accepted(tmp_path):
    # An unquoted YAML date is a datetime.date, which used to fail the Union[float, int, str] field.
    p = tmp_path / "c.yaml"
    p.write_text("name: t\ncolumns:\n  d: {type: date, min: 1970-01-01, max: today}\n")
    assert load_contract(p).columns["d"].min == "1970-01-01"


def test_malformed_yaml_is_a_clean_error(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("name: x\ncolumns: [a: {\n")
    with pytest.raises(ValueError, match="not valid YAML"):
        load_contract(p)


# Workflow: locking, retry, run id reuse


def test_concurrent_decides_apply_once(dirty_csv, tmp_path):
    db = tmp_path / "cp.sqlite"
    st = workflow.start_run(str(dirty_csv), CONTRACT, as_of=AS_OF, graph=fresh_graph(db), run_id="race")
    assert st["status"] == "awaiting_approval"
    barrier = threading.Barrier(2)
    results, errors = [], []

    def go(who):
        try:
            barrier.wait()
            results.append(workflow.decide("race", approve_all=True, reviewer=who, graph=fresh_graph(db)))
        except Exception as e:  # pragma: no cover, reported below
            errors.append(e)

    threads = [threading.Thread(target=go, args=(n,)) for n in ("ann", "bob")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors
    assert all(r["status"] == "completed" for r in results)
    applied = AuditLog().events("race", "applied")
    ids = [e["proposal_id"] for e in applied]
    assert len(ids) == len(set(ids)), "a proposal was applied twice"
    approvals = [e for e in AuditLog().events("race") if e["event"] == "decision"]
    assert len({e["reviewer"] for e in approvals}) == 1, "both reviewers' decisions were recorded"


def test_apply_failure_can_be_retried(dirty_csv, tmp_path):
    db = tmp_path / "cp.sqlite"
    workflow.start_run(str(dirty_csv), CONTRACT, as_of=AS_OF, graph=fresh_graph(db), run_id="retry")
    moved = dirty_csv.with_suffix(".moved")
    dirty_csv.rename(moved)
    with pytest.raises(Exception):
        workflow.decide("retry", approve_all=True, reviewer="ann", graph=fresh_graph(db))
    assert workflow.status("retry", fresh_graph(db))["status"] == "interrupted"
    moved.rename(dirty_csv)
    out = workflow.decide("retry", reviewer="ann", graph=fresh_graph(db))
    assert out["status"] == "completed"
    assert out["outcome"] in ("passed", "failed_checks")
    assert not open_sink(f"csv:{dirty_csv}").read_staging("retry").empty


def test_run_id_with_existing_audit_records_is_refused(dirty_csv, tmp_path):
    workflow.start_run(str(dirty_csv), CONTRACT, as_of=AS_OF, graph=fresh_graph(tmp_path / "a.sqlite"),
                       run_id="dup", auto_approve=True)
    assert AuditLog().events("dup")
    with pytest.raises(workflow.WorkflowError, match="audit records"):
        workflow.start_run(str(dirty_csv), CONTRACT, as_of=AS_OF, graph=fresh_graph(tmp_path / "b.sqlite"),
                           run_id="dup")


def test_relative_sink_dir_is_stored_absolute(dirty_csv, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    st = workflow.start_run(str(dirty_csv), CONTRACT, as_of=AS_OF, graph=fresh_graph(tmp_path / "cp.sqlite"),
                            run_id="rel", sink_dir="out")
    assert st["status"] == "awaiting_approval"
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.chdir(other)
    workflow.decide("rel", approve_all=True, reviewer="ann", graph=fresh_graph(tmp_path / "cp.sqlite"))
    assert list((tmp_path / "out").glob("*staging*")), "staging went somewhere other than the folder given at start"
    assert not (other / "out").exists()


# DuckDB: the table position cannot name a file


@pytest.mark.parametrize(
    "table",
    ['"/etc/hosts"', "'/etc/hosts'", "/etc/hosts", "read_csv('/etc/hosts')", "a.b.c", "sales; DROP TABLE sales", "säles"],
)
def test_duckdb_table_must_be_an_identifier(tmp_path, table):
    with pytest.raises(SourceURIError, match="plain identifier"):
        parse_uri(f"duckdb://{tmp_path}/x.duckdb#{table}")
    with pytest.raises(SourceURIError):
        parse_uri(f"postgres:{table}")


def test_duckdb_connections_cannot_read_files(tmp_path):
    from shdi.connectors.duckdb import connect

    secret = tmp_path / "secret.csv"
    secret.write_text("k\nhunter2\n")
    db = tmp_path / "x.duckdb"
    duckdb.connect(str(db)).close()
    # A plain connection reads the file through a quoted name (the original hole).
    plain = duckdb.connect(str(db), read_only=True)
    assert plain.execute(f'SELECT * FROM "{secret}"').fetchall() == [("hunter2",)]
    plain.close()
    # Every connection the connector opens refuses, even if a name got past parse_uri.
    for read_only in (True, False):
        con = connect(str(db), read_only=read_only)
        with pytest.raises(duckdb.Error, match="disabled"):
            con.execute(f'SELECT * FROM "{secret}"').fetchall()
        con.close()


def test_duckdb_sink_schema_widens_across_runs(tmp_path):
    sink = DuckDBSink(str(tmp_path / "x.duckdb"), "sales")
    # Run 1: integer amounts and an all-null note column (DuckDB type NULL).
    sink.write_staging(pd.DataFrame({"id": [1], "amt": [3], "note": pd.Series([None], dtype=object)}), "r1")
    # Run 2: a float amount used to be rounded into BIGINT; a note used to fail
    # on the NULL column; a new column used to break the insert.
    sink.write_staging(pd.DataFrame({"id": [2], "amt": [2.5], "note": ["hi"], "extra": [True]}), "r2")
    # Run 3: text in a numeric column widens it to VARCHAR.
    sink.write_staging(pd.DataFrame({"id": [3], "amt": ["abc"], "note": ["x"]}), "r3")
    out = sink.read_staging()
    by_run = {r: g.iloc[0] for r, g in out.groupby("shdi_run_id")}
    assert float(by_run["r2"]["amt"]) == 2.5
    assert by_run["r2"]["note"] == "hi"
    assert bool(by_run["r2"]["extra"]) is True
    assert by_run["r3"]["amt"] == "abc"


def number_runs():
    """A number column arrives as Int64 when every value is whole, Float64 otherwise;
    a run where a column is all null carries no type for it."""
    return [
        ("r1", pd.DataFrame({"id": [1, 2], "amt": pd.array([1.5, 2.0], dtype="Float64"),
                             "day": [dt.date(2024, 1, 1), dt.date(2024, 1, 2)]})),
        ("r2", pd.DataFrame({"id": [3, 4], "amt": pd.array([3, 4], dtype="Int64"),
                             "day": pd.Series([None, None], dtype=object)})),
        ("r3", pd.DataFrame({"id": [5], "amt": pd.array([None], dtype="Float64"),
                             "day": pd.Series([float("nan")])})),
    ]


def test_duckdb_sink_never_narrows_or_textifies_numbers_and_dates(tmp_path):
    db = tmp_path / "x.duckdb"
    sink = DuckDBSink(str(db), "sales")
    for run, frame in number_runs():
        sink.write_staging(frame, run)
    con = duckdb.connect(str(db), read_only=True)
    types = dict(con.execute("SELECT column_name, data_type FROM information_schema.columns "
                             "WHERE table_name = 'sales__shdi_staging'").fetchall())
    con.close()
    assert types["amt"] == "DOUBLE" and types["day"] == "DATE", types
    assert sorted(float(v) for v in sink.read_staging()["amt"].dropna()) == [1.5, 2.0, 3.0, 4.0]


# CLI exit codes


def run_cli(*args, env_home):
    env = dict(os.environ, SHDI_HOME=str(env_home), SHDI_LLM="off")
    return subprocess.run([PY, "-m", "shdi.cli", *args], capture_output=True, text=True, env=env, cwd=ROOT)


def test_cli_exit_code_is_nonzero_when_checks_fail_or_source_is_stale(dirty_csv, tmp_path, shdi_env):
    strict = load_contract(CONTRACT).model_dump(mode="json")
    # Report, never fix, an amount over max: the run completes with checks failing.
    strict["columns"]["sales"]["checks"]["max"] = {"severity": "error", "on_violation": "report_only"}
    strict_path = tmp_path / "strict.yaml"
    strict_path.write_text(yaml.safe_dump(strict))
    r = run_cli("run", str(dirty_csv), "--contract", str(strict_path), "--auto-approve", "--run-id", "x1",
                "--json", env_home=shdi_env)
    assert r.returncode == 1, r.stdout + r.stderr
    assert '"outcome": "failed_checks"' in r.stdout

    r = run_cli("run", str(dirty_csv), "--contract", str(CONTRACT), "--run-id", "x2", env_home=shdi_env)
    assert r.returncode == 0, r.stderr
    with open(dirty_csv, "a") as f:
        f.write("12,2024-03-13,10,North,late row\n")
    r = run_cli("approve", "x2", "--all", env_home=shdi_env)
    assert r.returncode == 1 and "stale_source" in r.stdout

    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\ncolumns: [a: {\n")
    r = run_cli("run", str(dirty_csv), "--contract", str(bad), env_home=shdi_env)
    assert r.returncode == 2
    assert r.stderr.startswith("error:") and "Traceback" not in r.stderr


# Bench scoring


def test_bench_counts_an_untouched_null_as_missed():
    from bench.corrupt import TEXT
    from bench.score import _outcome

    clean = pd.DataFrame({"id": ["1"], "date": ["2024-01-01"], "sales": ["500"], "region": ["North"], "notes": ["ok"]})
    label = {"source_row": 0, "defect_type": "null", "column": "region", "injected_value": ""}
    staged = {0: {"id": "1", "date": "2024-01-01", "sales": "500", "region": None, "notes": "ok"}}
    assert _outcome(TEXT, label, staged, set(), clean, {"region": "Unknown"}) == "missed"
    staged[0]["region"] = "Unknown"
    assert _outcome(TEXT, label, staged, set(), clean, {"region": "Unknown"}) == "neutralized"


# Postgres (needs SHDI_PG_DSN; see docker-compose.test.yml)

pg = pytest.mark.skipif(not os.environ.get("SHDI_PG_DSN"), reason="SHDI_PG_DSN not set")


@pg
def test_postgres_connect_error_does_not_leak_the_dsn(monkeypatch):
    from shdi.connectors.postgres import PostgresSource

    monkeypatch.setenv("SHDI_PG_DSN", "postgresql://shdi_reader:s3cretPassw0rd@127.0.0.1:1/nope?sslmode=bogus")
    with pytest.raises(RuntimeError) as e:
        PostgresSource("public.x").read()
    assert "s3cretPassw0rd" not in str(e.value)
    assert e.value.__cause__ is None and e.value.__suppress_context__


@pg
def test_postgres_sink_widens_and_handles_odd_names():
    import psycopg
    from psycopg import sql

    from shdi.connectors.postgres import PostgresSink

    writer = os.environ.get("SHDI_PG_SINK_DSN") or os.environ["SHDI_PG_DSN"]
    schema = f"shdi_h_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(writer, autocommit=True) as con:
        con.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        sink = PostgresSink(f"{schema}.Sales")  # mixed case: to_regclass needs the quoted form
        sink.write_staging(pd.DataFrame({"id": [1], "amt": [3], "note": pd.Series([None], dtype=object)}), "r1")
        sink.write_staging(pd.DataFrame({"id": [2], "amt": [2.5], "note": ["hi"], "extra": [True]}), "r2")
        out = sink.read_staging()
        assert len(out) == 2, "mixed-case table was not found on read"
        r2 = out[out.shdi_run_id == "r2"].iloc[0]
        assert float(r2["amt"]) == 2.5 and r2["note"] == "hi" and r2["extra"] is True
        with pytest.raises(ValueError, match="63 bytes"):
            PostgresSink(f"{schema}.{'t' * 60}").write_staging(pd.DataFrame({"id": [1]}), "r1")
    finally:
        with psycopg.connect(writer, autocommit=True) as con:
            con.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


@pg
def test_postgres_sink_never_narrows_or_textifies_numbers_and_dates():
    import psycopg
    from psycopg import sql

    from shdi.connectors.postgres import PostgresSink

    writer = os.environ.get("SHDI_PG_SINK_DSN") or os.environ["SHDI_PG_DSN"]
    schema = f"shdi_h_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(writer, autocommit=True) as con:
        con.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
    try:
        sink = PostgresSink(f"{schema}.sales")
        for run, frame in number_runs():
            sink.write_staging(frame, run)
        with psycopg.connect(writer) as con:
            types = dict(con.execute(
                "SELECT column_name, data_type FROM information_schema.columns WHERE table_schema = %s "
                "AND table_name = 'sales__shdi_staging'", [schema]).fetchall())
        assert types["amt"] == "double precision" and types["day"] == "date", types
    finally:
        with psycopg.connect(writer, autocommit=True) as con:
            con.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


def test_postgres_name_too_long_for_the_suffix_is_refused_up_front():
    parse_uri(f"postgres:public.{'t' * 46}")  # 46 + len("__shdi_quarantine") == 63
    with pytest.raises(SourceURIError, match="too long"):
        parse_uri(f"postgres:public.{'t' * 47}")


@pg
def test_postgres_source_with_unorderable_column():
    import psycopg
    from psycopg import sql

    writer = os.environ.get("SHDI_PG_SINK_DSN") or os.environ["SHDI_PG_DSN"]
    reader = psycopg.conninfo.conninfo_to_dict(os.environ["SHDI_PG_DSN"]).get("user")
    schema = f"shdi_h_{uuid.uuid4().hex[:8]}"
    with psycopg.connect(writer, autocommit=True) as con:
        con.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        con.execute(sql.SQL("CREATE TABLE {}.t (id int, doc json, p point)").format(sql.Identifier(schema)))
        con.execute(sql.SQL("""INSERT INTO {}.t VALUES (2, '{{"a":1}}', '(1,2)'), (1, '[]', '(0,0)')""").format(
            sql.Identifier(schema)))
        con.execute(sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(sql.Identifier(schema), sql.Identifier(reader)))
        con.execute(sql.SQL("GRANT SELECT ON {}.t TO {}").format(sql.Identifier(schema), sql.Identifier(reader)))
    try:
        df = open_source(f"postgres:{schema}.t").read()
        assert list(df["id"]) == [1, 2]
        # The whole run, with json and point columns outside the contract.
        contract = {"name": "t", "columns": {"id": {"type": "integer"}}}
        st = workflow.start_run(f"postgres:{schema}.t", contract, as_of=AS_OF, run_id="odd", auto_approve=True)
        assert st["status"] == "completed" and st["outcome"] == "passed"
        staged = open_sink(f"postgres:{schema}.t").read_staging("odd")
        assert sorted(staged["doc"]) == ['[]', '{"a": 1}']
        assert sorted(staged["p"]) == ["(0,0)", "(1,2)"]
    finally:
        with psycopg.connect(writer, autocommit=True) as con:
            con.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
