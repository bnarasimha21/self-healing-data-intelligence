"""Issue 1 (human approval with a durable checkpoint) and Issue 2 end to end,
driven through the real compiled LangGraph graph."""
from __future__ import annotations

import sqlite3

import pytest

from shdi import workflow
from shdi.audit import AuditLog, repair_log
from shdi.checks import run_checks
from shdi.connectors import open_sink
from shdi.connectors.base import to_csv_bytes
from shdi.connectors.csv import read_csv_text
from shdi.contract import load_contract
from shdi.graph import build_graph, make_checkpointer
from shdi.repairs import propose_repairs

from conftest import AS_OF, CONTRACT, FIXTURES, find

def fresh_graph(db):
    return build_graph(make_checkpointer(str(db)))


def start(source, db, **kw):
    return workflow.start_run(str(source), CONTRACT, as_of=AS_OF, graph=fresh_graph(db), run_id="r1", **kw)


def staging_without_meta(source, run_id="r1"):
    df = open_sink(f"csv:{source}").read_staging(run_id)
    return df[[c for c in df.columns if not c.startswith("shdi_")]]


def failing_check_ids(checks):
    return {c["check_id"] for c in checks if not c["passed"]}


def test_run_pauses_before_any_write(dirty_csv, tmp_path, shdi_env):
    st = start(dirty_csv, tmp_path / "cp.sqlite")
    assert st["status"] == "awaiting_approval"
    assert st["needs_approval"], "the sample must have proposals that need a human"
    assert not (shdi_env / "out").exists()
    assert AuditLog().events("r1", "applied") == []
    assert st["proposals"][0]["rule"] == "cast"
    # Every proposal is pending: nothing has been decided yet.
    assert {p["id"] for p in st["pending"]} == {p["id"] for p in st["proposals"]}


def test_quarantine_proposal_for_text_in_numeric_column(dirty_csv, tmp_path):
    st = start(dirty_csv, tmp_path / "cp.sqlite")
    q = find(st["proposals"], "quarantine_row", row=2)
    assert q["row_keys"] == [{"_row": 2, "id": 3}]
    assert q["risk"] == "needs_approval"
    assert "type:sales" in q["check_ids"]
    assert not any(p["rule"] in ("clamp", "fill_default", "rescale") and 2 in p["rows"] for p in st["proposals"])


def test_resume_in_a_new_graph_applies_exactly_the_approved_changes(dirty_csv, tmp_path):
    db = tmp_path / "cp.sqlite"
    st = start(dirty_csv, db)
    props = st["proposals"]
    quarantine = find(props, "quarantine_row", row=2)
    dup = find(props, "drop_duplicate")
    fill = find(props, "fill_default", column="region")
    clamp = find(props, "clamp", column="sales")
    approve = [quarantine["id"], dup["id"], fill["id"], clamp["id"]]
    reject = [p for p in st["needs_approval"] if p not in approve]
    assert len(reject) >= 3, "half rejected: date cast, date nulls and the rescale"

    # A brand new graph object and checkpointer connection on the same file,
    # as if the process had restarted between start and review.
    out = workflow.decide("r1", approved=approve, rejected=reject, reviewer="alice", graph=fresh_graph(db))
    assert out["status"] == "completed"

    decisions = out["decisions"]
    for pid in approve:
        assert decisions[pid]["decision"] == "approved" and decisions[pid]["reviewer"] == "alice"
    for pid in reject:
        assert decisions[pid]["decision"] == "rejected"
    for p in props:
        if p["risk"] == "low":
            assert decisions[p["id"]] == {**decisions[p["id"]], "decision": "approved", "reviewer": "policy:low-risk"}

    # Expected output: the source with only the approved edits applied.
    src = read_csv_text(FIXTURES / "dirty_sales.csv")
    exp = src.drop(index=quarantine["rows"] + dup["rows"]).copy()
    exp.loc[fill["rows"][0], "region"] = "Unknown"
    exp.loc[clamp["rows"][0], "sales"] = "0"
    got = staging_without_meta(dirty_csv)
    assert to_csv_bytes(got) == to_csv_bytes(exp.reset_index(drop=True))
    # Rejected changes did not happen: 400000, 2050 and 1960 are still there.
    assert {"400000"} <= set(got["sales"])
    assert {"2050-12-01", "1960-01-01", "01/06/2024"} <= set(got["date"])

    # One decision record per decision, with the row keys and values.
    audit = AuditLog()
    dec_events = audit.events("r1", "decision")
    assert sorted(e["proposal_id"] for e in dec_events) == sorted(p["id"] for p in props)
    assert len(dec_events) == len(props)
    by_pid = {e["proposal_id"]: e for e in dec_events}
    assert by_pid[fill["id"]]["row_keys"] == [{"_row": 10, "id": 10}]
    assert by_pid[fill["id"]]["old_values"] == [None] and by_pid[fill["id"]]["new_values"] == ["Unknown"]
    assert by_pid[clamp["id"]]["row_keys"] == [{"_row": 11, "id": 11}]
    assert by_pid[clamp["id"]]["old_values"] == ["-5"] and by_pid[clamp["id"]]["new_values"] == [0]
    assert by_pid[dup["id"]]["row_keys"] == [{"_row": 5, "id": 5}]
    assert by_pid[dup["id"]]["reviewer"] == "alice"
    applied = {e["proposal_id"] for e in audit.events("r1", "applied")}
    assert applied == set(approve) | {p["id"] for p in props if p["risk"] == "low"}
    assert any("clamp" in line and "-> applied" in line for line in repair_log("r1"))


def test_partial_decision_pauses_again_and_repeat_decisions_are_idempotent(dirty_csv, tmp_path):
    db = tmp_path / "cp.sqlite"
    st = start(dirty_csv, db)
    first, rest = st["needs_approval"][0], st["needs_approval"][1:]
    st2 = workflow.decide("r1", approved=[first], reviewer="bob", graph=fresh_graph(db))
    assert st2["status"] == "awaiting_approval"
    assert first not in st2["needs_approval"] and set(st2["needs_approval"]) == set(rest)
    st3 = workflow.decide("r1", rejected=[first], reviewer="mallory", graph=fresh_graph(db))
    assert st3["ignored"] == [first]
    assert st3["decisions"][first]["decision"] == "approved"
    final = workflow.decide("r1", approve_all=True, reviewer="bob", graph=fresh_graph(db))
    assert final["status"] == "completed"
    n_events = len(AuditLog().events("r1"))
    again = workflow.decide("r1", approved=[first], reviewer="bob", graph=fresh_graph(db))
    assert again["ignored"] == [first]
    assert len(AuditLog().events("r1")) == n_events
    with pytest.raises(workflow.WorkflowError):
        workflow.decide("r1", approved=["p999"], graph=fresh_graph(db))


def test_reject_all_writes_output_byte_identical_to_input(dirty_csv, tmp_path):
    db = tmp_path / "cp.sqlite"
    start(dirty_csv, db)
    out = workflow.decide("r1", reject_all=True, reviewer="carol", graph=fresh_graph(db))
    assert out["status"] == "completed"
    assert out["apply"]["applied"] == [] and out["apply"]["rows_quarantined"] == 0
    assert to_csv_bytes(staging_without_meta(dirty_csv)) == (FIXTURES / "dirty_sales.csv").read_bytes()
    assert AuditLog().events("r1", "applied") == []


def test_source_changed_during_pause_is_refused(dirty_csv, tmp_path):
    db = tmp_path / "cp.sqlite"
    start(dirty_csv, db)
    with open(dirty_csv, "a") as f:
        f.write("12,2024-03-13,10,North,late row\n")
    out = workflow.decide("r1", approve_all=True, reviewer="dave", graph=fresh_graph(db))
    assert out["status"] == "completed"
    assert out["outcome"] == "stale_source"
    assert open_sink(f"csv:{dirty_csv}").read_staging("r1").empty
    skipped = AuditLog().events("r1", "skipped")
    assert skipped and AuditLog().events("r1", "applied") == []


def test_audit_log_is_append_only(dirty_csv, tmp_path, shdi_env):
    start(dirty_csv, tmp_path / "cp.sqlite", auto_approve=True)
    con = sqlite3.connect(str(shdi_env / "audit.sqlite"))
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        con.execute("UPDATE audit_events SET reviewer = 'x'")
    with pytest.raises(sqlite3.DatabaseError, match="append-only"):
        con.execute("DELETE FROM audit_events")
    con.close()


def test_propose_repairs_does_not_mutate_its_input():
    df = read_csv_text(FIXTURES / "dirty_sales.csv")
    before = df.copy(deep=True)
    contract = load_contract(CONTRACT)
    propose_repairs(df, contract, run_checks(df, contract, AS_OF), AS_OF)
    assert df.equals(before)
