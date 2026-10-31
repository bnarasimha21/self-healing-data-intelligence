"""Issue 5: the MCP server, driven by the official SDK client over stdio.

The server is a real subprocess (`python -m shdi.mcp_server`); the test talks
to it with `mcp.ClientSession`, exactly as Claude or Cursor would.
"""
from __future__ import annotations

import os
import shutil
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

import anyio
import duckdb
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from shdi.audit import AuditLog
from shdi.connectors import open_sink
from shdi.connectors.base import to_csv_bytes
from shdi.connectors.csv import read_csv_text

from conftest import AS_OF, CONTRACT, FIXTURES, ROOT, find
from db_e2e import _by_id, _without_meta

sys.path.insert(0, str(ROOT / "examples"))
from seed_duckdb import seed  # noqa: E402

TOOLS = {"profile_source", "validate", "propose_repairs", "list_pending", "decide", "get_audit"}
EXPECTED_FAILING = {
    "format:date", "min:date", "max:date", "type:sales", "min:sales", "max:sales",
    "not_null:region", "primary_key:id", "duplicate_rows",
}


@pytest.fixture
def setup(tmp_path, shdi_env):
    allowed = (tmp_path / "allowed").resolve()
    other = (tmp_path / "other").resolve()
    contracts = (tmp_path / "contracts").resolve()
    for d in (allowed, other, contracts):
        d.mkdir()
    shutil.copy(CONTRACT, contracts / "sales.yaml")
    db = allowed / "landing.duckdb"
    assert seed(str(FIXTURES / "dirty_sales.csv"), str(db), "sales") == 12
    off = other / "secret.duckdb"
    assert seed(str(FIXTURES / "dirty_sales.csv"), str(off), "sales") == 12
    env = {
        **os.environ,
        "SHDI_HOME": str(shdi_env),
        "SHDI_ALLOWED_SOURCES": f"duckdb://{allowed}/*.duckdb#*",
        "SHDI_CONTRACT_DIR": str(contracts),
        "SHDI_LLM": "off",
    }
    return {"db": db, "uri": f"duckdb://{db}#sales", "other": off, "env": env, "tmp": tmp_path}


@asynccontextmanager
async def session(env: Dict[str, str]):
    params = StdioServerParameters(command=sys.executable, args=["-m", "shdi.mcp_server"], env=env, cwd=str(ROOT))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as s:
            await s.initialize()
            yield s


async def call(s: ClientSession, tool: str, **args) -> Dict[str, Any]:
    r = await s.call_tool(tool, args)
    assert not r.is_error, f"{tool} failed: {r.content}"
    return r.structured_content


async def call_error(s: ClientSession, tool: str, **args) -> str:
    r = await s.call_tool(tool, args)
    assert r.is_error, f"{tool} should have failed: {r.structured_content}"
    return " ".join(getattr(c, "text", "") for c in r.content)


def tables(path: Path) -> set:
    con = duckdb.connect(str(path), read_only=True)
    try:
        return {r[0] for r in con.execute("SELECT table_name FROM information_schema.tables").fetchall()}
    finally:
        con.close()


def test_sources_off_the_allowlist_and_bad_contracts_are_rejected(setup):
    other_uri = f"duckdb://{setup['other']}#sales"
    sneaky = f"duckdb://{setup['db'].parent}/../other/secret.duckdb#sales"

    async def main():
        async with session(setup["env"]) as s:
            for tool, args in [
                ("propose_repairs", {"uri": other_uri, "contract": "sales"}),
                ("validate", {"uri": other_uri, "contract": "sales"}),
                ("profile_source", {"uri": other_uri}),
                ("propose_repairs", {"uri": sneaky, "contract": "sales"}),
            ]:
                msg = await call_error(s, tool, **args)
                assert "not on the allowlist" in msg and "other/secret.duckdb" in msg
            msg = await call_error(s, "profile_source", uri="postgresql://u:pw@db.example.com/prod")
            assert "SHDI_PG_DSN" in msg
            for bad in ("../contracts/sales", "/etc/passwd", "sales/../../x"):
                msg = await call_error(s, "validate", uri=setup["uri"], contract=bad)
                assert "contract must be a file name" in msg
            assert "not found" in await call_error(s, "validate", uri=setup["uri"], contract="nope")
            assert "unknown run" in await call_error(s, "list_pending", thread_id="nope")
            assert "unknown run" in await call_error(s, "decide", thread_id="nope", approved_ids=["p1"])

        no_list = {k: v for k, v in setup["env"].items() if k != "SHDI_ALLOWED_SOURCES"}
        async with session(no_list) as s:
            assert "no sources are allowed" in await call_error(s, "profile_source", uri=setup["uri"])

    anyio.run(main)
    assert tables(setup["other"]) == {"sales"}, "a rejected source must not be touched"
    assert tables(setup["db"]) == {"sales"}


def test_propose_decide_subset_and_audit_over_stdio(setup):
    uri, db = setup["uri"], setup["db"]

    async def main():
        async with session(setup["env"]) as s:
            listed = {t.name for t in (await s.list_tools()).tools}
            assert TOOLS <= listed

            prof = await call(s, "profile_source", uri=uri)
            assert prof["rows"] == 12
            cols = {c["name"]: c for c in prof["columns"]}
            assert list(cols) == ["id", "date", "sales", "region", "notes"]
            assert cols["region"]["nulls"] == 1 and cols["id"]["inferred_type"] == "integer"

            val = await call(s, "validate", uri=uri, contract="sales", as_of=AS_OF.isoformat())
            assert set(val["failing"]) == EXPECTED_FAILING and not val["passed"]
            by_check = {c["check_id"]: c for c in val["checks"]}
            assert by_check["type:sales"]["failing_rows"] == [2]
            assert tables(db) == {"sales"}, "validate is read-only"

            run = await call(s, "propose_repairs", uri=uri, contract="sales.yaml", as_of=AS_OF.isoformat())
            tid = run["thread_id"]
            assert run["status"] == "awaiting_approval" and run["run_id"] == tid
            assert tables(db) == {"sales"}, "propose_repairs writes nothing"
            props = run["proposals"]
            quarantine = find(props, "quarantine_row", row=2)
            dup = find(props, "drop_duplicate")
            fill = find(props, "fill_default", column="region")
            clamp = find(props, "clamp", column="sales")
            approve = [quarantine["id"], dup["id"], fill["id"], clamp["id"]]
            reject = [p for p in run["needs_approval"] if p not in approve]
            assert reject, "some proposals are left for the second call"

            # First call: a subset. The run stays paused and still writes nothing.
            mid = await call(s, "decide", thread_id=tid, approved_ids=approve, reviewer="alice")
            assert mid["status"] == "awaiting_approval"
            assert set(mid["needs_approval"]) == set(reject)
            assert tables(db) == {"sales"}

            # Deciding the same proposal again (even the other way) is ignored.
            again = await call(s, "decide", thread_id=tid, rejected_ids=[quarantine["id"]], reviewer="mallory")
            assert again["ignored"] == [quarantine["id"]]
            assert again["decisions"][quarantine["id"]]["decision"] == "approved"

            pending = await call(s, "list_pending", thread_id=tid)
            assert set(pending["needs_approval"]) == set(reject)

            done = await call(s, "decide", thread_id=tid, rejected_ids=reject, reviewer="alice")
            assert done["status"] == "completed"
            assert done["apply"]["rows_quarantined"] == 1 and done["apply"]["rows_dropped"] == 1

            staged_once = to_csv_bytes(_by_id(_without_meta(open_sink(uri).read_staging(tid))))
            audit_once = await call(s, "get_audit", run_id=tid)

            # Repeating a decision after the run applied changes nothing.
            repeat = await call(s, "decide", thread_id=tid, approved_ids=[clamp["id"]], reviewer="alice")
            assert repeat["ignored"] == [clamp["id"]] and repeat["status"] == "completed"
            audit_twice = await call(s, "get_audit", run_id=tid)
            assert audit_twice["count"] == audit_once["count"]
            return tid, props, approve, reject, quarantine, dup, fill, clamp, staged_once, audit_twice

    tid, props, approve, reject, quarantine, dup, fill, clamp, staged_once, audit = anyio.run(main)

    # Staging holds the source with exactly the approved edits.
    src = read_csv_text(FIXTURES / "dirty_sales.csv")
    exp = src.drop(index=quarantine["rows"] + dup["rows"]).copy()
    exp.loc[fill["rows"][0], "region"] = "Unknown"
    exp.loc[clamp["rows"][0], "sales"] = "0"
    sink = open_sink(setup["uri"])
    staged = sink.read_staging(tid)
    assert to_csv_bytes(_by_id(_without_meta(staged))) == to_csv_bytes(_by_id(exp.reset_index(drop=True)))
    assert staged_once == to_csv_bytes(_by_id(_without_meta(staged))), "no double apply"
    q = sink.read_quarantine(tid)
    assert [str(v) for v in q["id"]] == ["3"] and list(q["shdi_proposal_id"]) == [quarantine["id"]]
    assert tables(setup["db"]) == {"sales", "sales__shdi_staging", "sales__shdi_quarantine"}

    # Audit: one decision per proposal, matching what the client sent.
    decisions = [e for e in audit["events"] if e["event"] == "decision"]
    assert sorted(e["proposal_id"] for e in decisions) == sorted(p["id"] for p in props)
    by_pid = {e["proposal_id"]: e for e in decisions}
    for pid in approve:
        assert by_pid[pid]["decision"] == "approved" and by_pid[pid]["reviewer"] == "alice"
    for pid in reject:
        assert by_pid[pid]["decision"] == "rejected" and by_pid[pid]["reviewer"] == "alice"
    for p in props:
        if p["risk"] == "low":
            assert by_pid[p["id"]]["reviewer"] == "policy:low-risk"
    assert by_pid[clamp["id"]]["old_values"] == ["-5"] and by_pid[clamp["id"]]["new_values"] == [0]
    applied = {e["proposal_id"] for e in audit["events"] if e["event"] == "applied"}
    assert applied == set(approve) | {p["id"] for p in props if p["risk"] == "low"}
    # The server and this process read the same audit log.
    assert len(AuditLog().events(tid)) == audit["count"]


def test_a_clean_source_pauses_and_only_decide_writes(setup):
    csv = setup["tmp"] / "clean.csv"
    csv.write_text("id,date,sales,region,notes\n1,2024-01-05,100,North,ok\n2,2024-02-05,200,South,promo\n3,2024-03-05,300,East,ok\n")
    db = setup["db"].parent / "clean.duckdb"
    seed(str(csv), str(db), "sales")
    uri = f"duckdb://{db}#sales"

    async def main():
        async with session(setup["env"]) as s:
            val = await call(s, "validate", uri=uri, contract="sales", as_of=AS_OF.isoformat())
            assert val["passed"] and val["outliers"] == []
            run = await call(s, "propose_repairs", uri=uri, contract="sales", as_of=AS_OF.isoformat())
            assert run["status"] == "awaiting_approval" and run["needs_approval"] == []
            assert all(p["risk"] == "low" and p["row_count"] == 0 for p in run["proposals"])
            assert tables(db) == {"sales"}
            done = await call(s, "decide", thread_id=run["thread_id"], reviewer="alice")
            assert done["status"] == "completed" and done["outcome"] == "passed"
            assert done["before_after"]["cells_changed"] == 0
            return run["thread_id"]

    tid = anyio.run(main)
    staged = open_sink(uri).read_staging(tid)
    assert len(staged) == 3
