"""Shared end-to-end assertions for database sources (DuckDB and Postgres).

The same run is done on the CSV fixture first; the database staging table must
hold exactly the same content, so every connector is held to one reference.
"""
from __future__ import annotations

from typing import Callable

from shdi import workflow
from shdi.connectors import open_sink, open_source
from shdi.connectors.base import to_csv_bytes
from shdi.graph import build_graph, make_checkpointer

from conftest import AS_OF, CONTRACT


def _without_meta(df):
    return df[[c for c in df.columns if not c.startswith("shdi_")]].reset_index(drop=True)


def _by_id(df):
    """Postgres tables have no insertion order (the source sorts on every
    column), so row ordinals differ between backends. Compare by primary key."""
    return df.assign(_k=[int(v) for v in df["id"]]).sort_values("_k", kind="stable").drop(columns="_k").reset_index(drop=True)


def reference_staging(csv_path, tmp_path) -> bytes:
    graph = build_graph(make_checkpointer(str(tmp_path / "ref.sqlite")))
    workflow.start_run(str(csv_path), CONTRACT, as_of=AS_OF, graph=graph, run_id="ref", auto_approve=True,
                       sink_dir=str(tmp_path / "ref_out"))
    return to_csv_bytes(_by_id(_without_meta(open_sink(f"csv:{csv_path}", str(tmp_path / "ref_out")).read_staging("ref"))))


def run_and_check(uri: str, csv_path, tmp_path, source_row_count: Callable[[], int]) -> dict:
    src = open_source(uri)
    hash_before, rows_before = src.fingerprint(), source_row_count()
    assert rows_before == 12

    db = tmp_path / "cp.sqlite"
    st = workflow.start_run(uri, CONTRACT, as_of=AS_OF, graph=build_graph(make_checkpointer(str(db))), run_id="db1")
    assert st["status"] == "awaiting_approval"
    sink = open_sink(uri)
    assert sink.read_staging("db1").empty, "nothing may be written before approval"

    quarantine = [p for p in st["proposals"] if p["rule"] == "quarantine_row" and "type:sales" in p["check_ids"]]
    assert len(quarantine) == 1 and [k["id"] for k in quarantine[0]["row_keys"]] == [3]
    quarantine = quarantine[0]
    done = workflow.decide("db1", approve_all=True, reviewer="db-test", graph=build_graph(make_checkpointer(str(db))))
    assert done["status"] == "completed" and done["outcome"] == "passed", done.get("summary")

    # The source is untouched.
    assert source_row_count() == rows_before
    assert src.fingerprint() == hash_before

    # Staging matches the CSV reference run exactly.
    staged = sink.read_staging("db1")
    assert len(staged) == 10
    assert to_csv_bytes(_by_id(_without_meta(staged))) == reference_staging(csv_path, tmp_path)
    assert len(set(int(v) for v in staged["shdi_source_row"])) == 10

    # The text-polluted row is quarantined with its raw values and the proposal's reason.
    q = sink.read_quarantine("db1")
    assert len(q) == 1
    row = q.iloc[0]
    assert (str(row["id"]), row["sales"]) == ("3", "three hundred")
    assert row["shdi_proposal_id"] == quarantine["id"]
    assert row["shdi_reason"] == quarantine["reason"]
    assert int(row["shdi_source_row"]) == quarantine["rows"][0]
    return done


def rerun_same_id_replaces_not_duplicates(uri: str, tmp_path) -> None:
    """Writing the same run twice (a retried apply) must not duplicate staging rows."""
    sink = open_sink(uri)
    before = sink.read_staging("db1")
    sink.write_staging(_without_meta(before).set_axis(list(before["shdi_source_row"].astype(int))), "db1")
    assert len(sink.read_staging("db1")) == len(before)
