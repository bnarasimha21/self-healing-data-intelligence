"""Issue 2: the full graph, driven by contracts/sales.yaml, on the dirty fixture."""
from __future__ import annotations

from shdi import workflow
from shdi.connectors import open_sink
from shdi.graph import build_graph, make_checkpointer

from conftest import AS_OF, CONTRACT

# check_id -> failing source rows (index labels). These are exactly the defects
# planted in the fixture, and nothing else.
EXPECTED_FAILING_BEFORE = {
    "format:date": [1, 2],  # 01/06/2024 and 2024/07/01 are not ISO
    "min:date": [4, 5],  # 1960-01-01, twice
    "max:date": [3],  # 2050-12-01
    "type:sales": [2],  # "three hundred"
    "min:sales": [11],  # -5
    "max:sales": [6],  # 400000
    "not_null:region": [10],  # empty region
    "primary_key:id": [4, 5],  # id 5 twice
    "duplicate_rows": [5],  # the second copy of row 4
}


def run(source, tmp_path, **kw):
    graph = build_graph(make_checkpointer(str(tmp_path / "cp.sqlite")))
    return workflow.start_run(str(source), CONTRACT, as_of=AS_OF, graph=graph, run_id="c1", **kw)


def test_exact_failing_checks_before_repair(dirty_csv, tmp_path):
    st = run(dirty_csv, tmp_path)
    assert st["status"] == "awaiting_approval"
    graph = build_graph(make_checkpointer(str(tmp_path / "cp.sqlite")))
    before = graph.get_state({"configurable": {"thread_id": "c1"}}).values["checks_before"]
    got = {c["check_id"]: c["failing_rows"] for c in before if not c["passed"]}
    assert got == EXPECTED_FAILING_BEFORE
    # Nothing about id is an outlier, and 400000 is flagged for review.
    anomalies = graph.get_state({"configurable": {"thread_id": "c1"}}).values["anomalies_before"]
    assert {a["column"] for a in anomalies} == {"sales"}
    assert 6 in anomalies[0]["rows"]


def test_approving_everything_leaves_zero_error_failures(dirty_csv, tmp_path):
    st = run(dirty_csv, tmp_path, auto_approve=True, reviewer="test-auto")
    assert st["status"] == "completed" and st["outcome"] == "passed"
    assert [c["check_id"] for c in st["checks_after"] if not c["passed"] and c["severity"] == "error"] == []
    ba = st["report"]["before_after"]
    assert ba["failing_checks_before"] == len(EXPECTED_FAILING_BEFORE)
    assert ba["error_checks_after"] == 0
    assert (ba["rows_quarantined"], ba["rows_dropped"], ba["rows_staged"]) == (1, 1, 10)

    staged = open_sink(f"csv:{dirty_csv}").read_staging("c1")
    assert "three hundred" not in set(staged["sales"])
    assert "400" in set(staged["sales"]) and "400000" not in set(staged["sales"])
    assert "0" in set(staged["sales"])  # -5 clamped to the contract minimum
    assert list(staged["date"][:2]) == ["2024-01-05", "2024-01-06"]
    q = open_sink(f"csv:{dirty_csv}").read_quarantine("c1")
    assert list(q["id"]) == ["3"]
    assert "type:sales" in q["shdi_reason"].iloc[0]


def test_approving_everything_through_the_review_path_matches_auto_approve(dirty_csv, tmp_path):
    st = run(dirty_csv, tmp_path)
    graph = build_graph(make_checkpointer(str(tmp_path / "cp.sqlite")))
    done = workflow.decide("c1", approve_all=True, reviewer="frank", graph=graph)
    assert done["outcome"] == "passed"
    assert done["report"]["approved"] == len(st["proposals"])
