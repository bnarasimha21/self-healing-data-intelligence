"""The LangGraph workflow: plan, load, validate, propose, approve, apply, summarize.

Outlier scans (shdi.anomaly) run inside `validate` on the input and inside
`apply` on the output, so the report can compare them.

Only `apply` produces changed data, and it writes it to staging and quarantine
tables, never the source. `approve` pauses the run with
`langgraph.types.interrupt()` until every `needs_approval` proposal has a
decision. The graph is compiled with a SQLite checkpointer, so a paused run can
be resumed by a different process, a day later, with
`Command(resume={"approved": [...], "rejected": [...], "reviewer": "..."})`.

The state holds the source URI and the source fingerprint, never the data.
Each node that needs rows re-reads the source; `propose` and `apply` refuse to
continue if the fingerprint changed since `load`.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import operator
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from langgraph.checkpoint.sqlite import SqliteSaver
from opentelemetry import trace as otel_trace
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from typing_extensions import Annotated, TypedDict

from .anomaly import detect_anomalies
from .audit import AuditLog, now_iso
from .checks import failing, parse_frame, run_checks
from .config import checkpoint_db_path
from .connectors import open_sink, open_source, to_csv_bytes
from .contract import contract_from_dict
from .llm import ChatModelFactory, explain_proposals, llm_mode, write_plan
from .repairs import apply_proposals, propose_repairs
from .tracing import traced_node

STATIC_PLAN = [
    "Load the source read-only and fingerprint it",
    "Validate every row against the data contract",
    "Propose repairs (nothing is changed yet)",
    "Pause for a reviewer to approve or reject each proposal",
    "Apply approved proposals to staging and quarantine tables, never the source",
    "Re-validate the output and scan measures for outliers",
    "Summarize before and after",
]


class PipelineState(TypedDict, total=False):
    # Inputs
    run_id: str
    source_uri: str
    sink_dir: Optional[str]
    contract: Dict[str, Any]
    as_of: str
    auto_approve: bool
    always_pause: bool
    reviewer: Optional[str]
    user_request: str
    # Working artifacts (JSON only, so they serialize into the checkpoint)
    source_fingerprint: str
    row_count: int
    columns: List[str]
    checks_before: List[Dict[str, Any]]
    anomalies_before: List[Dict[str, Any]]
    proposals: List[Dict[str, Any]]
    decisions: Dict[str, Dict[str, Any]]
    apply: Dict[str, Any]
    checks_after: List[Dict[str, Any]]
    anomalies_after: List[Dict[str, Any]]
    status: str
    report: Dict[str, Any]
    summary: str
    messages: Annotated[List[str], operator.add]


class StaleSourceError(RuntimeError):
    pass


def _snapshot(uri: str):
    """Rows plus the source's fingerprint. DuckDB and Postgres hash the
    database's own text and column types, which pandas values can hide."""
    return open_source(uri).snapshot()


def _read_checked(state: PipelineState):
    df, fp = _snapshot(state["source_uri"])
    if fp != state["source_fingerprint"]:
        raise StaleSourceError(
            f"source {state['source_uri']} changed since it was loaded "
            f"({state['source_fingerprint'][:12]} -> {fp[:12]}); start a new run"
        )
    return df


def _as_of(state: PipelineState) -> dt.date:
    return dt.date.fromisoformat(state["as_of"])


def node_plan(state: PipelineState, chat_model_factory: Optional[ChatModelFactory] = None) -> Dict[str, Any]:
    plan, note = STATIC_PLAN, None
    if llm_mode() == "plan":
        lines, note = write_plan(state.get("user_request", ""), chat_model_factory)
        plan = lines or STATIC_PLAN
    msgs = ([note] if note else []) + ["Plan:\n- " + "\n- ".join(plan)]
    return {"messages": msgs}


def node_load(state: PipelineState) -> Dict[str, Any]:
    df, fp = _snapshot(state["source_uri"])
    return {
        "source_fingerprint": fp,
        "row_count": int(len(df)),
        "columns": [str(c) for c in df.columns],
        "messages": [f"Loaded {state['source_uri']}: {len(df)} rows, {len(df.columns)} columns, fingerprint {fp[:12]}"],
    }


def node_validate(state: PipelineState) -> Dict[str, Any]:
    contract = contract_from_dict(state["contract"])
    df = _read_checked(state)
    parsed = parse_frame(df, contract)
    checks = [c.to_dict() for c in run_checks(df, contract, _as_of(state), parsed)]
    anomalies = detect_anomalies(df, contract, parsed)
    bad = failing(checks)
    msg = f"Validation: {len(bad)} of {len(checks)} checks failing"
    if bad:
        msg += "\n- " + "\n- ".join(f"{c['check_id']} ({c['severity']}): rows {c['failing_rows']}" for c in bad)
    return {"checks_before": checks, "anomalies_before": anomalies, "messages": [msg]}


def node_propose(state: PipelineState, chat_model_factory: Optional[ChatModelFactory] = None) -> Dict[str, Any]:
    from .checks import CheckResult

    contract = contract_from_dict(state["contract"])
    df = _read_checked(state)
    checks = [CheckResult(**c) for c in state["checks_before"]]
    proposals = [p.to_dict() for p in propose_repairs(df, contract, checks, _as_of(state))]
    need = sum(1 for p in proposals if p["risk"] == "needs_approval")
    messages = [f"Proposed {len(proposals)} changes, {need} need approval"]
    if llm_mode() == "explain" and need:
        # Text for the reviewer only: apply never reads `explanation`.
        explanations, note = explain_proposals(proposals, chat_model_factory)
        for p in proposals:
            if p["id"] in explanations:
                p["explanation"] = explanations[p["id"]]
        messages.append(f"LLM explained {len(explanations)} of {need} proposals")
        if note:
            messages.append(note)
    return {"proposals": proposals, "messages": messages}


def _resume_ids(value: Any, key: str) -> List[str]:
    ids = value.get(key) or []
    if isinstance(ids, str):
        ids = [ids]
    return [str(i) for i in ids]


def node_approve(state: PipelineState) -> Dict[str, Any]:
    run_id = state["run_id"]
    proposals = state.get("proposals") or []
    by_id = {p["id"]: p for p in proposals}
    audit = AuditLog()
    decisions: Dict[str, Dict[str, Any]] = dict(state.get("decisions") or {})

    def decide(pid: str, decision: str, reviewer: str, ts: str) -> None:
        if pid in decisions or pid not in by_id:
            return
        decisions[pid] = {"decision": decision, "reviewer": reviewer, "decided_at": ts}
        audit.record(run_id, by_id[pid], "decision", state["source_fingerprint"], decision, reviewer, ts)

    if state.get("auto_approve"):
        ts = now_iso()
        for p in proposals:
            decide(p["id"], "approved", state.get("reviewer") or "auto-approve", ts)
    elif proposals or state.get("always_pause"):
        # always_pause (set by the MCP server) pauses even with nothing to
        # decide, so the only call that writes is the one that resumes.
        first = True
        while first or any(p["risk"] == "needs_approval" and p["id"] not in decisions for p in proposals):
            first = False
            pending = [p for p in proposals if p["id"] not in decisions]
            value = interrupt(
                {
                    "run_id": run_id,
                    "pending": pending,
                    "needs_approval": [p["id"] for p in pending if p["risk"] == "needs_approval"],
                    "decided": decisions,
                }
            )
            # On resume this node runs again from the top and every earlier
            # interrupt() returns its stored answer, so decisions are replayed;
            # the audit log ignores records it already holds.
            value = value or {}
            reviewer = str(value.get("reviewer") or "unknown")
            ts = str(value.get("decided_at") or now_iso())
            undecided = [p["id"] for p in proposals if p["id"] not in decisions]
            for pid in _resume_ids(value, "rejected"):
                decide(pid, "rejected", reviewer, ts)
            for pid in _resume_ids(value, "approved"):
                decide(pid, "approved", reviewer, ts)
            if value.get("reject_all"):
                for pid in undecided:
                    decide(pid, "rejected", reviewer, ts)
            if value.get("approve_all"):
                for pid in undecided:
                    decide(pid, "approved", reviewer, ts)
        ts = now_iso()
        for p in proposals:
            if p["risk"] == "low":
                decide(p["id"], "approved", "policy:low-risk", ts)

    approved = sum(1 for d in decisions.values() if d["decision"] == "approved")
    return {
        "decisions": decisions,
        "messages": [f"Decisions: {approved} approved, {len(decisions) - approved} rejected"],
    }


def node_apply(state: PipelineState) -> Dict[str, Any]:
    run_id = state["run_id"]
    contract = contract_from_dict(state["contract"])
    proposals = state.get("proposals") or []
    by_id = {p["id"]: p for p in proposals}
    decisions = state.get("decisions") or {}
    approved_ids = [pid for pid, d in decisions.items() if d["decision"] == "approved"]
    audit = AuditLog()
    sink = open_sink(state["source_uri"], state.get("sink_dir"))
    try:
        df = _read_checked(state)
    except StaleSourceError as e:
        # An earlier attempt of this run may have written staging rows before
        # failing; nothing of a run whose source changed may stay behind.
        sink.delete_run(run_id)
        for pid in approved_ids:
            audit.record(run_id, by_id[pid], "skipped", state["source_fingerprint"], detail=str(e))
        return {"apply": {"status": "stale_source", "error": str(e)}, "status": "stale_source", "messages": [f"Apply refused: {e}"]}

    result = apply_proposals(df, contract, proposals, approved_ids)
    # Both tables in one transaction (one lock for CSV), replacing whatever an
    # earlier attempt of this run left in either of them.
    staged, quarantined = sink.write_run(result.output, result.quarantine, run_id)

    for pid in result.applied:
        audit.record(run_id, by_id[pid], "applied", state["source_fingerprint"])
    for s in result.skipped:
        audit.record(run_id, by_id[s["id"]], "skipped", state["source_fingerprint"], detail=s["reason"])

    parsed = parse_frame(result.output, contract)
    checks_after = [c.to_dict() for c in run_checks(result.output, contract, _as_of(state), parsed)]
    anomalies_after = detect_anomalies(result.output, contract, parsed)
    errors_after = failing(checks_after, "error")
    status = "passed" if not errors_after else "failed_checks"
    otel_trace.get_current_span().add_event(
        "shdi.apply.written",
        {
            "shdi.applied": len(result.applied),
            "shdi.skipped": len(result.skipped),
            "shdi.rows_staged": staged,
            "shdi.rows_quarantined": quarantined,
            "shdi.rows_dropped": result.rows_dropped,
            "shdi.cells_changed": result.cells_changed,
        },
    )
    apply_info = {
        "status": "applied",
        "applied": result.applied,
        "skipped": result.skipped,
        "rows_staged": staged,
        "rows_quarantined": quarantined,
        "rows_dropped": result.rows_dropped,
        "cells_changed": result.cells_changed,
        "output_sha256": hashlib.sha256(to_csv_bytes(result.output)).hexdigest(),
    }
    return {
        "apply": apply_info,
        "checks_after": checks_after,
        "anomalies_after": anomalies_after,
        "status": status,
        "messages": [
            f"Applied {len(result.applied)} proposals ({len(result.skipped)} skipped): "
            f"{staged} rows to staging, {quarantined} to quarantine, {result.rows_dropped} duplicates dropped"
        ],
    }


def node_summarize(state: PipelineState) -> Dict[str, Any]:
    before = state.get("checks_before") or []
    after = state.get("checks_after") or []
    after_by_id = {c["check_id"]: c for c in after}
    per_check = []
    for c in before:
        a = after_by_id.get(c["check_id"])
        per_check.append(
            {
                "check_id": c["check_id"],
                "severity": c["severity"],
                "failing_before": len(c["failing_rows"]) if not c["passed"] else 0,
                "passed_before": c["passed"],
                "failing_after": (len(a["failing_rows"]) if a and not a["passed"] else 0) if a else None,
                "passed_after": a["passed"] if a else None,
            }
        )
    proposals = state.get("proposals") or []
    decisions = state.get("decisions") or {}
    apply_info = state.get("apply") or {}
    report = {
        "run_id": state["run_id"],
        "source": state["source_uri"],
        "status": state.get("status", "unknown"),
        "rows_in": state.get("row_count"),
        "proposals": len(proposals),
        "needs_approval": sum(1 for p in proposals if p["risk"] == "needs_approval"),
        "approved": sum(1 for d in decisions.values() if d["decision"] == "approved"),
        "rejected": sum(1 for d in decisions.values() if d["decision"] == "rejected"),
        "before_after": {
            "checks": per_check,
            "failing_checks_before": len(failing(before)),
            "failing_checks_after": len(failing(after)) if after else None,
            "error_checks_after": len(failing(after, "error")) if after else None,
            "cells_changed": apply_info.get("cells_changed", 0),
            "rows_quarantined": apply_info.get("rows_quarantined", 0),
            "rows_dropped": apply_info.get("rows_dropped", 0),
            "rows_staged": apply_info.get("rows_staged", 0),
            "outlier_findings_before": sum(len(a["rows"]) for a in state.get("anomalies_before") or []),
            "outlier_findings_after": sum(len(a["rows"]) for a in state.get("anomalies_after") or []),
        },
    }
    lines = list(state.get("messages") or [])
    lines.append("")
    lines.append(f"Run {report['run_id']}: {report['status']}")
    lines.append(
        f"Proposals: {report['proposals']} ({report['needs_approval']} needed approval), "
        f"{report['approved']} approved, {report['rejected']} rejected"
    )
    ba = report["before_after"]
    lines.append(
        f"Before/after: failing checks {ba['failing_checks_before']} -> {ba['failing_checks_after']}, "
        f"cells changed {ba['cells_changed']}, rows quarantined {ba['rows_quarantined']}, "
        f"duplicates dropped {ba['rows_dropped']}"
    )
    for c in per_check:
        if not c["passed_before"] or c["passed_after"] is False:
            lines.append(f"  {c['check_id']:<28} {c['failing_before']:>5} -> {c['failing_after']}")
    for a in state.get("anomalies_after") or []:
        lines.append(f"Outliers in '{a['column']}' (review only): rows {a['rows']}")
    return {"report": report, "summary": "\n".join(lines) + "\n"}


def make_checkpointer(path: Optional[str] = None) -> SqliteSaver:
    p = Path(path or checkpoint_db_path())
    p.parent.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(sqlite3.connect(str(p), check_same_thread=False))


NODE_ORDER = ["plan", "load", "validate", "propose", "approve", "apply", "summarize"]


def build_graph(checkpointer: Any = None, chat_model_factory: Optional[ChatModelFactory] = None):
    """Compile the workflow.

    Uses a SQLite checkpointer from SHDI_CHECKPOINT_DB by default.
    `chat_model_factory` replaces the default Gradient model for SHDI_LLM=plan
    or explain (tests pass a stub).
    """
    graph = StateGraph(PipelineState)
    nodes = {
        "plan": lambda state: node_plan(state, chat_model_factory),
        "load": node_load,
        "validate": node_validate,
        "propose": lambda state: node_propose(state, chat_model_factory),
        "approve": node_approve,
        "apply": node_apply,
        "summarize": node_summarize,
    }
    for name in NODE_ORDER:
        graph.add_node(name, traced_node(name, nodes[name]))
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "load")
    graph.add_edge("load", "validate")
    graph.add_edge("validate", "propose")
    graph.add_edge("propose", "approve")
    graph.add_edge("approve", "apply")
    graph.add_edge("apply", "summarize")
    graph.add_edge("summarize", END)
    return graph.compile(checkpointer=checkpointer if checkpointer is not None else make_checkpointer())
