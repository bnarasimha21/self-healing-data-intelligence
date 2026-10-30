"""Run-level operations shared by the CLI and the MCP server.

Every call goes through the compiled graph and its checkpointer, so a run
started by one process can be reviewed and resumed by another.
"""
from __future__ import annotations

import datetime as dt
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

from langgraph.types import Command

from .audit import AuditLog, now_iso
from .connectors import canonical_uri
from .contract import Contract, load_contract
from .graph import build_graph


class WorkflowError(ValueError):
    pass


def _config(run_id: str) -> Dict[str, Any]:
    return {"configurable": {"thread_id": run_id}}


def start_run(
    source_uri: str,
    contract: Union[str, Path, Contract, Dict[str, Any]],
    *,
    auto_approve: bool = False,
    reviewer: Optional[str] = None,
    run_id: Optional[str] = None,
    sink_dir: Optional[str] = None,
    as_of: Optional[dt.date] = None,
    graph: Any = None,
    user_request: str = "Validate against the contract and propose repairs.",
) -> Dict[str, Any]:
    if isinstance(contract, (str, Path)):
        contract = load_contract(contract)
    if isinstance(contract, dict):
        contract = Contract.model_validate(contract)
    graph = graph or build_graph()
    run_id = run_id or uuid.uuid4().hex[:12]
    if graph.get_state(_config(run_id)).values:
        raise WorkflowError(f"run {run_id} already exists")
    state = {
        "run_id": run_id,
        "source_uri": canonical_uri(source_uri),
        "sink_dir": sink_dir,
        "contract": contract.model_dump(mode="json"),
        "as_of": (as_of or dt.date.today()).isoformat(),
        "auto_approve": auto_approve,
        "reviewer": reviewer,
        "user_request": user_request,
        "messages": [],
    }
    graph.invoke(state, _config(run_id))
    return status(run_id, graph)


def status(run_id: str, graph: Any = None) -> Dict[str, Any]:
    graph = graph or build_graph()
    snap = graph.get_state(_config(run_id))
    values = snap.values or {}
    if not values:
        raise WorkflowError(f"unknown run {run_id}")
    out: Dict[str, Any] = {
        "run_id": run_id,
        "source": values.get("source_uri"),
        "proposals": values.get("proposals") or [],
    }
    interrupts = list(getattr(snap, "interrupts", ()) or ())
    if not interrupts:
        for task in snap.tasks or ():
            interrupts.extend(task.interrupts or ())
    if interrupts:
        payload = interrupts[0].value
        out.update(
            status="awaiting_approval",
            pending=payload["pending"],
            needs_approval=payload["needs_approval"],
            decisions=payload["decided"],
        )
    elif snap.next:
        out.update(status="interrupted", next=list(snap.next), decisions=values.get("decisions") or {})
    else:
        out.update(
            status="completed",
            outcome=values.get("status"),
            decisions=values.get("decisions") or {},
            apply=values.get("apply"),
            report=values.get("report"),
            summary=values.get("summary"),
            checks_before=values.get("checks_before"),
            checks_after=values.get("checks_after"),
        )
    return out


def decide(
    run_id: str,
    approved: Iterable[str] = (),
    rejected: Iterable[str] = (),
    reviewer: str = "unknown",
    *,
    approve_all: bool = False,
    reject_all: bool = False,
    graph: Any = None,
) -> Dict[str, Any]:
    """Record decisions and resume the run.

    Idempotent: ids that already have a decision are ignored, and calling it on
    a completed run changes nothing and writes no audit records.
    """
    approved, rejected = [str(i) for i in approved], [str(i) for i in rejected]
    if approve_all and reject_all:
        raise WorkflowError("choose approve_all or reject_all, not both")
    both = sorted(set(approved) & set(rejected))
    if both:
        raise WorkflowError(f"proposals both approved and rejected: {both}")
    graph = graph or build_graph()
    st = status(run_id, graph)
    known = {p["id"] for p in st["proposals"]}
    unknown = sorted((set(approved) | set(rejected)) - known)
    if unknown:
        raise WorkflowError(f"unknown proposal ids for run {run_id}: {unknown}")
    if st["status"] != "awaiting_approval":
        st["ignored"] = sorted(set(approved) | set(rejected))
        return st
    already = set(st["decisions"])
    new_approved = [i for i in approved if i not in already]
    new_rejected = [i for i in rejected if i not in already]
    if not (new_approved or new_rejected or approve_all or reject_all):
        st["ignored"] = sorted(set(approved) | set(rejected))
        return st
    graph.invoke(
        Command(
            resume={
                "approved": new_approved,
                "rejected": new_rejected,
                "approve_all": approve_all,
                "reject_all": reject_all,
                "reviewer": reviewer,
                "decided_at": now_iso(),
            }
        ),
        _config(run_id),
    )
    out = status(run_id, graph)
    out["ignored"] = sorted((set(approved) | set(rejected)) & already)
    return out


def audit_records(run_id: str) -> list:
    return AuditLog().events(run_id)
