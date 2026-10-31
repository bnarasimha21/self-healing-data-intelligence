"""The workflow as an MCP server (stdio).

    python -m shdi.mcp_server        or        shdi-mcp

Tools: profile_source, validate, propose_repairs, list_pending, decide, get_audit.

Safety model:

* Every source URI is checked against SHDI_ALLOWED_SOURCES, a comma-separated
  list of fnmatch patterns matched against the canonical URI (for example
  `duckdb:///data/warehouse.duckdb#*,postgres:sales.*`). Nothing is allowed
  when it is unset. Paths are resolved first, so `..` and symlinks cannot step
  outside a pattern. Run-scoped tools re-check the source of the run.
* Contracts are names resolved inside SHDI_CONTRACT_DIR (default ./contracts);
  paths are not accepted.
* Connection strings come from the environment (SHDI_PG_DSN, SHDI_PG_SINK_DSN),
  never from tool arguments; a URI carrying credentials is rejected.
* Only `decide` writes, and only to the staging and quarantine tables.
  `propose_repairs` always pauses before apply, even when nothing needs a
  decision, so a run is only written when a client calls `decide`.
"""
from __future__ import annotations

import datetime as dt
import fnmatch
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from . import workflow
from .anomaly import detect_anomalies
from .audit import AuditLog
from .checks import parse_frame, run_checks
from .connectors import SourceURIError, canonical_uri, open_source
from .contract import Contract, load_contract
from .infer import infer_contract

MAX_LIST = 50  # row lists in results are cut to this many entries
_CONTRACT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

server = MCPServer(
    name="self-healing-data-intelligence",
    instructions=(
        "Data contract checks with a human approval gate. Call propose_repairs to start a run; "
        "it pauses and writes nothing. Show the proposals to the user, then call decide with the "
        "ids they approve and reject. Only decide writes, and only to staging and quarantine tables."
    ),
)


def allowed_patterns() -> List[str]:
    raw = os.environ.get("SHDI_ALLOWED_SOURCES", "")
    return [p.strip() for p in raw.split(",") if p.strip()]


def check_source(uri: str) -> str:
    """Return the canonical URI, or raise ToolError if it is malformed or not allowed."""
    try:
        canon = canonical_uri(uri)
    except SourceURIError as e:
        raise ToolError(str(e)) from None
    patterns = allowed_patterns()
    if not patterns:
        raise ToolError("no sources are allowed: set SHDI_ALLOWED_SOURCES on the server")
    if not any(fnmatch.fnmatchcase(canon, p) for p in patterns):
        raise ToolError(f"source {canon} is not on the allowlist (SHDI_ALLOWED_SOURCES)")
    return canon


def contract_dir() -> Path:
    return Path(os.environ.get("SHDI_CONTRACT_DIR", "contracts")).expanduser().resolve()


def resolve_contract(name: str) -> Contract:
    if not _CONTRACT_NAME.match(name or "") or ".." in name:
        raise ToolError(f"contract must be a file name inside the contract directory, got {name!r}")
    base = contract_dir()
    candidates = [name] if name.endswith((".yaml", ".yml")) else [f"{name}.yaml", f"{name}.yml"]
    for c in candidates:
        path = (base / c).resolve()
        if path.parent == base and path.is_file():
            try:
                return load_contract(path)
            except Exception as e:  # pydantic or yaml errors, reported to the client
                raise ToolError(f"contract {name!r} is invalid: {e}") from None
    raise ToolError(f"contract {name!r} not found in {base}")


def _as_of(value: Optional[str]) -> dt.date:
    if not value:
        return dt.date.today()
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        raise ToolError(f"as_of must be an ISO date (YYYY-MM-DD), got {value!r}") from None


def _cut(items: List[Any]) -> List[Any]:
    return list(items[:MAX_LIST])


def _brief_check(c: Dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in c.items() if k not in ("failing_rows", "failing_row_keys")}
    out["failing_count"] = len(c["failing_rows"])
    out["failing_rows"] = _cut(c["failing_rows"])
    out["failing_row_keys"] = _cut(c["failing_row_keys"])
    out["truncated"] = len(c["failing_rows"]) > MAX_LIST
    return out


def _brief_proposal(p: Dict[str, Any]) -> dict[str, Any]:
    out = dict(p)
    n = len(p["row_keys"])
    for k in ("rows", "row_keys", "old_values", "new_values"):
        out[k] = _cut(p[k])
    out["row_count"] = n
    out["truncated"] = n > MAX_LIST
    return out


def _run_source(run_id: str) -> dict[str, Any]:
    try:
        st = workflow.status(run_id)
    except workflow.WorkflowError as e:
        raise ToolError(str(e)) from None
    check_source(st["source"])
    return st


def _run_view(st: Dict[str, Any]) -> dict[str, Any]:
    out: Dict[str, Any] = {
        "run_id": st["run_id"],
        "thread_id": st["run_id"],
        "source": st["source"],
        "status": st["status"],
        "proposals": [_brief_proposal(p) for p in st["proposals"]],
    }
    if st["status"] == "awaiting_approval":
        out["pending"] = [p["id"] for p in st["pending"]]
        out["needs_approval"] = st["needs_approval"]
        out["decisions"] = st["decisions"]
    else:
        out["decisions"] = st.get("decisions") or {}
    if st["status"] == "completed":
        out["outcome"] = st.get("outcome")
        out["apply"] = st.get("apply")
        report = st.get("report") or {}
        ba = dict(report.get("before_after") or {})
        ba.pop("checks", None)
        out["before_after"] = ba
        out["failing_checks_after"] = [c["check_id"] for c in st.get("checks_after") or [] if not c["passed"]]
    if "ignored" in st:
        out["ignored"] = st["ignored"]
    return out


@server.tool()
def profile_source(uri: str) -> dict[str, Any]:
    """Profile a source: row count and, per column, nulls, distinct values, a few samples and the inferred type."""
    canon = check_source(uri)
    try:
        df = open_source(canon).read()
    except Exception as e:
        raise ToolError(f"could not read {canon}: {e}") from None
    inferred = infer_contract(df, "profile")["columns"] if len(df.columns) else {}
    columns = []
    for c in df.columns:
        s = df[c]
        non_null = s.dropna()
        non_null = non_null[non_null.astype(str).str.strip() != ""]
        samples = [str(v)[:80] for v in non_null.drop_duplicates().head(5).tolist()]
        columns.append(
            {
                "name": str(c),
                "inferred_type": inferred.get(str(c), {}).get("type"),
                "nulls": int(len(s) - len(non_null)),
                "distinct": int(non_null.nunique()),
                "samples": samples,
            }
        )
    return {"source": canon, "rows": int(len(df)), "columns": columns}


@server.tool()
def validate(uri: str, contract: str, as_of: Optional[str] = None) -> dict[str, Any]:
    """Run every contract check on a source without starting a run. Read-only."""
    canon = check_source(uri)
    c = resolve_contract(contract)
    day = _as_of(as_of)
    try:
        df = open_source(canon).read()
    except Exception as e:
        raise ToolError(f"could not read {canon}: {e}") from None
    parsed = parse_frame(df, c)
    checks = [r.to_dict() for r in run_checks(df, c, day, parsed)]
    anomalies = detect_anomalies(df, c, parsed)
    return {
        "source": canon,
        "contract": c.name,
        "as_of": day.isoformat(),
        "rows": int(len(df)),
        "passed": all(r["passed"] for r in checks),
        "failing": [r["check_id"] for r in checks if not r["passed"]],
        "checks": [_brief_check(r) for r in checks],
        "outliers": [{**a, "rows": _cut(a["rows"])} for a in anomalies],
    }


@server.tool()
def propose_repairs(uri: str, contract: str, as_of: Optional[str] = None) -> dict[str, Any]:
    """Start a run: validate, propose repairs and pause for decisions. Writes nothing.

    Returns the thread_id (also the run_id) and the proposals. Pass the ids to decide."""
    canon = check_source(uri)
    c = resolve_contract(contract)
    try:
        st = workflow.start_run(canon, c, as_of=_as_of(as_of), always_pause=True,
                                user_request=f"Propose repairs for {canon} against contract {c.name}.")
    except Exception as e:  # connector and driver errors go back to the client as tool errors
        raise ToolError(f"{type(e).__name__}: {e}") from None
    return _run_view(st)


@server.tool()
def list_pending(thread_id: str) -> dict[str, Any]:
    """Show a run: its proposals, which are still undecided and which need a person's approval."""
    return _run_view(_run_source(thread_id))


@server.tool()
def decide(
    thread_id: str,
    approved_ids: Optional[List[str]] = None,
    rejected_ids: Optional[List[str]] = None,
    reviewer: str = "mcp-client",
) -> dict[str, Any]:
    """Approve and reject proposals by id, then resume the run.

    When every needs_approval proposal has a decision the run applies: approved
    changes go to the staging table, quarantined rows to the quarantine table.
    The source is never written. Ids that already have a decision are ignored,
    so repeating a call changes nothing. Low-risk proposals left undecided are
    approved by policy when the run applies."""
    _run_source(thread_id)
    try:
        st = workflow.decide(thread_id, approved_ids or [], rejected_ids or [], reviewer=reviewer)
    except workflow.WorkflowError as e:
        raise ToolError(str(e)) from None
    return _run_view(st)


@server.tool()
def get_audit(run_id: str) -> dict[str, Any]:
    """The audit trail of a run: every decision and every applied change, with row keys and old and new values."""
    _run_source(run_id)
    events = []
    for e in AuditLog().events(run_id):
        e = {k: v for k, v in e.items() if k != "proposal"}  # the full proposal is in list_pending
        e["row_count"] = len(e["row_keys"])
        e["truncated"] = e["row_count"] > MAX_LIST
        for k in ("row_keys", "old_values", "new_values"):
            e[k] = _cut(e[k])
        events.append(e)
    return {"run_id": run_id, "count": len(events), "events": events}


def main() -> None:
    from .tracing import shutdown

    try:
        server.run("stdio")
    finally:
        shutdown()


if __name__ == "__main__":
    main()
