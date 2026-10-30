"""Command line interface.

    shdi run <source> --contract contracts/sales.yaml      start a run, print pending proposals
    shdi review <run_id>                                   list proposals still waiting
    shdi approve <run_id> --all | --ids p1 p2              approve and resume
    shdi reject <run_id> --all | --ids p3                  reject and resume
    shdi status <run_id>                                   summary of a finished run
    shdi audit <run_id>                                    the audit trail
    shdi infer-contract <source> [-o draft.yaml]           draft a contract for a human to edit
"""
from __future__ import annotations

import argparse
import getpass
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import workflow
from .connectors import open_source
from .infer import infer_contract_yaml
from .tracing import shutdown as flush_traces


def _fmt_keys(keys: List[Dict[str, Any]]) -> str:
    return ", ".join(" ".join(f"{k}={v}" for k, v in key.items()) for key in keys) or "(column)"


def print_proposals(proposals: List[Dict[str, Any]], out=sys.stdout) -> None:
    for p in proposals:
        print(f"  {p['id']:<5} [{p['risk']}] {p['rule']} {p.get('column') or ''}  rows: {_fmt_keys(p['row_keys'])}", file=out)
        for old, new in list(zip(p["old_values"], p["new_values"]))[:5]:
            print(f"        {json.dumps(old)} -> {json.dumps(new)}", file=out)
        if len(p["old_values"]) > 5:
            print(f"        ... {len(p['old_values']) - 5} more", file=out)
        print(f"        reason: {p['reason']}", file=out)
        if p.get("explanation"):
            print(f"        explanation: {p['explanation']}", file=out)


def print_status(st: Dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(st, indent=2, default=str))
        return
    print(f"run {st['run_id']}: {st['status']}")
    if st["status"] == "awaiting_approval":
        pending = st["pending"]
        need = st["needs_approval"]
        print(f"{len(st['proposals'])} proposed changes, {len(need)} need approval, {len(pending)} undecided")
        print_proposals(pending)
        print(f"\nApprove: shdi approve {st['run_id']} --all | --ids <id> ...")
        print(f"Reject:  shdi reject {st['run_id']} --all | --ids <id> ...")
        print("Low-risk proposals not explicitly rejected are approved when the last needs_approval one is decided.")
    elif st["status"] == "completed":
        print(st.get("summary") or "")
    if st.get("ignored"):
        print(f"already decided, ignored: {', '.join(st['ignored'])}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="shdi", description="Self-healing data intelligence")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="start a run")
    r.add_argument("source", help="file.csv, duckdb:///path/file.duckdb#table or postgres:schema.table")
    r.add_argument("--contract", required=True, help="path to the data contract YAML")
    r.add_argument("--auto-approve", action="store_true", help="approve every proposal without pausing")
    r.add_argument("--reviewer", default=None)
    r.add_argument("--run-id", default=None)
    r.add_argument("--sink-dir", default=None, help="CSV sources only: folder for staging/quarantine files")
    r.add_argument("--json", action="store_true")

    for name in ("review", "status"):
        s = sub.add_parser(name)
        s.add_argument("run_id")
        s.add_argument("--json", action="store_true")

    for name in ("approve", "reject"):
        s = sub.add_parser(name)
        s.add_argument("run_id")
        g = s.add_mutually_exclusive_group(required=True)
        g.add_argument("--all", action="store_true")
        g.add_argument("--ids", nargs="+")
        s.add_argument("--reviewer", default=None)
        s.add_argument("--json", action="store_true")

    a = sub.add_parser("audit")
    a.add_argument("run_id")

    i = sub.add_parser("infer-contract")
    i.add_argument("source")
    i.add_argument("-o", "--output")
    i.add_argument("--name", default=None)

    args = ap.parse_args(argv)
    try:
        if args.cmd == "run":
            st = workflow.start_run(
                args.source,
                args.contract,
                auto_approve=args.auto_approve,
                reviewer=args.reviewer,
                run_id=args.run_id,
                sink_dir=args.sink_dir,
            )
            print_status(st, args.json)
        elif args.cmd in ("review", "status"):
            print_status(workflow.status(args.run_id), args.json)
        elif args.cmd in ("approve", "reject"):
            reviewer = args.reviewer or getpass.getuser()
            ids = args.ids or []
            st = workflow.decide(
                args.run_id,
                approved=ids if args.cmd == "approve" else [],
                rejected=ids if args.cmd == "reject" else [],
                reviewer=reviewer,
                approve_all=args.all and args.cmd == "approve",
                reject_all=args.all and args.cmd == "reject",
            )
            print_status(st, args.json)
        elif args.cmd == "audit":
            for e in workflow.audit_records(args.run_id):
                print(json.dumps({k: e[k] for k in ("seq", "ts", "proposal_id", "event", "decision", "reviewer", "rule",
                                                    "column_name", "row_keys", "old_values", "new_values", "detail")}))
        elif args.cmd == "infer-contract":
            src = open_source(args.source)
            text = infer_contract_yaml(src.read(), args.name or src.name)
            if args.output:
                Path(args.output).write_text(text)
                print(f"wrote {args.output}")
            else:
                print(text, end="")
    except (workflow.WorkflowError, ValueError, FileNotFoundError, RuntimeError, PermissionError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    finally:
        flush_traces()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
