"""Score the pipeline against labelled, injected defects.

    python -m bench.score --rows 10000 --seed 7 --rate 0.015 --out bench/data --results bench/RESULTS.md

Generates the data (bench/corrupt.py), runs the real graph on the dirty DuckDB
table with every proposal auto-approved (the most a reviewer could accept),
then compares the staging and quarantine tables with the clean data and the
labels. It also runs the clean data on its own: a correct pipeline proposes
nothing there.

Per defect type it reports:

* detection precision and recall: the rows the expected check flagged
  (bench.corrupt.EXPECTED_CHECKS) against the rows that type was injected into;
* the outcome of every defect after apply:
    restored     the staged value equals the clean value (a dropped duplicate counts here)
    quarantined  the row went to quarantine for a person to fix
    neutralized  the bad value was replaced by null or the column default
    wrong        the staged value is some other value, or the row vanished
    missed       the injected value went through unchanged
  fix accuracy = (restored + quarantined) / defects.

Across the clean cells of the dirty table it reports the false-change rate:
cells changed, plus every cell of a clean row that was quarantined or dropped.
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from bench import corrupt

HERE = Path(__file__).resolve().parent
CONTRACT = HERE / "contract.yaml"
ENV_KEYS = ("SHDI_HOME", "SHDI_AUDIT_DB", "SHDI_LLM")
OUTCOMES = ["restored", "quarantined", "neutralized", "wrong", "missed"]


def norm(column: str, v: Any) -> Optional[str]:
    """One text form per value, so typed DuckDB output compares with clean text."""
    if v is None or (isinstance(v, float) and math.isnan(v)) or v is pd.NA or v is pd.NaT:
        return None
    if column in ("id", "sales"):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return str(v)
        return str(int(f)) if f.is_integer() else repr(f)
    if column == "date":
        if isinstance(v, (dt.date, dt.datetime, pd.Timestamp)):
            return v.isoformat()[:10]
        return str(v)[:10]
    return str(v)


def _run(uri: str, run_id: str, workdir: Path) -> Dict[str, Any]:
    from shdi import workflow
    from shdi.connectors import open_sink
    from shdi.graph import build_graph, make_checkpointer

    checkpoint = workdir / f"{run_id}.sqlite"
    for stale in workdir.glob(f"{run_id}.sqlite*"):
        stale.unlink()  # a rerun into the same directory starts fresh
    graph = build_graph(make_checkpointer(str(checkpoint)))
    t0 = time.perf_counter()
    st = workflow.start_run(uri, CONTRACT, auto_approve=True, reviewer="bench", run_id=run_id,
                            as_of=corrupt.AS_OF, graph=graph)
    elapsed = time.perf_counter() - t0
    values = graph.get_state({"configurable": {"thread_id": run_id}}).values
    sink = open_sink(uri)
    return {
        "status": st,
        "seconds": elapsed,
        "anomalies": values.get("anomalies_before") or [],
        "staging": sink.read_staging(run_id),
        "quarantine": sink.read_quarantine(run_id),
    }


def _by_source_row(df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    return {int(r["shdi_source_row"]): r for r in df.to_dict("records")}


def _detection(checks: List[Dict[str, Any]], labels: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    failing = {c["check_id"]: set(int(r) for r in c["failing_rows"]) for c in checks}
    out = {}
    for kind, check_ids in corrupt.EXPECTED_CHECKS.items():
        truth = set(int(r) for r in labels.loc[labels.defect_type == kind, "source_row"])
        pred = set().union(*(failing.get(c, set()) for c in check_ids))
        hit = len(truth & pred)
        out[kind] = {
            "injected": len(truth),
            "flagged": len(pred),
            "precision": hit / len(pred) if pred else (1.0 if not truth else 0.0),
            "recall": hit / len(truth) if truth else 1.0,
        }
    return out


def _outcome(label: Dict[str, Any], staged: Dict[int, Dict[str, Any]], quarantined: set, clean: pd.DataFrame,
             defaults: Dict[str, Any]) -> str:
    row = int(label["source_row"])
    if row in quarantined:
        return "quarantined"
    if label["defect_type"] == "duplicate":
        return "restored" if row not in staged else "missed"
    if row not in staged:
        return "wrong"
    col = label["column"]
    got = norm(col, staged[row].get(col))
    want = norm(col, clean.at[row, col])
    if got == want:
        return "restored"
    if got is None or (defaults.get(col) is not None and got == str(defaults[col])):
        return "neutralized"
    injected = label["injected_value"]
    injected = None if injected is None or (isinstance(injected, float) and math.isnan(injected)) or injected == "" else injected
    if got == norm(col, injected):
        return "missed"
    return "wrong"


def score(rows: int = 10000, seed: int = 7, rate: float = 0.015, out: Optional[Path] = None) -> Dict[str, Any]:
    """Generate, run and score. Pure function of (rows, seed, rate) apart from timings."""
    from shdi.contract import load_contract

    tmp = None
    if out is None:
        tmp = tempfile.TemporaryDirectory(prefix="shdi-bench-")
        out = Path(tmp.name)
    out = Path(out)
    saved = {k: os.environ.get(k) for k in ENV_KEYS}
    state = out / ".shdi"
    shutil.rmtree(state, ignore_errors=True)  # a rerun starts from an empty audit log
    os.environ["SHDI_HOME"] = str(state)
    os.environ["SHDI_AUDIT_DB"] = str(state / "audit.sqlite")
    os.environ["SHDI_LLM"] = "off"
    try:
        clean, dirty, labels, db = corrupt.build(out, rows, seed, rate)
        clean_db = out / "clean_sales.duckdb"
        corrupt.write_duckdb(clean, clean_db)

        res = _run(f"duckdb://{db.resolve()}#sales", f"bench-{rows}-{seed}", out)
        base = _run(f"duckdb://{clean_db.resolve()}#sales", f"clean-{rows}-{seed}", out)

        contract = load_contract(CONTRACT)
        defaults = {name: c.default for name, c in contract.columns.items()}
        staged = _by_source_row(res["staging"])
        quarantined = set(_by_source_row(res["quarantine"]))

        per_type: Dict[str, Dict[str, Any]] = {}
        for kind in corrupt.DEFECT_TYPES:
            per_type[kind] = {o: 0 for o in OUTCOMES}
        for label in labels.to_dict("records"):
            per_type[label["defect_type"]][_outcome(label, staged, quarantined, clean, defaults)] += 1
        detection = _detection(res["status"]["checks_before"], labels)
        for kind, counts in per_type.items():
            n = sum(counts.values())
            counts["defects"] = n
            counts["fix_accuracy"] = (counts["restored"] + counts["quarantined"]) / n if n else 1.0
            counts.update(detection[kind])

        # False changes on rows that carry no defect.
        defect_rows = set(int(r) for r in labels.source_row)
        dup_sources = set(int(s.split()[-1]) for s in labels.loc[labels.defect_type == "duplicate", "original_value"])
        clean_rows = [r for r in range(len(clean)) if r not in defect_rows]
        cells = len(clean_rows) * len(corrupt.COLUMNS)
        changed = 0
        lost_rows = 0
        for r in clean_rows:
            if r not in staged:
                lost_rows += 1
                changed += len(corrupt.COLUMNS)
                continue
            changed += sum(norm(c, staged[r].get(c)) != norm(c, clean.at[r, c]) for c in corrupt.COLUMNS)
        clean_outliers = sorted(
            {int(x) for a in res["anomalies"] for x in a["rows"]} - defect_rows
        )
        all_defects = sum(v["defects"] for v in per_type.values())
        totals = {o: sum(v[o] for v in per_type.values()) for o in OUTCOMES}
        failing_after = [c["check_id"] for c in res["status"]["checks_after"] if not c["passed"]]
        base_failing = [c["check_id"] for c in base["status"]["checks_before"] if not c["passed"]]
        return {
            "rows": rows,
            "seed": seed,
            "rate": rate,
            "dirty_rows": len(dirty),
            "defects": all_defects,
            "per_type": per_type,
            "totals": totals,
            "fix_accuracy": (totals["restored"] + totals["quarantined"]) / all_defects if all_defects else 1.0,
            "clean_cells": cells,
            "false_changes": changed,
            "false_change_rate": changed / cells if cells else 0.0,
            "clean_rows_lost": lost_rows,
            "duplicate_originals_kept": sum(1 for r in dup_sources if r in staged),
            "clean_outlier_findings": len(clean_outliers),
            "failing_checks_after": failing_after,
            "rows_staged": len(staged),
            "rows_quarantined": len(quarantined),
            "clean_only": {
                "proposals": len(base["status"]["proposals"]),
                "value_proposals": sum(1 for p in base["status"]["proposals"] if p["rows"]),
                "failing_checks": base_failing,
                "outlier_findings": sum(len(a["rows"]) for a in base["anomalies"]),
                "cells_changed": (base["status"].get("apply") or {}).get("cells_changed"),
            },
            "seconds": round(res["seconds"], 2),
        }
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        if tmp is not None:
            tmp.cleanup()


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def render(r: Dict[str, Any]) -> str:
    lines = [
        "# Benchmark results",
        "",
        f"Generated by `make bench` (`python -m bench.score --rows {r['rows']} --seed {r['seed']} --rate {r['rate']}`).",
        f"{r['rows']} clean rows, {r['defects']} injected defects ({r['dirty_rows']} rows in the dirty table),",
        "every proposal auto-approved, contract `bench/contract.yaml`, as_of "
        f"{corrupt.AS_OF.isoformat()}. Run time {r['seconds']}s on the dirty table.",
        "",
        "| Defect | Injected | Detection precision | Detection recall | Restored | Quarantined | Neutralized | Wrong | Missed | Fix accuracy |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for kind in corrupt.DEFECT_TYPES:
        v = r["per_type"][kind]
        lines.append(
            f"| {kind} | {v['defects']} | {_pct(v['precision'])} | {_pct(v['recall'])} | {v['restored']} | "
            f"{v['quarantined']} | {v['neutralized']} | {v['wrong']} | {v['missed']} | {_pct(v['fix_accuracy'])} |"
        )
    t = r["totals"]
    lines += [
        f"| **all** | {r['defects']} | | | {t['restored']} | {t['quarantined']} | {t['neutralized']} | {t['wrong']} | "
        f"{t['missed']} | {_pct(r['fix_accuracy'])} |",
        "",
        f"* False-change rate on clean cells: {_pct(r['false_change_rate'])} "
        f"({r['false_changes']} of {r['clean_cells']}; clean rows lost: {r['clean_rows_lost']}).",
        f"* Outlier findings on clean rows of the dirty table: {r['clean_outlier_findings']}.",
        f"* Failing checks on the repaired output: {len(r['failing_checks_after'])}"
        + (f" ({', '.join(r['failing_checks_after'])})" if r["failing_checks_after"] else "") + ".",
        f"* Duplicates: the original row was kept for {r['duplicate_originals_kept']} of "
        f"{r['per_type']['duplicate']['defects']}.",
        f"* Clean data alone: {r['clean_only']['value_proposals']} proposals that change a value "
        f"(plus {r['clean_only']['proposals'] - r['clean_only']['value_proposals']} type casts that change no text), "
        f"{len(r['clean_only']['failing_checks'])} failing checks, {r['clean_only']['outlier_findings']} outlier findings, "
        f"{r['clean_only']['cells_changed']} cells changed.",
        "",
        "Fix accuracy counts a defect as handled when its clean value is back in staging or",
        "the row is in quarantine for a person. Neutralized means the bad value became null",
        "or the column default: the bad value is gone but the real one is not recovered, so",
        "it does not count. Dates outside the contract range are nulled because the date",
        "column is nullable and the true date cannot be known.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Score self-healing on labelled injected defects")
    ap.add_argument("--rows", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--rate", type=float, default=0.015)
    ap.add_argument("--out", default=str(HERE / "data"))
    ap.add_argument("--results", default=str(HERE / "RESULTS.md"))
    a = ap.parse_args()
    r = score(a.rows, a.seed, a.rate, Path(a.out))
    text = render(r)
    Path(a.results).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
