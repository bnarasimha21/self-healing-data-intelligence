"""Score the pipeline against labelled, injected defects.

    python -m bench.score --rows 10000 --seed 7 --rate 0.015 --out bench/data --results bench/RESULTS.md

Generates the data for each scenario (bench/corrupt.py: a VARCHAR table and a
typed one), runs the real graph on the dirty DuckDB table with every proposal
auto-approved (the most a reviewer could accept), then compares the staging
and quarantine tables with the clean data and the labels. It also runs the
clean data on its own: a correct pipeline proposes no value change there.

Per defect type it reports:

* detection precision and recall. For a type the contract describes, the
  rows its expected check flagged (Scenario.expected_checks) against the rows
  that type was injected into. For an undescribed type there is no expected
  check, so recall counts a row as detected when any failing check or outlier
  finding names it, and precision is not reported;
* the outcome of every defect after apply, each counted on its own:
    restored     the staged value equals the clean value (a dropped duplicate counts here)
    quarantined  the row went to quarantine; a person still has to fix it
    neutralized  the bad value was replaced by null or the column default; the real value is lost
    wrong        the staged value is some other value, or the row vanished
    missed       the injected value went through unchanged
  restored rate = restored / defects. Quarantined and neutralized defects are
  not fixed and are not counted as fixed anywhere. contained = restored +
  quarantined + neutralized: the bad value is not in staging.

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
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from bench import corrupt
from bench.corrupt import Scenario

HERE = Path(__file__).resolve().parent
ENV_KEYS = ("SHDI_HOME", "SHDI_AUDIT_DB", "SHDI_LLM")
OUTCOMES = ["restored", "quarantined", "neutralized", "wrong", "missed"]


def _missing(v: Any) -> bool:
    return v is None or v is pd.NA or v is pd.NaT or (isinstance(v, float) and math.isnan(v))


def norm(type_: str, v: Any) -> Optional[str]:
    """One text form per value, so typed DuckDB output compares exactly with
    clean values: numbers through Decimal (no float rounding), dates as ISO,
    timestamps as the UTC instant."""
    if _missing(v):
        return None
    if type_ in ("integer", "number"):
        try:
            d = Decimal(repr(v)) if isinstance(v, float) else Decimal(str(v).strip())
        except InvalidOperation:
            return str(v)
        if not d.is_finite():
            return str(v)
        d = d.normalize()
        return format(d, "f")
    if type_ == "date":
        if isinstance(v, (dt.date, dt.datetime, pd.Timestamp)):
            return v.isoformat()[:10]
        return str(v)[:10]
    if type_ == "timestamp":
        try:
            ts = pd.Timestamp(v)
        except (TypeError, ValueError):
            return str(v)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        return ts.isoformat()
    return str(v)


def _run(uri: str, contract: Path, run_id: str, workdir: Path) -> Dict[str, Any]:
    from shdi import workflow
    from shdi.connectors import open_sink
    from shdi.graph import build_graph, make_checkpointer

    checkpoint = workdir / f"{run_id}.sqlite"
    for stale in workdir.glob(f"{run_id}.sqlite*"):
        stale.unlink()  # a rerun into the same directory starts fresh
    graph = build_graph(make_checkpointer(str(checkpoint)))
    t0 = time.perf_counter()
    st = workflow.start_run(uri, contract, auto_approve=True, reviewer="bench", run_id=run_id,
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


def _detection(sc: Scenario, checks: List[Dict[str, Any]], anomalies: List[Dict[str, Any]],
               labels: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    failing = {c["check_id"]: set(int(r) for r in c["failing_rows"]) for c in checks}
    anything = set().union(*failing.values(), *(set(int(x) for x in a["rows"]) for a in anomalies))
    out = {}
    for kind in sc.defect_types:
        check_ids = sc.expected_checks[kind]
        truth = set(int(r) for r in labels.loc[labels.defect_type == kind, "source_row"])
        pred = anything if check_ids is None else set().union(*(failing.get(c, set()) for c in check_ids))
        hit = len(truth & pred)
        out[kind] = {
            "injected": len(truth),
            "described": check_ids is not None,
            "flagged": len(pred) if check_ids is not None else None,
            "precision": None if check_ids is None else (hit / len(pred) if pred else (1.0 if not truth else 0.0)),
            "recall": hit / len(truth) if truth else 1.0,
        }
    return out


def _outcome(sc: Scenario, label: Dict[str, Any], staged: Dict[int, Dict[str, Any]], quarantined: set,
             clean: pd.DataFrame, defaults: Dict[str, Any]) -> str:
    row = int(label["source_row"])
    if row in quarantined:
        return "quarantined"
    if label["defect_type"] == "duplicate":
        return "restored" if row not in staged else "missed"
    if row not in staged:
        return "wrong"
    col = label["column"]
    t = sc.columns[col]
    got = norm(t, staged[row].get(col))
    if got == norm(t, clean.at[row, col]):
        return "restored"
    # A defect left exactly as injected was missed, even when the injected value
    # is a null: checking this first stops an untouched null counting as neutralized.
    injected = label["injected_value"]
    injected = None if _missing(injected) or injected == "" else injected
    if got == norm(t, injected):
        return "missed"
    if got is None or (defaults.get(col) is not None and got == norm(t, defaults[col])):
        return "neutralized"
    return "wrong"


def score(rows: int = 10000, seed: int = 7, rate: float = 0.015, out: Optional[Path] = None,
          scenario: str = "text") -> Dict[str, Any]:
    """Generate, run and score one scenario. Pure function of the arguments
    apart from timings."""
    from shdi.contract import load_contract

    sc = corrupt.SCENARIOS[scenario]
    cols = list(sc.columns)
    tmp = None
    if out is None:
        tmp = tempfile.TemporaryDirectory(prefix="shdi-bench-")
        out = Path(tmp.name)
    out = Path(out)
    saved = {k: os.environ.get(k) for k in ENV_KEYS}
    state = out / f".shdi-{sc.name}"
    shutil.rmtree(state, ignore_errors=True)  # a rerun starts from an empty audit log
    os.environ["SHDI_HOME"] = str(state)
    os.environ["SHDI_AUDIT_DB"] = str(state / "audit.sqlite")
    os.environ["SHDI_LLM"] = "off"
    try:
        clean, dirty, labels, db = corrupt.build(out, rows, seed, rate, sc)
        clean_db = out / f"clean_{sc.name}.duckdb"
        corrupt.write_duckdb(clean, clean_db, sc)

        res = _run(f"duckdb://{db.resolve()}#{sc.table}", sc.contract, f"bench-{sc.name}-{rows}-{seed}", out)
        base = _run(f"duckdb://{clean_db.resolve()}#{sc.table}", sc.contract, f"clean-{sc.name}-{rows}-{seed}", out)

        contract = load_contract(sc.contract)
        defaults = {name: c.default for name, c in contract.columns.items()}
        staged = _by_source_row(res["staging"])
        quarantined = set(_by_source_row(res["quarantine"]))

        per_type: Dict[str, Dict[str, Any]] = {kind: {o: 0 for o in OUTCOMES} for kind in sc.defect_types}
        for label in labels.to_dict("records"):
            per_type[label["defect_type"]][_outcome(sc, label, staged, quarantined, clean, defaults)] += 1
        detection = _detection(sc, res["status"]["checks_before"], res["anomalies"], labels)
        for kind, counts in per_type.items():
            n = sum(counts.values())
            counts["defects"] = n
            counts["restored_rate"] = counts["restored"] / n if n else 1.0
            counts["contained"] = counts["restored"] + counts["quarantined"] + counts["neutralized"]
            counts.update(detection[kind])

        def differs(r: int, c: str) -> bool:
            return norm(sc.columns[c], staged[r].get(c)) != norm(sc.columns[c], clean.at[r, c])

        # False changes on rows that carry no defect.
        defect_rows = set(int(r) for r in labels.source_row)
        dup_sources = set(int(s.split()[-1]) for s in labels.loc[labels.defect_type == "duplicate", "original_value"])
        clean_rows = [r for r in range(len(clean)) if r not in defect_rows]
        cells = len(clean_rows) * len(cols)
        changed = 0
        lost_rows = 0
        for r in clean_rows:
            if r not in staged:
                lost_rows += 1
                changed += len(cols)
                continue
            changed += sum(differs(r, c) for c in cols)
        # Changes to the other columns of a row that carries a defect. Fixing one
        # cell must not disturb its neighbours.
        collateral = 0
        for label in labels.to_dict("records"):
            r = int(label["source_row"])
            if label["defect_type"] == "duplicate" or r not in staged:
                continue
            collateral += sum(differs(r, c) for c in cols if c != label["column"])
        clean_outliers = sorted({int(x) for a in res["anomalies"] for x in a["rows"]} - defect_rows)
        all_defects = sum(v["defects"] for v in per_type.values())
        totals = {o: sum(v[o] for v in per_type.values()) for o in OUTCOMES}
        described = [k for k in sc.defect_types if sc.expected_checks[k] is not None]
        undescribed = [k for k in sc.defect_types if sc.expected_checks[k] is None]

        def group(kinds: List[str]) -> Dict[str, Any]:
            g = {o: sum(per_type[k][o] for k in kinds) for o in OUTCOMES}
            g["defects"] = sum(per_type[k]["defects"] for k in kinds)
            return g

        failing_after = [c["check_id"] for c in res["status"]["checks_after"] if not c["passed"]]
        base_failing = [c["check_id"] for c in base["status"]["checks_before"] if not c["passed"]]
        base_staged = _by_source_row(base["staging"])
        base_changed = sum(
            norm(sc.columns[c], base_staged[r].get(c)) != norm(sc.columns[c], clean.at[r, c])
            if r in base_staged else True
            for r in range(len(clean)) for c in cols
        )
        return {
            "scenario": sc.name,
            "table": sc.table,
            "sql_types": dict(sc.sql_types),
            "contract": sc.contract.name,
            "rows": rows,
            "seed": seed,
            "rate": rate,
            "dirty_rows": len(dirty),
            "defects": all_defects,
            "per_type": per_type,
            "totals": totals,
            "described": group(described),
            "undescribed": group(undescribed),
            "restored_rate": totals["restored"] / all_defects if all_defects else 1.0,
            "clean_cells": cells,
            "false_changes": changed,
            "false_change_rate": changed / cells if cells else 0.0,
            "clean_rows_lost": lost_rows,
            "collateral_changes": collateral,
            "duplicate_originals_kept": sum(1 for r in dup_sources if r in staged),
            "clean_outlier_findings": len(clean_outliers),
            "failing_checks_after": failing_after,
            "rows_staged": len(staged),
            "rows_quarantined": len(quarantined),
            "clean_only": {
                "proposals": len(base["status"]["proposals"]),
                "value_proposals": sum(1 for p in base["status"]["proposals"] if p["old_values"] != p["new_values"]),
                "failing_checks": base_failing,
                "outlier_findings": sum(len(a["rows"]) for a in base["anomalies"]),
                "cells_changed": (base["status"].get("apply") or {}).get("cells_changed"),
                "staged_cells_differing": base_changed,
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


def score_all(rows: int = 10000, seed: int = 7, rate: float = 0.015, out: Optional[Path] = None) -> Dict[str, Any]:
    return {name: score(rows, seed, rate, out, name) for name in corrupt.SCENARIOS}


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _render_one(r: Dict[str, Any]) -> List[str]:
    types = ", ".join(f"{c} {t}" for c, t in r["sql_types"].items())
    lines = [
        f"## Scenario `{r['scenario']}`: table `{r['table']}` ({types})",
        "",
        f"{r['rows']} clean rows, {r['defects']} injected defects ({r['dirty_rows']} rows in the dirty table),",
        f"contract `bench/{r['contract']}`. Run time {r['seconds']}s on the dirty table.",
        "",
        "| Defect | In contract | Injected | Detection precision | Detection recall | Restored | Quarantined | Neutralized | Wrong | Missed | Restored rate |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for kind, v in r["per_type"].items():
        lines.append(
            f"| {kind} | {'yes' if v['described'] else 'no'} | {v['defects']} | {_pct(v['precision'])} | "
            f"{_pct(v['recall'])} | {v['restored']} | {v['quarantined']} | {v['neutralized']} | {v['wrong']} | "
            f"{v['missed']} | {_pct(v['restored_rate'])} |"
        )
    for name, g in (("described by the contract", r["described"]), ("not in the contract", r["undescribed"])):
        lines.append(
            f"| **{name}** | | {g['defects']} | | | {g['restored']} | {g['quarantined']} | {g['neutralized']} | "
            f"{g['wrong']} | {g['missed']} | {_pct(g['restored'] / g['defects'] if g['defects'] else 1.0)} |"
        )
    t = r["totals"]
    lines += [
        f"| **all** | | {r['defects']} | | | {t['restored']} | {t['quarantined']} | {t['neutralized']} | {t['wrong']} | "
        f"{t['missed']} | {_pct(r['restored_rate'])} |",
        "",
        f"* Of {r['defects']} defects: {t['restored']} restored, {t['quarantined']} quarantined for a person, "
        f"{t['neutralized']} neutralized (value lost), {t['wrong'] + t['missed']} still wrong in staging.",
        f"* False-change rate on clean cells: {_pct(r['false_change_rate'])} "
        f"({r['false_changes']} of {r['clean_cells']}; clean rows lost: {r['clean_rows_lost']}).",
        f"* Collateral changes to other cells of defect rows: {r['collateral_changes']}.",
        f"* Outlier findings on clean rows of the dirty table: {r['clean_outlier_findings']}.",
        f"* Failing checks on the repaired output: {len(r['failing_checks_after'])}"
        + (f" ({', '.join(r['failing_checks_after'])})" if r["failing_checks_after"] else "") + ".",
        f"* Duplicates: the original row was kept for {r['duplicate_originals_kept']} of "
        f"{r['per_type']['duplicate']['defects']}.",
        f"* Clean data alone: {r['clean_only']['value_proposals']} proposals that change a value "
        f"(plus {r['clean_only']['proposals'] - r['clean_only']['value_proposals']} type casts that change no text), "
        f"{len(r['clean_only']['failing_checks'])} failing checks, {r['clean_only']['outlier_findings']} outlier findings, "
        f"{r['clean_only']['cells_changed']} cells changed, "
        f"{r['clean_only']['staged_cells_differing']} staged cells differing from the clean values.",
        "",
    ]
    return lines


def render(results: Dict[str, Dict[str, Any]]) -> str:
    first = next(iter(results.values()))
    lines = [
        "# Benchmark results",
        "",
        f"Generated by `make bench` (`python -m bench.score --rows {first['rows']} --seed {first['seed']} "
        f"--rate {first['rate']}`).",
        f"Every proposal auto-approved, as_of {corrupt.AS_OF.isoformat()}.",
        "",
        "Each outcome is counted on its own. Only restored means the clean value is back.",
        "Quarantined means a person still has to fix the row. Neutralized means the bad",
        "value became null or the column default, so the real value is lost. Neither is",
        "counted as fixed. The defect types marked \"no\" are not described by the contract",
        "(a x10 slip inside the allowed range, a wrong but plausible value, a date with day",
        "and month swapped); their detection recall counts any failing check or outlier",
        "finding on the row.",
        "",
    ]
    for r in results.values():
        lines += _render_one(r)
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Score self-healing on labelled injected defects")
    ap.add_argument("--rows", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--rate", type=float, default=0.015)
    ap.add_argument("--out", default=str(HERE / "data"))
    ap.add_argument("--results", default=str(HERE / "RESULTS.md"))
    a = ap.parse_args()
    text = render(score_all(a.rows, a.seed, a.rate, Path(a.out)))
    Path(a.results).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
