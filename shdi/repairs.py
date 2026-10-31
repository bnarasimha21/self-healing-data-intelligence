"""Repair proposals: computed without touching data, applied only once approved.

`propose_repairs` reads the frame and the failing checks and returns a list of
`RepairProposal`. It never mutates its input. `apply_proposals` is the only
function that produces changed data, and it only applies the ids it is given.

Proposals are computed in a single pass over the contract's checks, so the
old repair/re-validate loop (MAX_REPAIR_PASSES) is gone: the checks are run
again on the output after apply, and anything still failing is reported.

Risk policy:
- `cast` that changes no cell text (only the storage type): `low`.
- everything else (fill, clamp, rescale, null_out, drop, quarantine, a cast
  that rewrites text such as 01/06/2024 -> 2024-01-06): `needs_approval`.
"""
from __future__ import annotations

import datetime as dt
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set

import pandas as pd

from .checks import CheckResult, ParsedFrame, parse_frame, raw_text_frame, row_key
from .contract import Contract, resolve_bound
from .values import INVALID, canonical_text, parse, raw_text, to_jsonable

ROW_RULES = ("quarantine_row", "drop_duplicate")
CELL_RULES = ("fill_default", "replace_invalid", "clamp", "rescale", "null_out")


@dataclass
class RepairProposal:
    id: str
    rule: str
    column: Optional[str]
    rows: List[int]
    row_keys: List[Dict[str, Any]]
    old_values: List[Any]
    new_values: List[Any]
    reason: str
    risk: str
    check_ids: List[str] = field(default_factory=list)
    explanation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(asdict(self))


def _row_dict(df: pd.DataFrame, pos: int) -> Dict[str, Any]:
    return {c: to_jsonable(raw_text(df[c].iloc[pos])) for c in df.columns}


def _storage_matches(series: pd.Series, type_: str) -> bool:
    if type_ == "integer":
        return pd.api.types.is_integer_dtype(series.dtype)
    if type_ == "number":
        return pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_bool_dtype(series.dtype)
    if type_ == "boolean":
        return pd.api.types.is_bool_dtype(series.dtype)
    if type_ == "timestamp":
        return pd.api.types.is_datetime64_any_dtype(series.dtype)
    if type_ == "date":
        if pd.api.types.is_datetime64_any_dtype(series.dtype):
            return True
        non_null = [x for x in series.tolist() if x is not None and not (isinstance(x, float) and x != x)]
        return bool(non_null) and all(isinstance(x, dt.date) for x in non_null)
    return True


def propose_repairs(
    df: pd.DataFrame,
    contract: Contract,
    checks: List[CheckResult],
    as_of: dt.date,
    parsed: Optional[ParsedFrame] = None,
) -> List[RepairProposal]:
    """Return proposals for every failing check whose policy is not report_only."""
    if not df.index.equals(pd.RangeIndex(len(df))):
        raise ValueError("propose_repairs needs the source frame as read (a 0..n-1 RangeIndex)")
    if parsed is None:
        parsed = parse_frame(df, contract)
    by_id = {c.check_id: c for c in checks}
    proposals: List[Dict[str, Any]] = []

    # Rows that leave the table. Collected first so no cell fix is proposed on a
    # row that is already proposed for removal.
    dup_rows: Dict[int, int] = {}
    dup_check = by_id.get("duplicate_rows")
    if dup_check is not None and not dup_check.passed and dup_check.on_violation != "report_only":
        text = raw_text_frame(df)
        first_seen: Dict[tuple, int] = {}
        for i, row in enumerate(text.itertuples(index=False, name=None)):
            if row in first_seen:
                dup_rows[i] = first_seen[row]
            else:
                first_seen[row] = i

    quarantine: Dict[int, List[tuple]] = defaultdict(list)

    def want_quarantine(row: int, check_id: str, why: str) -> None:
        if row not in dup_rows:
            quarantine[row].append((check_id, why))

    cell_fixes: List[Dict[str, Any]] = []
    for c in checks:
        if c.passed or c.on_violation == "report_only" or c.kind in ("format", "required", "max_null_ratio", "duplicate_rows"):
            continue
        name = c.column
        col = contract.columns.get(name) if name else None
        if c.kind == "primary_key":
            groups: Dict[tuple, List[int]] = defaultdict(list)
            for r in c.failing_rows:
                if r in dup_rows:
                    continue
                groups[tuple(parsed[k][r].value for k in contract.primary_key)].append(r)
            for key, rows in groups.items():
                if len(rows) > 1:
                    for r in rows:
                        want_quarantine(r, c.check_id, f"primary key {key} appears in {len(rows)} rows with different values")
            continue
        assert col is not None and name is not None
        for r in c.failing_rows:
            raw = df[name].iloc[r]
            value = parsed[name][r].value
            if c.on_violation == "quarantine":
                want_quarantine(r, c.check_id, f"{name}={raw_text(raw)!r}")
                continue
            # propose_fix: pick the fix that fits the check and the column.
            if c.kind == "type" or c.kind == "unique":
                if c.kind == "type" and col.nullable:
                    cell_fixes.append(dict(rule="null_out", column=name, row=r, old=raw, new=None, check=c.check_id,
                                           reason=f"'{raw_text(raw)}' is not a {col.type}; set to null (column is nullable)"))
                else:
                    want_quarantine(r, c.check_id, f"{name}={raw_text(raw)!r}")
            elif c.kind == "not_null":
                if col.default is not None:
                    cell_fixes.append(dict(rule="fill_default", column=name, row=r, old=raw, new=col.default, check=c.check_id,
                                           reason=f"null in non-nullable '{name}'; fill with contract default {col.default!r}"))
                else:
                    want_quarantine(r, c.check_id, f"null in non-nullable '{name}' and the contract gives no default")
            elif c.kind in ("accepted_values", "pattern"):
                if col.default is not None:
                    cell_fixes.append(dict(rule="replace_invalid", column=name, row=r, old=raw, new=col.default, check=c.check_id,
                                           reason=f"'{raw_text(raw)}' fails {c.kind}; replace with contract default {col.default!r}"))
                elif col.nullable:
                    cell_fixes.append(dict(rule="null_out", column=name, row=r, old=raw, new=None, check=c.check_id,
                                           reason=f"'{raw_text(raw)}' fails {c.kind}; set to null (column is nullable)"))
                else:
                    want_quarantine(r, c.check_id, f"{name}={raw_text(raw)!r} fails {c.kind}")
            elif c.kind in ("min", "max"):
                lo = resolve_bound(col.min, col.type, as_of)
                hi = resolve_bound(col.max, col.type, as_of)
                if col.type in ("integer", "number"):
                    rescaled = None
                    if c.kind == "max":
                        for f in col.unit_factors:
                            cand = value / f
                            if (lo is None or cand >= lo) and (hi is None or cand <= hi):
                                rescaled = (f, cand)
                                break
                    if rescaled is not None:
                        f, cand = rescaled
                        cand = int(cand) if float(cand).is_integer() else cand
                        cell_fixes.append(dict(rule="rescale", column=name, row=r, old=raw, new=cand, check=c.check_id,
                                               reason=f"{value} is above max {hi} but {value}/{f:g} = {cand} is in range; likely a unit error"))
                    else:
                        bound = lo if c.kind == "min" else hi
                        if col.type == "integer":
                            bound = int(bound)
                        cell_fixes.append(dict(rule="clamp", column=name, row=r, old=raw, new=bound, check=c.check_id,
                                               reason=f"{value} is {'below min' if c.kind == 'min' else 'above max'} {bound}; clamp to the bound"))
                elif col.nullable:
                    cell_fixes.append(dict(rule="null_out", column=name, row=r, old=raw, new=None, check=c.check_id,
                                           reason=f"{value} is {'before' if c.kind == 'min' else 'after'} {lo if c.kind == 'min' else hi}; set to null (column is nullable)"))
                else:
                    want_quarantine(r, c.check_id, f"{name}={raw_text(raw)!r} out of range and column is not nullable")

    # 1) storage casts, one per column, in contract order.
    for name, col in contract.columns.items():
        if name not in df.columns or col.type == "string":
            continue
        fmt = by_id.get(f"format:{name}")
        if fmt is not None and fmt.on_violation == "report_only":
            continue
        changed = list(fmt.failing_rows) if fmt is not None else []
        if not changed and _storage_matches(df[name], col.type):
            continue
        ambiguous = [r for r in changed if parsed[name][r].ambiguous]
        reason = f"store '{name}' as {col.type}"
        if changed:
            reason += f"; rewrites {len(changed)} values into canonical form"
        if ambiguous:
            reason += f"; {len(ambiguous)} are ambiguous and were read month-first"
        proposals.append(dict(
            rule="cast", column=name, rows=changed,
            row_keys=[row_key(df, r, contract) for r in changed],
            old_values=[to_jsonable(raw_text(df[name].iloc[r])) for r in changed],
            new_values=[canonical_text(parsed[name][r].value, col.type) for r in changed],
            reason=reason, risk="low" if not changed else "needs_approval",
            check_ids=[f"format:{name}"] if changed else [],
        ))

    # 2) required columns that are missing and have a default.
    for c in checks:
        if c.kind == "required" and not c.passed and c.on_violation == "propose_fix":
            col = contract.columns[c.column]
            if col.default is not None:
                proposals.append(dict(
                    rule="add_column", column=c.column, rows=[], row_keys=[], old_values=[], new_values=[col.default],
                    reason=f"required column '{c.column}' is missing; add it filled with default {col.default!r}",
                    risk="needs_approval", check_ids=[c.check_id],
                ))

    # 3) row removals and cell fixes, by row then column order.
    col_order = {n: i for i, n in enumerate(df.columns)}
    entries: List[tuple] = []
    for r, original in dup_rows.items():
        entries.append((r, -2, dict(
            rule="drop_duplicate", column=None, rows=[r], row_keys=[row_key(df, r, contract)],
            old_values=[_row_dict(df, r)], new_values=[None],
            reason=f"exact duplicate of row {original}", risk="needs_approval", check_ids=["duplicate_rows"],
        )))
    for r, reasons in quarantine.items():
        entries.append((r, -1, dict(
            rule="quarantine_row", column=None, rows=[r], row_keys=[row_key(df, r, contract)],
            old_values=[_row_dict(df, r)], new_values=[None],
            reason="quarantine: " + "; ".join(f"{cid} ({why})" for cid, why in reasons), risk="needs_approval",
            check_ids=sorted({cid for cid, _ in reasons}),
        )))
    seen_cells: Set[tuple] = set()
    for fx in cell_fixes:
        r = fx["row"]
        if r in dup_rows or r in quarantine or (r, fx["column"]) in seen_cells:
            continue
        seen_cells.add((r, fx["column"]))
        entries.append((r, col_order.get(fx["column"], 0), dict(
            rule=fx["rule"], column=fx["column"], rows=[r], row_keys=[row_key(df, r, contract)],
            old_values=[to_jsonable(raw_text(fx["old"]))], new_values=[to_jsonable(fx["new"])],
            reason=fx["reason"], risk="needs_approval", check_ids=[fx["check"]],
        )))
    entries.sort(key=lambda e: (e[0], e[1]))
    proposals.extend(e[2] for e in entries)

    return [RepairProposal(id=f"p{i + 1}", **p) for i, p in enumerate(proposals)]


@dataclass
class ApplyResult:
    output: pd.DataFrame
    quarantine: pd.DataFrame
    applied: List[str]
    skipped: List[Dict[str, str]]
    cells_changed: int
    rows_dropped: int
    rows_quarantined: int


def _typed_column(values: List[Any], type_: str) -> pd.Series:
    if type_ == "integer":
        return pd.Series(values, dtype="Int64")
    if type_ == "number":
        if all(v is None or isinstance(v, int) for v in values):
            return pd.Series(values, dtype="Int64")
        return pd.Series([None if v is None else float(v) for v in values], dtype="Float64")
    if type_ == "boolean":
        return pd.Series(values, dtype="boolean")
    if type_ == "timestamp":
        return pd.Series(pd.to_datetime(values), dtype="datetime64[us]")
    # date: python date objects keep date semantics in every sink.
    return pd.Series(values, dtype="object")


def apply_proposals(
    df: pd.DataFrame,
    contract: Contract,
    proposals: Iterable[Dict[str, Any]],
    approved_ids: Iterable[str],
) -> ApplyResult:
    """Apply only the approved proposals to a copy of `df`.

    `df` must be the same data the proposals were computed from (the caller
    checks the source fingerprint). Rows are identified by ordinal position.
    """
    approved: Set[str] = set(approved_ids)
    props = [p for p in proposals if p["id"] in approved]
    columns = {c: df[c].tolist() for c in df.columns}
    dtypes = {c: df[c].dtype for c in df.columns}
    n = len(df)
    removed: Dict[int, Dict[str, Any]] = {}
    applied: List[str] = []
    skipped: List[Dict[str, str]] = []
    changed_cells: Set[tuple] = set()  # (row, column); a cell fixed and cast counts once
    quarantine_rows: List[Dict[str, Any]] = []

    for p in props:
        if p["rule"] in ROW_RULES:
            r = p["rows"][0]
            if r in removed:
                skipped.append({"id": p["id"], "reason": f"row already removed by {removed[r]['id']}"})
                continue
            removed[r] = p
            applied.append(p["id"])
            if p["rule"] == "quarantine_row":
                row = {c: columns[c][r] for c in df.columns}
                row.update({"shdi_source_row": r, "shdi_proposal_id": p["id"], "shdi_reason": p["reason"]})
                quarantine_rows.append(row)

    for p in props:
        if p["rule"] == "add_column":
            columns[p["column"]] = [p["new_values"][0]] * n
            dtypes[p["column"]] = None
            applied.append(p["id"])
            changed_cells.update((r, p["column"]) for r in range(n) if r not in removed)

    def store(column: str, value: Any) -> Any:
        dtype = dtypes.get(column)
        if value is None:
            return None
        if dtype is None or not (pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_datetime64_any_dtype(dtype)):
            type_ = contract.columns[column].type if column in contract.columns else "string"
            return canonical_text(parse(value, type_).value, type_) or str(value)
        return value

    for p in props:
        if p["rule"] not in CELL_RULES:
            continue
        r, col = p["rows"][0], p["column"]
        if r in removed:
            skipped.append({"id": p["id"], "reason": f"row removed by {removed[r]['id']}"})
            continue
        current = raw_text(columns[col][r])
        if current != p["old_values"][0] and not (current is None and p["old_values"][0] is None):
            skipped.append({"id": p["id"], "reason": f"cell changed since proposal ({current!r})"})
            continue
        columns[col][r] = store(col, p["new_values"][0])
        applied.append(p["id"])
        changed_cells.add((r, col))

    keep = [r for r in range(n) if r not in removed]
    out = pd.DataFrame({c: [columns[c][r] for r in keep] for c in columns}, index=keep)
    for c in df.columns:
        if dtypes.get(c) is not None and c in out.columns:
            try:
                out[c] = out[c].astype(dtypes[c])
            except (TypeError, ValueError):
                pass

    for p in props:
        if p["rule"] != "cast":
            continue
        name = p["column"]
        type_ = contract.columns[name].type
        parsed = [parse(x, type_) for x in out[name].tolist()]
        bad = [int(out.index[i]) for i, x in enumerate(parsed) if x.value is INVALID]
        if bad:
            skipped.append({"id": p["id"], "reason": f"rows {bad} still hold values that are not {type_}"})
            continue
        out[name] = _typed_column([x.value for x in parsed], type_).set_axis(out.index)
        applied.append(p["id"])
        changed_cells.update((r, name) for r in p["rows"] if r not in removed)

    quarantine = pd.DataFrame(quarantine_rows)
    return ApplyResult(
        output=out,
        quarantine=quarantine,
        applied=applied,
        skipped=skipped,
        cells_changed=len(changed_cells),
        rows_dropped=sum(1 for p in removed.values() if p["rule"] == "drop_duplicate"),
        rows_quarantined=len(quarantine_rows),
    )
