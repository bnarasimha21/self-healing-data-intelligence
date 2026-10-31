"""Repair proposals: computed without touching data, applied only once approved.

`propose_repairs` reads the frame and the failing checks and returns a list of
`RepairProposal`. It never mutates its input. `apply_proposals` is the only
function that produces changed data, and it only applies the ids it is given.

Proposals are computed in a single pass over the contract's checks, so the
old repair/re-validate loop (MAX_REPAIR_PASSES) is gone: the checks are run
again on the output after apply, and anything still failing is reported.

Risk policy:
- `cast` where every value, once stored as the contract type, reads back as
  exactly its source text (only the storage type changes): `low`.
- everything else (fill, clamp, rescale, null_out, drop, quarantine, a cast
  that rewrites text such as 01/06/2024 -> 2024-01-06 or 1.5 -> 1.50):
  `needs_approval`.

Every proposal lists the rows it touches with their keys and old and new
values, casts and added columns included, so the audit names every cell.

`apply_proposals` checks every cell it stores: if the column's type cannot
hold a value exactly (a clamp to 99.5 in a BIGINT column), the column is
widened, and the write is refused if any cell would still differ from the
value in its proposal.
"""
from __future__ import annotations

import datetime as dt
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from fractions import Fraction
from typing import Any, Dict, Iterable, List, Optional, Set

import pandas as pd
import pyarrow as pa

from .checks import CheckResult, ParsedFrame, parse_frame, raw_text_frame, row_key
from .contract import Contract, resolve_bound
from .values import (
    FLOAT_EXACT_INT,
    INVALID,
    canonical_text,
    exact_number,
    is_canonical,
    is_null,
    number_key,
    is_nan,
    number_text,
    parse,
    raw_text,
    to_jsonable,
)

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
    d = series.dtype
    if type_ == "integer" and pd.api.types.is_integer_dtype(d):
        return True
    if type_ == "number" and pd.api.types.is_numeric_dtype(d) and not pd.api.types.is_bool_dtype(d):
        return True
    if type_ == "boolean" and pd.api.types.is_bool_dtype(d):
        return True
    if type_ in ("timestamp", "date") and pd.api.types.is_datetime64_any_dtype(d):
        return True
    # An object column of Python values (a Postgres read) is already typed
    # when every value is of the contract type.
    non_null = [x for x in series.tolist() if not is_null(x)]
    if not non_null:
        return False
    if type_ == "integer":
        return all(isinstance(x, int) and not isinstance(x, bool) for x in non_null)
    if type_ == "number":
        return all(isinstance(x, (int, float, Decimal)) and not isinstance(x, bool) for x in non_null)
    if type_ == "boolean":
        return all(isinstance(x, bool) for x in non_null)
    if type_ == "timestamp":
        return all(isinstance(x, dt.datetime) for x in non_null)
    if type_ == "date":
        return all(isinstance(x, dt.date) for x in non_null)
    return True


def _factor(f: Any) -> Fraction:
    # 0.001 in a contract means the decimal 0.001, not the nearest float.
    return Fraction(repr(f)) if isinstance(f, float) else Fraction(f)


def _final_text(value: Any, type_: str) -> Optional[str]:
    """Canonical text of a value as the typed column stores it."""
    return canonical_text(parse(value, type_).value, type_)


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
                            # Exact arithmetic: 12345 / 1000 is 12.345, which an
                            # integer column cannot hold, so that factor is skipped.
                            q = number_key(value) / _factor(f)
                            if (lo is not None and q < lo) or (hi is not None and q > hi):
                                continue
                            if col.type == "integer" and q.denominator != 1:
                                continue
                            cand = exact_number(q)
                            if cand is not None:
                                rescaled = (f, cand)
                                break
                    if rescaled is not None:
                        f, cand = rescaled
                        cell_fixes.append(dict(rule="rescale", column=name, row=r, old=raw, new=cand, check=c.check_id,
                                               reason=f"{number_text(value)} is above max {hi} but {number_text(value)}/{f:g} = "
                                                      f"{number_text(cand)} is in range; likely a unit error"))
                    else:
                        bound = lo if c.kind == "min" else hi
                        if col.type == "integer":
                            # The nearest whole number inside the range, not int() toward zero.
                            bound = math.ceil(bound) if c.kind == "min" else math.floor(bound)
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
        values = [p.value for p in parsed[name]]
        rows = [r for r, v in enumerate(values) if v is not None and v is not INVALID]
        # The value each cell will hold once stored as the contract type. Risk
        # comes from comparing its text with the source text, so a cast that
        # would change any value (precision, scale, a reformatted date) is
        # never low risk.
        typed = _typed_column([v if v is not INVALID else None for v in values], col.type).tolist()
        # A column holding a value exactly reports the row's own canonical
        # text (a DECIMAL(19,1) pads 5 to 5.0 for display, which is the same
        # value); one that would change it (float rounding) reports what it
        # actually stores, so the change shows up below.
        new_text = {
            r: canonical_text(values[r], col.type) if same_value(typed[r], values[r], col.type)
            else _final_text(typed[r], col.type)
            for r in rows
        }
        changed = [
            r for r in rows
            if parsed[name][r].ambiguous
            or (new_text[r] != raw_text(df[name].iloc[r]) and not is_canonical(df[name].iloc[r], values[r], col.type))
        ]
        if not changed and _storage_matches(df[name], col.type):
            continue
        ambiguous = [r for r in changed if parsed[name][r].ambiguous]
        reason = f"store '{name}' as {col.type}"
        if changed:
            reason += f"; rewrites {len(changed)} values"
        if ambiguous:
            reason += f"; {len(ambiguous)} are ambiguous and were read month-first"
        proposals.append(dict(
            rule="cast", column=name, rows=rows,
            row_keys=[row_key(df, r, contract) for r in rows],
            old_values=[to_jsonable(raw_text(df[name].iloc[r])) for r in rows],
            new_values=[new_text[r] for r in rows],
            reason=reason, risk="low" if not changed else "needs_approval",
            check_ids=[f"format:{name}"] if changed else [],
        ))

    # 2) required columns that are missing and have a default: every row gets it.
    for c in checks:
        if c.kind == "required" and not c.passed and c.on_violation == "propose_fix":
            col = contract.columns[c.column]
            if col.default is not None:
                n = len(df)
                proposals.append(dict(
                    rule="add_column", column=c.column, rows=list(range(n)),
                    row_keys=[row_key(df, r, contract) for r in range(n)],
                    old_values=[None] * n, new_values=[to_jsonable(col.default)] * n,
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


def _decimal_column(values: List[Any]) -> pd.Series:
    """Exact numbers that neither Int64 nor Float64 holds, as a DECIMAL column
    wide enough for every value; text when even DECIMAL(38) is too narrow."""
    decs = [None if v is None else Decimal(number_text(v)) for v in values]
    present = [d for d in decs if d is not None]
    scale = max([max(-d.as_tuple().exponent, 0) for d in present] or [0])
    whole = max([max(d.adjusted() + 1, 1) for d in present] or [1])
    if whole + scale <= 38:
        arr = pa.array(decs, type=pa.decimal128(whole + scale, scale))
        return pd.Series(arr, dtype=pd.ArrowDtype(arr.type))
    return pd.Series([None if d is None else format(d, "f") for d in decs], dtype="object")


def _typed_column(values: List[Any], type_: str) -> pd.Series:
    """A column of parsed values in the narrowest storage that holds each exactly."""
    if type_ == "integer":
        return pd.Series(values, dtype="Int64")
    if type_ == "number":
        present = [v for v in values if v is not None]
        if all(isinstance(v, int) for v in present):
            return pd.Series(values, dtype="Int64")
        # A float64 holds every int up to 2**53 and every float exactly; past
        # that, 123456789012345678 next to 1.5 would round to ...680.
        if all(isinstance(v, float) or (isinstance(v, int) and abs(v) <= FLOAT_EXACT_INT) for v in present):
            return pd.Series([None if v is None else float(v) for v in values], dtype="Float64")
        return _decimal_column(values)
    if type_ == "boolean":
        return pd.Series(values, dtype="boolean")
    if type_ == "timestamp":
        return pd.Series(pd.to_datetime(values), dtype="datetime64[us]")
    # date: python date objects keep date semantics in every sink.
    return pd.Series(values, dtype="object")


def same_value(a: Any, b: Any, type_: Optional[str]) -> bool:
    """True when two cells hold the same value under the contract type (1.5 and
    Decimal('1.50') in a number column), or the same text outside the contract."""
    ta, tb = raw_text(a), raw_text(b)
    if ta is None or tb is None:
        return ta is None and tb is None
    if type_ is not None:
        pa_, pb_ = parse(a, type_).value, parse(b, type_).value
        if pa_ is not INVALID and pb_ is not INVALID:
            if type_ in ("integer", "number"):
                return number_key(pa_) == number_key(pb_)
            return pa_ == pb_
    return ta == tb


def _is_text(series: pd.Series) -> bool:
    d = series.dtype
    if pd.api.types.is_numeric_dtype(d) or pd.api.types.is_datetime64_any_dtype(d) or pd.api.types.is_bool_dtype(d):
        return False
    return all(isinstance(v, str) for v in series.tolist() if not is_null(v))


def _restore(values: List[Any], dtype: Any, type_: Optional[str]) -> pd.Series:
    """Put a column back in its source dtype, but only if that keeps every value.

    `astype` truncates silently (99.5 into BIGINT gives 99), so each cell is
    compared with what was put in. When the dtype loses anything the column is
    widened: exact numbers for a number column, text for anything else."""
    if dtype is not None:
        src = values
        if isinstance(dtype, pd.ArrowDtype) and pa.types.is_decimal(dtype.pyarrow_dtype):
            # Arrow will not take a float into a decimal column; the float's
            # printed digits are its value here, so hand it over as a Decimal.
            src = [Decimal(number_text(v)) if isinstance(v, (int, float)) and not isinstance(v, bool)
                   and not is_nan(v) else v for v in values]
        try:
            if isinstance(dtype, pd.ArrowDtype):
                # Built by Arrow itself: pandas' astype would turn a NaN into a
                # null, and Arrow refuses (rather than truncates) 99.5 as int64.
                arr = pa.array([None if v is pd.NA or v is pd.NaT else v for v in src],
                               type=dtype.pyarrow_dtype, from_pandas=False)
                s = pd.Series(arr, dtype=dtype)
            else:
                s = pd.Series(src, dtype="object").astype(dtype)
        except Exception:  # noqa: BLE001 - any failure means "does not fit"
            s = None
        if s is not None and all(same_value(a, b, type_) for a, b in zip(values, s.tolist())):
            return s
    if type_ in ("integer", "number"):
        parsed = [parse(v, "number").value for v in values]
        if INVALID not in parsed:
            s = _typed_column(parsed, "number")
            if all(same_value(a, b, type_) for a, b in zip(values, s.tolist())):
                return s
    if dtype is not None and all(isinstance(v, str) for v in values if not is_null(v)):
        return pd.Series([None if is_null(v) else v for v in values], dtype="object")
    return pd.Series([raw_text(v) for v in values], dtype="object")


def apply_proposals(
    df: pd.DataFrame,
    contract: Contract,
    proposals: Iterable[Dict[str, Any]],
    approved_ids: Iterable[str],
) -> ApplyResult:
    """Apply only the approved proposals to a copy of `df`.

    `df` must be the same data the proposals were computed from (the caller
    checks the source fingerprint). Rows are identified by ordinal position.
    Raises RuntimeError rather than return an output in which an applied cell
    differs from the value its proposal (and so the audit log) records.
    """
    approved: Set[str] = set(approved_ids)
    props = [p for p in proposals if p["id"] in approved]
    columns = {c: df[c].tolist() for c in df.columns}
    dtypes = {c: df[c].dtype for c in df.columns}
    text_columns = {c for c in df.columns if _is_text(df[c])}
    n = len(df)
    removed: Dict[int, Dict[str, Any]] = {}
    applied: List[str] = []
    skipped: List[Dict[str, str]] = []
    quarantine_rows: List[Dict[str, Any]] = []

    def ctype(column: str) -> Optional[str]:
        return contract.columns[column].type if column in contract.columns else None

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

    added: Set[str] = set()
    for p in props:
        if p["rule"] == "add_column":
            columns[p["column"]] = [p["new_values"][0]] * n
            dtypes[p["column"]] = None
            text_columns.add(p["column"])
            added.add(p["column"])
            applied.append(p["id"])

    def store(column: str, value: Any) -> Any:
        if value is None:
            return None
        type_ = ctype(column)
        if column in text_columns:
            # A text source (CSV, VARCHAR) keeps text, in canonical form.
            return canonical_text(parse(value, type_).value, type_) if type_ else str(value)
        if type_ is None:
            return value
        v = parse(value, type_).value
        return value if v is INVALID else v

    applied_cells: List[tuple] = []
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
        applied_cells.append((r, col, p["new_values"][0], p["id"]))

    keep = [r for r in range(n) if r not in removed]
    out = pd.DataFrame(
        {c: _restore([columns[c][r] for r in keep], dtypes.get(c), ctype(c)).set_axis(keep) for c in columns},
        index=keep,
    )

    cast_cells: List[tuple] = []
    fixed = {(r, c) for r, c, _, _ in applied_cells}
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
        cast_cells.extend(
            (r, name, new, p["id"]) for r, new in zip(p["rows"], p["new_values"]) if r not in removed and (r, name) not in fixed
        )

    # The written value must be the audited value, for every applied cell.
    for r, col, new, pid in applied_cells + cast_cells:
        got = out.at[r, col]
        if not same_value(got, new, ctype(col)):
            raise RuntimeError(
                f"refusing to write: {pid} records {col}={new!r} for row {r} but the output holds {raw_text(got)!r}"
            )

    # A cell counts as changed when its text in the output differs from the source.
    cells_changed = len(keep) * len(added)
    for c in df.columns:
        before, after = df[c].tolist(), out[c].tolist()
        cells_changed += sum(1 for i, r in enumerate(keep) if raw_text(before[r]) != raw_text(after[i]))

    quarantine = pd.DataFrame(quarantine_rows)
    return ApplyResult(
        output=out,
        quarantine=quarantine,
        applied=applied,
        skipped=skipped,
        cells_changed=cells_changed,
        rows_dropped=sum(1 for p in removed.values() if p["rule"] == "drop_duplicate"),
        rows_quarantined=len(quarantine_rows),
    )
