"""Contract validation with structured results.

Each result names the check (`check_id`, for example `max:sales`), the column,
the rows that fail it (`failing_rows` are source row ordinals,
`failing_row_keys` add the primary key values), the severity and the
on_violation policy from the contract.
"""
from __future__ import annotations

import datetime as dt
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from .contract import Contract, resolve_bound
from .values import INVALID, Parsed, is_canonical, parse, raw_text, to_jsonable, value_key


@dataclass
class CheckResult:
    check_id: str
    kind: str
    column: Optional[str]
    severity: str
    on_violation: str
    passed: bool
    failing_rows: List[int] = field(default_factory=list)
    failing_row_keys: List[Dict[str, Any]] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return to_jsonable(asdict(self))


ParsedFrame = Dict[str, List[Parsed]]


def parse_frame(df: pd.DataFrame, contract: Contract) -> ParsedFrame:
    """Parse every declared column that exists in the frame, once."""
    out: ParsedFrame = {}
    for name, col in contract.columns.items():
        if name in df.columns:
            out[name] = [parse(x, col.type) for x in df[name].tolist()]
    return out


def row_key(df: pd.DataFrame, pos: int, contract: Contract) -> Dict[str, Any]:
    """Identify a row by its source ordinal (the frame index) plus primary key values."""
    key: Dict[str, Any] = {"_row": int(df.index[pos])}
    for c in contract.primary_key:
        if c in df.columns:
            raw = df[c].iloc[pos]
            value = parse(raw, contract.columns[c].type).value
            key[c] = to_jsonable(raw_text(raw) if value is INVALID else value)
    return key


def raw_text_frame(df: pd.DataFrame) -> pd.DataFrame:
    """All cells as text (None for nulls), used for exact-duplicate detection."""
    return pd.DataFrame({c: [raw_text(x) for x in df[c].tolist()] for c in df.columns}, index=df.index)


def run_checks(
    df: pd.DataFrame,
    contract: Contract,
    as_of: dt.date,
    parsed: Optional[ParsedFrame] = None,
) -> List[CheckResult]:
    if parsed is None:
        parsed = parse_frame(df, contract)
    results: List[CheckResult] = []

    def add(kind: str, check_id: str, column: Optional[str], rows: List[int], message: str) -> None:
        pol = contract.policy(kind, column)
        positions = sorted(set(int(r) for r in rows))
        rows = [int(df.index[r]) for r in positions]
        results.append(
            CheckResult(
                check_id=check_id,
                kind=kind,
                column=column,
                severity=pol["severity"],
                on_violation=pol["on_violation"],
                passed=not rows and not message.startswith("missing"),
                failing_rows=rows,
                failing_row_keys=[row_key(df, r, contract) for r in positions],
                message=message,
            )
        )

    n = len(df)
    for name, col in contract.columns.items():
        if name not in df.columns:
            if col.required:
                pol = contract.policy("required", name)
                results.append(
                    CheckResult(
                        check_id=f"required:{name}",
                        kind="required",
                        column=name,
                        severity=pol["severity"],
                        on_violation=pol["on_violation"],
                        passed=False,
                        message=f"missing required column '{name}'",
                    )
                )
            continue
        if col.required:
            add("required", f"required:{name}", name, [], "")
        values = parsed[name]
        raw = df[name].tolist()

        invalid = [i for i, p in enumerate(values) if p.value is INVALID]
        add("type", f"type:{name}", name, invalid, f"{len(invalid)} values are not {col.type}" if invalid else "")

        if col.type != "string":
            fmt = [
                i
                for i, p in enumerate(values)
                if p.value is not None
                and p.value is not INVALID
                and (p.ambiguous or not is_canonical(raw[i], p.value, col.type))
            ]
            add("format", f"format:{name}", name, fmt, f"{len(fmt)} values are not in canonical {col.type} form" if fmt else "")

        nulls = [i for i, p in enumerate(values) if p.value is None]
        if not col.nullable:
            add("not_null", f"not_null:{name}", name, nulls, f"{len(nulls)} nulls in a non-nullable column" if nulls else "")

        threshold = col.max_null_ratio if col.max_null_ratio is not None else contract.max_null_ratio
        if threshold is not None:
            ratio = (len(nulls) / n) if n else 0.0
            failing = nulls if ratio > threshold else []
            add(
                "max_null_ratio",
                f"max_null_ratio:{name}",
                name,
                failing,
                f"null ratio {ratio:.0%} above {threshold:.0%}" if failing else "",
            )

        if col.unique:
            groups: Dict[Any, List[int]] = defaultdict(list)
            for i, p in enumerate(values):
                if p.value is not None and p.value is not INVALID:
                    groups[value_key(p.value, col.type)].append(i)
            dup = [i for rows in groups.values() if len(rows) > 1 for i in rows]
            add("unique", f"unique:{name}", name, dup, f"{len(dup)} rows share a value" if dup else "")

        for kind in ("min", "max"):
            bound = resolve_bound(getattr(col, kind), col.type, as_of)
            if bound is None:
                continue
            bad = []
            bkey = value_key(bound, col.type)
            for i, p in enumerate(values):
                v = p.value
                if v is None or v is INVALID:
                    continue
                v = value_key(v, col.type)
                if (kind == "min" and v < bkey) or (kind == "max" and v > bkey):
                    bad.append(i)
            add(kind, f"{kind}:{name}", name, bad, f"{len(bad)} values {'below' if kind == 'min' else 'above'} {bound}" if bad else "")

        if col.accepted_values is not None:
            accepted = {value_key(parse(a, col.type).value, col.type) for a in col.accepted_values}
            bad = [
                i for i, p in enumerate(values)
                if p.value is not None and p.value is not INVALID and value_key(p.value, col.type) not in accepted
            ]
            add("accepted_values", f"accepted_values:{name}", name, bad, f"{len(bad)} values outside accepted_values" if bad else "")

        if col.pattern is not None:
            rx = re.compile(col.pattern)
            bad = [i for i, x in enumerate(raw) if raw_text(x) is not None and not rx.fullmatch(raw_text(x))]
            add("pattern", f"pattern:{name}", name, bad, f"{len(bad)} values do not match {col.pattern}" if bad else "")

    if contract.primary_key and all(c in df.columns for c in contract.primary_key):
        groups = defaultdict(list)
        for i in range(n):
            key = tuple(value_key(parsed[c][i].value, _type_of(contract, c)) for c in contract.primary_key)
            if any(k is None or k is INVALID for k in key):
                continue
            groups[key].append(i)
        dup = [i for rows in groups.values() if len(rows) > 1 for i in rows]
        cid = "primary_key:" + "+".join(contract.primary_key)
        add("primary_key", cid, None, dup, f"{len(dup)} rows share a primary key" if dup else "")

    if not contract.allow_duplicate_rows and n:
        dups = raw_text_frame(df).duplicated(keep="first")
        rows = [i for i, d in enumerate(dups.tolist()) if d]
        add("duplicate_rows", "duplicate_rows", None, rows, f"{len(rows)} exact duplicate rows" if rows else "")

    return results


def failing(results: List[Dict[str, Any]], severity: Optional[str] = None) -> List[Dict[str, Any]]:
    return [r for r in results if not r["passed"] and (severity is None or r["severity"] == severity)]


def _type_of(contract: Contract, column: str) -> Optional[str]:
    col = contract.columns.get(column)
    return col.type if col is not None else None
