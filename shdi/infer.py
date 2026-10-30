"""Draft a contract from data, for a human to edit and commit.

This is the only place column types are guessed. The pipeline itself never
infers: it validates against the contract it is given.
"""
from __future__ import annotations

import re
from typing import Any, Dict

import pandas as pd
import yaml

from .contract import Contract
from .values import INVALID, parse

_ID_NAME = re.compile(r"(^id$)|(_id$)", re.I)
_DATE_NAME = re.compile(r"date|time|timestamp|_at$", re.I)


def _parse_rate(values, type_: str) -> float:
    present = [v for v in values if parse(v, "string").value is not None]
    if not present:
        return 0.0
    ok = sum(1 for v in present if parse(v, type_).value is not INVALID)
    return ok / len(present)


def infer_contract(df: pd.DataFrame, name: str, threshold: float = 0.9) -> Dict[str, Any]:
    columns: Dict[str, Any] = {}
    for c in df.columns:
        values = df[c].tolist()
        if _parse_rate(values, "integer") >= threshold:
            type_ = "integer"
        elif _parse_rate(values, "number") >= threshold:
            type_ = "number"
        elif _DATE_NAME.search(str(c)) and _parse_rate(values, "date") >= threshold:
            type_ = "date"
        elif _parse_rate(values, "boolean") >= 1.0:
            type_ = "boolean"
        else:
            type_ = "string"
        has_nulls = any(parse(v, "string").value is None for v in values)
        spec: Dict[str, Any] = {"type": type_, "required": True, "nullable": bool(has_nulls)}
        if type_ in ("integer", "number") and not _ID_NAME.search(str(c)):
            spec["measure"] = True
        columns[str(c)] = spec
    data: Dict[str, Any] = {
        "apiVersion": "shdi/v1",
        "name": name,
        "version": "0.1.0",
        "description": "Draft inferred from data. Review types, add min/max, accepted_values and primary_key before use.",
        "columns": columns,
    }
    pk = [c for c in columns if _ID_NAME.search(c) and columns[c]["type"] in ("integer", "string")]
    if len(pk) == 1:
        columns[pk[0]]["nullable"] = False
        data["primary_key"] = pk
    Contract.model_validate(data)
    return data


def infer_contract_yaml(df: pd.DataFrame, name: str) -> str:
    header = "# Draft data contract written by `shdi infer-contract`. Edit before use.\n"
    return header + yaml.safe_dump(infer_contract(df, name), sort_keys=False)
