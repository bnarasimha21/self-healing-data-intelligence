"""Strict, type-aware parsing of single cell values.

Every check and every repair works on parsed values instead of pandas dtypes,
so the same contract behaves the same whether the source column arrived as
text (CSV, a VARCHAR landing table) or already typed (an INTEGER column).
"""
from __future__ import annotations

import datetime as dt
import math
import re
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from dateutil import parser as dateparser

_NUMBER_RE = re.compile(r"^\s*[-+]?((\d+(\.\d*)?)|(\.\d+))([eE][-+]?\d+)?\s*$")
_INT_RE = re.compile(r"^\s*[-+]?\d+\s*$")
_YEAR_RE = re.compile(r"\d{4}")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_YMD_RE = re.compile(r"^\s*\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\s*$")
_BOOL_TRUE = {"true", "t", "yes", "y", "1"}
_BOOL_FALSE = {"false", "f", "no", "n", "0"}


class _Invalid:
    """Sentinel for a value that is present but does not parse as the type."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "INVALID"


INVALID = _Invalid()


@dataclass(frozen=True)
class Parsed:
    """Result of parsing one cell.

    value: the typed value, None for a null, INVALID when it does not parse.
    ambiguous: True when a text date could be read two ways (01/06/2024).
    """

    value: Any
    ambiguous: bool = False


def is_null(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, str):
        return False
    try:
        res = pd.isna(x)
    except (TypeError, ValueError):
        return False
    return bool(res) if isinstance(res, (bool, np.bool_)) else False


def parse_integer(x: Any) -> Parsed:
    if is_null(x):
        return Parsed(None)
    if isinstance(x, (bool, np.bool_)):
        return Parsed(INVALID)
    if isinstance(x, (int, np.integer)):
        return Parsed(int(x))
    if isinstance(x, (float, np.floating)):
        f = float(x)
        if math.isfinite(f) and f.is_integer():
            return Parsed(int(f))
        return Parsed(INVALID)
    s = str(x)
    if _INT_RE.match(s):
        return Parsed(int(s.strip()))
    if _NUMBER_RE.match(s):
        f = float(s)
        if math.isfinite(f) and f.is_integer():
            return Parsed(int(f))
    return Parsed(INVALID)


def parse_number(x: Any) -> Parsed:
    if is_null(x):
        return Parsed(None)
    if isinstance(x, (bool, np.bool_)):
        return Parsed(INVALID)
    if isinstance(x, (int, np.integer)):
        return Parsed(int(x))
    if isinstance(x, (float, np.floating)):
        f = float(x)
        if not math.isfinite(f):
            return Parsed(INVALID)
        return Parsed(int(f) if f.is_integer() else f)
    s = str(x)
    if _INT_RE.match(s):
        return Parsed(int(s.strip()))
    if _NUMBER_RE.match(s):
        f = float(s)
        if not math.isfinite(f):
            return Parsed(INVALID)
        return Parsed(f)
    return Parsed(INVALID)


def _to_date(x: Any) -> Optional[dt.date]:
    if isinstance(x, pd.Timestamp):
        return x.date()
    if isinstance(x, dt.datetime):
        return x.date()
    if isinstance(x, dt.date):
        return x
    if isinstance(x, np.datetime64):
        return pd.Timestamp(x).date()
    return None


def parse_date(x: Any) -> Parsed:
    if is_null(x):
        return Parsed(None)
    native = _to_date(x)
    if native is not None:
        return Parsed(native)
    s = str(x).strip()
    if not s or not _YEAR_RE.search(s):
        # dateutil happily reads "3" as the 3rd of this month. A date without a
        # four digit year is rejected instead of guessed.
        return Parsed(INVALID)
    try:
        if _YMD_RE.match(s):
            d = dateparser.parse(s, yearfirst=True, dayfirst=False)
            return Parsed(d.date())
        month_first = dateparser.parse(s, dayfirst=False)
        try:
            day_first = dateparser.parse(s, dayfirst=True)
        except (ValueError, OverflowError):
            day_first = month_first
    except (ValueError, OverflowError):
        return Parsed(INVALID)
    return Parsed(month_first.date(), ambiguous=month_first.date() != day_first.date())


def parse_timestamp(x: Any) -> Parsed:
    if is_null(x):
        return Parsed(None)
    if isinstance(x, (pd.Timestamp, dt.datetime)):
        return Parsed(pd.Timestamp(x).to_pydatetime())
    if isinstance(x, dt.date):
        return Parsed(dt.datetime(x.year, x.month, x.day))
    s = str(x).strip()
    if not s or not _YEAR_RE.search(s):
        return Parsed(INVALID)
    try:
        return Parsed(dateparser.parse(s))
    except (ValueError, OverflowError):
        return Parsed(INVALID)


def parse_boolean(x: Any) -> Parsed:
    if is_null(x):
        return Parsed(None)
    if isinstance(x, (bool, np.bool_)):
        return Parsed(bool(x))
    s = str(x).strip().lower()
    if s in _BOOL_TRUE:
        return Parsed(True)
    if s in _BOOL_FALSE:
        return Parsed(False)
    return Parsed(INVALID)


def parse_string(x: Any) -> Parsed:
    if is_null(x):
        return Parsed(None)
    return Parsed(str(x))


PARSERS = {
    "integer": parse_integer,
    "number": parse_number,
    "date": parse_date,
    "timestamp": parse_timestamp,
    "boolean": parse_boolean,
    "string": parse_string,
}


def parse(x: Any, type_: str) -> Parsed:
    return PARSERS[type_](x)


def canonical_text(value: Any, type_: str) -> Optional[str]:
    """Text form of a parsed value, as it is written to CSV output."""
    if value is None or value is INVALID:
        return None
    if type_ == "date":
        return value.isoformat()
    if type_ == "timestamp":
        return value.isoformat(sep=" ")
    if type_ == "boolean":
        return "true" if value else "false"
    if type_ in ("integer", "number"):
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)
    return str(value)


def raw_text(x: Any) -> Optional[str]:
    """How a raw cell reads as text, or None for a null."""
    if is_null(x):
        return None
    if isinstance(x, (float, np.floating)) and float(x).is_integer():
        return str(int(x))
    if isinstance(x, (pd.Timestamp, dt.datetime)):
        ts = pd.Timestamp(x)
        if ts == ts.normalize():
            return ts.date().isoformat()
        return ts.isoformat(sep=" ")
    if isinstance(x, dt.date):
        return x.isoformat()
    return str(x)


def to_jsonable(x: Any) -> Any:
    """Convert numpy/pandas/datetime values to plain JSON types."""
    if x is None or x is INVALID:
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        f = float(x)
        return f if math.isfinite(f) else None
    if isinstance(x, (pd.Timestamp, dt.datetime)):
        ts = pd.Timestamp(x)
        return ts.date().isoformat() if ts == ts.normalize() else ts.isoformat()
    if isinstance(x, dt.date):
        return x.isoformat()
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if is_null(x):
        return None
    return str(x) if not isinstance(x, str) else x
