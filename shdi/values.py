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

_NUMBER_RE = re.compile(r"^\s*[-+]?((\d+(\.\d*)?)|(\.\d+))([eE][-+]?\d+)?\s*$", re.ASCII)
_INT_RE = re.compile(r"^\s*[-+]?\d+\s*$", re.ASCII)
_YEAR_RE = re.compile(r"\d{4}", re.ASCII)
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_YMD_RE = re.compile(r"^\s*\d{4}[-/.]\d{1,2}[-/.]\d{1,2}(?:[T ]\S.*)?\s*$", re.ASCII)
# Integers are stored as Int64; anything wider is not a valid integer here.
INT64_MIN, INT64_MAX = -(2**63), 2**63 - 1
# Two far-apart defaults: a text that parses differently under each one was
# missing a part (year, month or day) that dateutil would fill in from them.
_DEFAULT_A = dt.datetime(2001, 1, 1)
_DEFAULT_B = dt.datetime(2002, 2, 2)
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


def _int_in_range(n: int) -> Parsed:
    return Parsed(n) if INT64_MIN <= n <= INT64_MAX else Parsed(INVALID)


def parse_integer(x: Any) -> Parsed:
    if is_null(x):
        return Parsed(None)
    if isinstance(x, (bool, np.bool_)):
        return Parsed(INVALID)
    if isinstance(x, (int, np.integer)):
        return _int_in_range(int(x))
    if isinstance(x, (float, np.floating)):
        f = float(x)
        if math.isfinite(f) and f.is_integer():
            return _int_in_range(int(f))
        return Parsed(INVALID)
    s = str(x)
    if _INT_RE.match(s):
        return _int_in_range(int(s.strip()))
    if _NUMBER_RE.match(s):
        f = float(s)
        if math.isfinite(f) and f.is_integer():
            return _int_in_range(int(f))
    return Parsed(INVALID)


def parse_number(x: Any) -> Parsed:
    if is_null(x):
        return Parsed(None)
    if isinstance(x, (bool, np.bool_)):
        return Parsed(INVALID)
    if isinstance(x, (int, np.integer)):
        n = int(x)
        return Parsed(n if INT64_MIN <= n <= INT64_MAX else float(n))
    if isinstance(x, (float, np.floating)):
        f = float(x)
        if not math.isfinite(f):
            return Parsed(INVALID)
        return Parsed(int(f) if f.is_integer() and INT64_MIN <= f <= INT64_MAX else f)
    s = str(x)
    if _INT_RE.match(s):
        n = int(s.strip())
        # Numbers are stored as floats once they do not fit in an Int64.
        return Parsed(n if INT64_MIN <= n <= INT64_MAX else float(n))
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
            d = _complete(s, yearfirst=True, dayfirst=False)
            return Parsed(d.date()) if d is not None else Parsed(INVALID)
        month_first = _complete(s, dayfirst=False)
        if month_first is None:
            return Parsed(INVALID)
        try:
            day_first = _complete(s, dayfirst=True) or month_first
        except (ValueError, OverflowError):
            day_first = month_first
    except (ValueError, OverflowError):
        return Parsed(INVALID)
    return Parsed(month_first.date(), ambiguous=month_first.date() != day_first.date())


def _complete(s: str, **kw) -> Optional[dt.datetime]:
    """Parse with dateutil, or None when the text lacks a year, month or day.

    dateutil fills missing parts from a default (today, unless told otherwise),
    so "2024" would become a different date every day. Parsing with two
    different defaults exposes that: a complete date gives the same answer."""
    a = dateparser.parse(s, default=_DEFAULT_A, **kw)
    b = dateparser.parse(s, default=_DEFAULT_B, **kw)
    if (a.year, a.month, a.day) != (b.year, b.month, b.day):
        return None
    return a


def _naive_utc(d: dt.datetime) -> dt.datetime:
    """Timestamps with an offset are converted to UTC and stored without one,
    so aware and naive values in one column compare and store together."""
    if d.tzinfo is None:
        return d
    return d.astimezone(dt.timezone.utc).replace(tzinfo=None)


def parse_timestamp(x: Any) -> Parsed:
    if is_null(x):
        return Parsed(None)
    if isinstance(x, (pd.Timestamp, dt.datetime)):
        return Parsed(_naive_utc(pd.Timestamp(x).to_pydatetime()))
    if isinstance(x, dt.date):
        return Parsed(dt.datetime(x.year, x.month, x.day))
    s = str(x).strip()
    if not s or not _YEAR_RE.search(s):
        return Parsed(INVALID)
    try:
        d = _complete(s)
    except (ValueError, OverflowError):
        return Parsed(INVALID)
    return Parsed(_naive_utc(d)) if d is not None else Parsed(INVALID)


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
