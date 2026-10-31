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
from decimal import Decimal
from fractions import Fraction
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
# Every integer up to this magnitude is exact in a float64.
FLOAT_EXACT_INT = 2**53
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


def is_nan(x: Any) -> bool:
    """A float NaN. Sources read NULL as None or pd.NA, so a NaN that reaches a
    check is a real NaN stored in the source (a DOUBLE 'NaN'), not a null."""
    return isinstance(x, (float, np.floating)) and math.isnan(x)


def is_null(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, (str, Decimal)):
        return False
    try:
        res = pd.isna(x)
    except (TypeError, ValueError):
        return False
    return bool(res) if isinstance(res, (bool, np.bool_)) else False


def _int_in_range(n: int) -> Parsed:
    return Parsed(n) if INT64_MIN <= n <= INT64_MAX else Parsed(INVALID)


def parse_integer(x: Any) -> Parsed:
    if is_nan(x):
        return Parsed(INVALID)
    if is_null(x):
        return Parsed(None)
    if isinstance(x, (bool, np.bool_)):
        return Parsed(INVALID)
    if isinstance(x, (int, np.integer)):
        return _int_in_range(int(x))
    if isinstance(x, Decimal):
        if x.is_finite() and x == x.to_integral_value():
            return _int_in_range(int(x))
        return Parsed(INVALID)
    if isinstance(x, (float, np.floating)):
        f = float(x)
        if math.isfinite(f) and f.is_integer():
            return _int_in_range(int(f))
        return Parsed(INVALID)
    s = str(x)
    if _INT_RE.match(s):
        return _int_in_range(int(s.strip()))
    if _NUMBER_RE.match(s):
        # Exact, not through a float: "1000000000000000000.5" is not an
        # integer, and "9007199254740993.0" is 9007199254740993.
        d = Decimal(s.strip())
        # adjusted() first: int() of 1e999999999 would build a huge integer.
        if d.is_finite() and (not d or d.adjusted() <= 19) and d == d.to_integral_value():
            return _int_in_range(int(d))
    return Parsed(INVALID)


def parse_number(x: Any) -> Parsed:
    """Numbers are exact: int inside the Int64 range, float when the text is
    exactly what the float prints as, Decimal otherwise (a value beyond Int64,
    a DECIMAL column, or more digits than a float holds)."""
    if is_nan(x):
        return Parsed(INVALID)
    if is_null(x):
        return Parsed(None)
    if isinstance(x, (bool, np.bool_)):
        return Parsed(INVALID)
    if isinstance(x, (int, np.integer)):
        n = int(x)
        return Parsed(n if INT64_MIN <= n <= INT64_MAX else Decimal(n))
    if isinstance(x, Decimal):
        return Parsed(x if x.is_finite() else INVALID)
    if isinstance(x, (float, np.floating)):
        f = float(x)
        if not math.isfinite(f):
            return Parsed(INVALID)
        return Parsed(int(f) if f.is_integer() and abs(f) <= FLOAT_EXACT_INT else f)
    s = str(x)
    if _INT_RE.match(s):
        n = int(s.strip())
        return Parsed(n if INT64_MIN <= n <= INT64_MAX else Decimal(n))
    if _NUMBER_RE.match(s):
        exact = Decimal(s.strip())
        # An exponent beyond +-1000 is not data; exact arithmetic on it
        # (Fraction, int) would also build numbers of that many digits.
        if not exact.is_finite() or (exact and abs(exact.adjusted()) > 1000):
            return Parsed(INVALID)
        if exact == exact.to_integral_value():
            # "1e5" is the integer 100000, as the float 100000.0 is. Past
            # Int64 it stays Decimal, never int() of 1e999999999.
            if not exact or (exact.adjusted() <= 18 and INT64_MIN <= int(exact) <= INT64_MAX):
                return Parsed(int(exact))
            return Parsed(exact)
        f = float(exact)
        # "0.1" reads back as the float 0.1; "123456789012345678.5" does not,
        # and "1e-400" underflows a float, so both stay Decimal.
        return Parsed(f if math.isfinite(f) and f != 0.0 and Decimal(repr(f)) == exact else exact)
    return Parsed(INVALID)


def number_key(v: Any) -> Optional[Fraction]:
    """Exact value of an int, float or Decimal, for comparing across types.

    A float stands for the decimal it prints as (291.92, not the nearest
    binary fraction), the same reading parse_number and exact_number use, so
    the float 291.92 and Decimal('291.92') are the same value."""
    if v is None or v is INVALID or isinstance(v, (bool, np.bool_)):
        return None
    if isinstance(v, (int, np.integer)):
        return Fraction(int(v))
    if isinstance(v, (float, np.floating)):
        return Fraction(repr(float(v))) if math.isfinite(v) else None
    if isinstance(v, Decimal):
        return Fraction(v) if v.is_finite() else None
    return None


def exact_number(q: Fraction) -> Optional[Any]:
    """The int, float or Decimal holding exactly `q`, or None when no finite
    decimal does (a third, say)."""
    if q.denominator == 1:
        n = q.numerator
        return n if INT64_MIN <= n <= INT64_MAX else Decimal(n)
    den = q.denominator
    while den % 2 == 0:
        den //= 2
    while den % 5 == 0:
        den //= 5
    if den != 1:
        return None
    # A terminating decimal: scale it to an integer and back, exactly.
    scale = 0
    while (q * 10**scale).denominator != 1:
        scale += 1
    d = Decimal((q * 10**scale).numerator).scaleb(-scale)
    f = float(d)
    # A float when it prints as exactly this decimal (12.345), like parse_number.
    return f if math.isfinite(f) and Decimal(repr(f)) == d else d


def _to_date(x: Any) -> Optional[dt.date]:
    # An instant with an offset is a date in UTC, never in the local timezone,
    # so the same row gives the same date on every machine.
    if isinstance(x, (pd.Timestamp, dt.datetime)):
        if x.tzinfo is not None:
            x = x.astimezone(dt.timezone.utc)
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
            return Parsed(_to_date(d)) if d is not None else Parsed(INVALID)
        month_first = _complete(s, dayfirst=False)
        if month_first is None:
            return Parsed(INVALID)
        try:
            day_first = _complete(s, dayfirst=True) or month_first
        except (ValueError, OverflowError):
            day_first = month_first
    except (ValueError, OverflowError):
        return Parsed(INVALID)
    # Text with an offset ("2024-01-01T20:00-05:00") is a moment: its date is
    # the UTC date, as for a native aware datetime.
    a, b = _to_date(month_first), _to_date(day_first)
    return Parsed(a, ambiguous=a != b)


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


def number_text(v: Any) -> str:
    """Text of an int, float or Decimal that reads back as the same value."""
    if isinstance(v, Decimal):
        return format(v, "f")
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if f.is_integer() and abs(f) <= FLOAT_EXACT_INT:
            return str(int(f))
        return repr(f)
    return str(v)


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
        return number_text(value)
    return str(value)


def is_canonical(raw: Any, value: Any, type_: str) -> bool:
    """True when a cell already holds `value` in its canonical form.

    A typed instant with an offset (a TIMESTAMPTZ) is canonical as it is: the
    contract reads it as the same moment in UTC, and nothing needs rewriting."""
    if type_ == "timestamp" and isinstance(raw, (pd.Timestamp, dt.datetime)) and raw.tzinfo is not None:
        return True
    return raw_text(raw) == canonical_text(value, type_)


def raw_text(x: Any) -> Optional[str]:
    """How a raw cell reads as text, or None for a null."""
    if is_nan(x):
        return "NaN"
    if is_null(x):
        return None
    if isinstance(x, (float, np.floating, Decimal)):
        return number_text(x)
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
    if isinstance(x, Decimal):
        # JSON numbers are floats to most readers, so a Decimal that a float
        # cannot hold exactly is kept as its text.
        if not x.is_finite():
            return None
        if x == x.to_integral_value() and abs(x) <= FLOAT_EXACT_INT:
            return int(x)
        if Decimal(repr(float(x))) == x:
            return float(x)
        return format(x, "f")
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
