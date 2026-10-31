"""Source and Sink protocols plus helpers shared by every connector.

A Source is read-only. A Sink writes only `<table>__shdi_staging` and
`<table>__shdi_quarantine`, never the source table. `write_run` replaces the
run's rows in both tables inside one transaction (one file lock for CSV), so
retrying a write is idempotent and a failure leaves neither table half
written; `delete_run` removes them, which a run whose source went stale uses.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np
import pandas as pd

from ..values import is_nan, is_null, raw_text

STAGING_SUFFIX = "__shdi_staging"
QUARANTINE_SUFFIX = "__shdi_quarantine"
META_COLUMNS = ("shdi_run_id", "shdi_applied_at", "shdi_source_row")


class Source(Protocol):
    uri: str
    name: str

    def read(self, limit: Optional[int] = None) -> pd.DataFrame: ...

    def snapshot(self) -> Tuple[pd.DataFrame, str]: ...

    def fingerprint(self) -> str: ...


class Sink(Protocol):
    def write_staging(self, df: pd.DataFrame, run_id: str) -> int: ...

    def write_quarantine(self, df: pd.DataFrame, run_id: str) -> int: ...

    def write_run(self, staging: pd.DataFrame, quarantine: pd.DataFrame, run_id: str) -> Tuple[int, int]: ...

    def delete_run(self, run_id: str) -> None: ...

    def read_staging(self, run_id: Optional[str] = None) -> pd.DataFrame: ...

    def read_quarantine(self, run_id: Optional[str] = None) -> pd.DataFrame: ...


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Serialize with minimal quoting, '\\n' line ends and empty cells for nulls.

    For a CSV that was read as text this reproduces the original bytes, which
    is how "nothing approved means nothing changed" is checked.
    """
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow([str(c) for c in df.columns])
    for row in df.itertuples(index=False, name=None):
        w.writerow(["" if raw_text(v) is None else raw_text(v) for v in row])
    return buf.getvalue().encode()


def frame_fingerprint(df: pd.DataFrame) -> str:
    """Content hash of a frame: column names plus every cell as text."""
    h = hashlib.sha256()
    h.update(("\x1f".join(map(str, df.columns)) + "\x1e").encode())
    for row in df.itertuples(index=False, name=None):
        h.update(("\x1f".join("\x00" if raw_text(v) is None else raw_text(v) for v in row) + "\x1e").encode())
    return h.hexdigest()


def with_meta(df: pd.DataFrame, run_id: str, applied_at: Optional[str] = None) -> pd.DataFrame:
    """Add run metadata; the frame index must hold source row ordinals.

    A NaN in a plain numpy float column is pandas' missing marker and is
    written as null. The connectors read through Arrow, where a NaN is a
    stored value distinct from null, and those columns are left as they are."""
    out = df.copy()
    for c in out.columns:
        if isinstance(out[c].dtype, np.dtype) and out[c].dtype.kind == "f" and out[c].isna().any():
            col = out[c]
            out[c] = pd.Series([None if is_nan(v) else float(v) for v in col], index=col.index, dtype=object)
    out["shdi_run_id"] = run_id
    out["shdi_applied_at"] = applied_at or dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    if "shdi_source_row" not in out.columns:
        out["shdi_source_row"] = [int(i) for i in df.index]
    return out.reset_index(drop=True)


def quarantine_as_text(df: pd.DataFrame) -> pd.DataFrame:
    """Quarantined rows keep their raw values, stored as text."""
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        if c == "shdi_source_row":
            out[c] = [int(v) for v in df[c]]
        else:
            out[c] = pd.Series([raw_text(v) for v in df[c]], dtype="object", index=df.index)
    return out


def py_value(v: Any) -> Any:
    """A value a DB driver accepts: None for nulls, plain Python scalars. A NaN
    stays NaN: it is a stored value, not a null."""
    if is_nan(v):
        return float("nan")
    if is_null(v):
        return None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()
    return v


def check_target(table: str) -> None:
    if not (table.endswith(STAGING_SUFFIX) or table.endswith(QUARANTINE_SUFFIX)):
        raise PermissionError(f"refusing to write {table!r}: sinks only write *{STAGING_SUFFIX} or *{QUARANTINE_SUFFIX}")


def records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    return [{c: py_value(v) for c, v in zip(df.columns, row)} for row in df.itertuples(index=False, name=None)]
