"""CSV source and sink.

The source reads every cell as text so nothing is reinterpreted on load; the
contract decides types. Empty cells are None. A file whose header repeats a
name, or whose rows carry more fields than the header, is refused rather than
read with renamed or shifted columns.

The sink writes `<stem>__shdi_staging.csv` and `<stem>__shdi_quarantine.csv`
in the sink folder (SHDI_SINK_DIR, default $SHDI_HOME/out), never next to or
over the input file. Every read-merge-replace of those files holds an
exclusive lock on `.<stem>.shdi.lock` in the same folder, so two runs writing
the same sink at once cannot drop each other's rows.
"""
from __future__ import annotations

import contextlib
import csv
import fcntl
import os
import tempfile
import warnings
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from .base import (
    QUARANTINE_SUFFIX,
    STAGING_SUFFIX,
    check_target,
    frame_fingerprint,
    quarantine_as_text,
    to_csv_bytes,
    with_meta,
)


def read_csv_text(path: Path, limit: Optional[int] = None) -> pd.DataFrame:
    with open(path, newline="", encoding="utf-8") as f:
        header = next(csv.reader(f), None)
    if header is None:
        raise ValueError(f"CSV {path} is empty; it needs at least a header row")
    repeated = sorted({h for h in header if header.count(h) > 1})
    if repeated:
        raise ValueError(f"CSV {path} repeats the column names {repeated}; every column needs a unique name")
    try:
        with warnings.catch_warnings():
            # With index_col=False pandas warns, then drops the extra fields.
            warnings.simplefilter("error", pd.errors.ParserWarning)
            df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""], nrows=limit, index_col=False)
    except pd.errors.ParserWarning:
        raise ValueError(f"CSV {path} has rows with more fields than the header ({len(header)} columns)") from None
    except pd.errors.ParserError as e:
        raise ValueError(f"CSV {path} could not be parsed: {e}") from None
    df = df.astype(object)
    return df.where(df.notna(), None)


class CsvSource:
    def __init__(self, path: str):
        self.path = Path(path).expanduser().resolve()
        self.uri = f"csv:{self.path}"
        self.name = self.path.stem

    def read(self, limit: Optional[int] = None) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"CSV source not found: {self.path}")
        return read_csv_text(self.path, limit)

    def snapshot(self):
        df = self.read()
        return df, frame_fingerprint(df)

    def fingerprint(self) -> str:
        return self.snapshot()[1]


class CsvSink:
    def __init__(self, directory: Path, name: str):
        self.dir = Path(directory)
        self.name = name

    def _path(self, suffix: str) -> Path:
        check_target(self.name + suffix)
        return self.dir / f"{self.name}{suffix}.csv"

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / f".{self.name}.shdi.lock", "a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _replace_run(self, new: Optional[pd.DataFrame], run_id: str, suffix: str) -> None:
        """Drop the run's earlier rows from one file and append `new`. Call with the lock held."""
        path = self._path(suffix)
        if path.exists():
            old = read_csv_text(path)
            old = old[old["shdi_run_id"] != run_id]
            if new is None:
                new = old
            else:
                new = pd.concat([old, new.astype(object)], ignore_index=True)
        elif new is None:
            return
        # Write to a temp file and rename, so a crash never leaves half a file.
        fd, tmp = tempfile.mkstemp(dir=self.dir, suffix=".tmp")
        with os.fdopen(fd, "wb") as f:
            f.write(to_csv_bytes(new))
        os.replace(tmp, path)

    def write_staging(self, df: pd.DataFrame, run_id: str) -> int:
        with self._locked():
            self._replace_run(with_meta(df, run_id), run_id, STAGING_SUFFIX)
        return len(df)

    def write_quarantine(self, df: pd.DataFrame, run_id: str) -> int:
        if df.empty:
            return 0
        with self._locked():
            self._replace_run(self._quarantine_frame(df, run_id), run_id, QUARANTINE_SUFFIX)
        return len(df)

    def _quarantine_frame(self, df: pd.DataFrame, run_id: str) -> pd.DataFrame:
        return with_meta(quarantine_as_text(df).set_index("shdi_source_row", drop=False), run_id)

    def write_run(self, staging: pd.DataFrame, quarantine: pd.DataFrame, run_id: str) -> tuple:
        """Replace the run's rows in both files under one lock. An empty
        quarantine still removes rows an earlier attempt of the run left."""
        with self._locked():
            self._replace_run(with_meta(staging, run_id), run_id, STAGING_SUFFIX)
            q = None if quarantine.empty else self._quarantine_frame(quarantine, run_id)
            self._replace_run(q, run_id, QUARANTINE_SUFFIX)
        return len(staging), len(quarantine)

    def delete_run(self, run_id: str) -> None:
        with self._locked():
            self._replace_run(None, run_id, STAGING_SUFFIX)
            self._replace_run(None, run_id, QUARANTINE_SUFFIX)

    def _read(self, suffix: str, run_id: Optional[str]) -> pd.DataFrame:
        path = self._path(suffix)
        if not path.exists():
            return pd.DataFrame()
        df = read_csv_text(path)
        return df[df["shdi_run_id"] == run_id].reset_index(drop=True) if run_id else df

    def read_staging(self, run_id: Optional[str] = None) -> pd.DataFrame:
        return self._read(STAGING_SUFFIX, run_id)

    def read_quarantine(self, run_id: Optional[str] = None) -> pd.DataFrame:
        return self._read(QUARANTINE_SUFFIX, run_id)
