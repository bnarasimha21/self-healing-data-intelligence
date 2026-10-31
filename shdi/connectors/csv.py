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
    # utf-8-sig: a byte order mark is not part of the first column's name.
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = csv.reader(f)
            header = next(rows, None)
            if header is None:
                raise ValueError(f"CSV {path} is empty; it needs at least a header row")
            repeated = sorted({h for h in header if header.count(h) > 1})
            if repeated:
                raise ValueError(f"CSV {path} repeats the column names {repeated}; every column needs a unique name")
            # Checked here, not left to pandas, which silently drops a trailing
            # empty field ("1,2," under a two-column header).
            for i, row in enumerate(rows):
                if limit is not None and i >= limit:
                    break
                if len(row) > len(header):
                    raise ValueError(
                        f"CSV {path} has rows with more fields than the header "
                        f"({len(header)} columns, data row {i + 1} has {len(row)})"
                    )
    except UnicodeDecodeError as e:
        raise ValueError(f"CSV {path} is not UTF-8 text ({e.reason} at byte {e.start})") from None
    except csv.Error as e:
        raise ValueError(f"CSV {path} could not be parsed: {e}") from None
    try:
        with warnings.catch_warnings():
            # With index_col=False pandas warns, then drops the extra fields.
            warnings.simplefilter("error", pd.errors.ParserWarning)
            df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""], nrows=limit, index_col=False,
                             encoding="utf-8-sig")
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

    def _merged(self, new: Optional[pd.DataFrame], run_id: str, suffix: str) -> Optional[pd.DataFrame]:
        """One file's content with the run's earlier rows dropped and `new`
        appended, or None when there is nothing to write. Call with the lock held."""
        path = self._path(suffix)
        if path.exists():
            old = read_csv_text(path)
            old = old[old["shdi_run_id"] != run_id]
            return old if new is None else pd.concat([old, new.astype(object)], ignore_index=True)
        return new

    def _replace_files(self, contents: list) -> None:
        """Write every (suffix, frame) to a temp file first, then rename them
        all into place. A failure while building or writing any of them
        leaves every sink file as it was, and no temp file behind."""
        temps = []
        try:
            for suffix, frame in contents:
                if frame is None:
                    continue
                fd, tmp = tempfile.mkstemp(dir=self.dir, prefix=f".{self.name}.", suffix=".tmp")
                temps.append((tmp, self._path(suffix)))
                with os.fdopen(fd, "wb") as f:
                    f.write(to_csv_bytes(frame))
            for tmp, path in temps:
                os.replace(tmp, path)
        finally:
            for tmp, _ in temps:
                if os.path.exists(tmp):
                    os.unlink(tmp)

    def _replace_run(self, new: Optional[pd.DataFrame], run_id: str, suffix: str) -> None:
        self._replace_files([(suffix, self._merged(new, run_id, suffix))])

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
            q = None if quarantine.empty else self._quarantine_frame(quarantine, run_id)
            self._replace_files([
                (STAGING_SUFFIX, self._merged(with_meta(staging, run_id), run_id, STAGING_SUFFIX)),
                (QUARANTINE_SUFFIX, self._merged(q, run_id, QUARANTINE_SUFFIX)),
            ])
        return len(staging), len(quarantine)

    def delete_run(self, run_id: str) -> None:
        with self._locked():
            self._replace_files([
                (STAGING_SUFFIX, self._merged(None, run_id, STAGING_SUFFIX)),
                (QUARANTINE_SUFFIX, self._merged(None, run_id, QUARANTINE_SUFFIX)),
            ])

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
