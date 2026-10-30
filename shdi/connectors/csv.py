"""CSV source and sink.

The source reads every cell as text so nothing is reinterpreted on load; the
contract decides types. The sink writes `<stem>__shdi_staging.csv` and
`<stem>__shdi_quarantine.csv` in the sink folder (SHDI_SINK_DIR, default
$SHDI_HOME/out), never next to or over the input file.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

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
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[""], nrows=limit)


class CsvSource:
    def __init__(self, path: str):
        self.path = Path(path).expanduser().resolve()
        self.uri = f"csv:{self.path}"
        self.name = self.path.stem

    def read(self, limit: Optional[int] = None) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"CSV source not found: {self.path}")
        return read_csv_text(self.path, limit)

    def fingerprint(self) -> str:
        return frame_fingerprint(self.read())


class CsvSink:
    def __init__(self, directory: Path, name: str):
        self.dir = Path(directory)
        self.name = name

    def _path(self, suffix: str) -> Path:
        check_target(self.name + suffix)
        return self.dir / f"{self.name}{suffix}.csv"

    def _write(self, df: pd.DataFrame, run_id: str, suffix: str) -> int:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self._path(suffix)
        new = with_meta(df, run_id)
        if path.exists():
            old = read_csv_text(path)
            old = old[old["shdi_run_id"] != run_id]
            new = pd.concat([old, new.astype(object)], ignore_index=True)
        # Write to a temp file and rename, so a crash never leaves half a file.
        fd, tmp = tempfile.mkstemp(dir=self.dir, suffix=".tmp")
        with os.fdopen(fd, "wb") as f:
            f.write(to_csv_bytes(new))
        os.replace(tmp, path)
        return len(df)

    def write_staging(self, df: pd.DataFrame, run_id: str) -> int:
        return self._write(df, run_id, STAGING_SUFFIX)

    def write_quarantine(self, df: pd.DataFrame, run_id: str) -> int:
        if df.empty:
            return 0
        return self._write(quarantine_as_text(df).set_index("shdi_source_row", drop=False), run_id, QUARANTINE_SUFFIX)

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
