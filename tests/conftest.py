from __future__ import annotations

import datetime as dt
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
CONTRACT = ROOT / "contracts" / "sales.yaml"
AS_OF = dt.date(2026, 10, 30)


@pytest.fixture(autouse=True)
def shdi_env(tmp_path, monkeypatch):
    """Every test gets its own state folder and runs with the LLM off."""
    home = tmp_path / "shdi_home"
    monkeypatch.setenv("SHDI_HOME", str(home))
    monkeypatch.delenv("SHDI_CHECKPOINT_DB", raising=False)
    monkeypatch.delenv("SHDI_AUDIT_DB", raising=False)
    monkeypatch.delenv("SHDI_SINK_DIR", raising=False)
    monkeypatch.setenv("SHDI_LLM", "off")
    monkeypatch.delenv("SHDI_OTEL", raising=False)
    return home


@pytest.fixture
def dirty_csv(tmp_path) -> Path:
    """A private copy of the dirty fixture, so a test may modify it."""
    dst = tmp_path / "dirty_sales.csv"
    shutil.copy(FIXTURES / "dirty_sales.csv", dst)
    return dst


def find(proposals: List[Dict[str, Any]], rule: str, row: Optional[int] = None, column: Optional[str] = None) -> Dict[str, Any]:
    hits = [
        p
        for p in proposals
        if p["rule"] == rule and (row is None or row in p["rows"]) and (column is None or p["column"] == column)
    ]
    assert len(hits) == 1, f"expected one {rule} proposal for row={row} column={column}, got {hits}"
    return hits[0]


def interrupt_payload(snapshot) -> Dict[str, Any]:
    interrupts = list(getattr(snapshot, "interrupts", ()) or ())
    if not interrupts:
        for t in snapshot.tasks:
            interrupts.extend(t.interrupts)
    assert interrupts, "graph is not paused at an interrupt"
    return interrupts[0].value


def initial_state(source: str, run_id: str, **extra) -> Dict[str, Any]:
    from shdi.connectors import canonical_uri
    from shdi.contract import load_contract

    state = {
        "run_id": run_id,
        "source_uri": canonical_uri(source),
        "sink_dir": None,
        "contract": load_contract(CONTRACT).model_dump(mode="json"),
        "as_of": AS_OF.isoformat(),
        "auto_approve": False,
        "reviewer": None,
        "user_request": "test",
        "messages": [],
    }
    state.update(extra)
    return state
