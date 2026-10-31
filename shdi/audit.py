"""Append-only audit trail of proposal decisions and what apply did with them.

Stored in SQLite (path from SHDI_AUDIT_DB, default $SHDI_HOME/audit.sqlite).
Triggers reject UPDATE and DELETE, so the table can only grow. A unique key on
(run_id, proposal_id, event) makes every write idempotent: replaying a
decision after a graph resume, or a client sending the same decision twice,
cannot create a second record.

`repair_log` (a SQL view and a function of the same name) is the
human-readable log over this table.
"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import audit_db_path

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    proposal_id TEXT NOT NULL,
    event TEXT NOT NULL CHECK (event IN ('decision', 'applied', 'skipped')),
    decision TEXT,
    reviewer TEXT,
    ts TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    rule TEXT NOT NULL,
    column_name TEXT,
    row_keys TEXT NOT NULL,
    old_values TEXT NOT NULL,
    new_values TEXT NOT NULL,
    reason TEXT NOT NULL,
    risk TEXT NOT NULL,
    detail TEXT,
    proposal TEXT NOT NULL,
    UNIQUE (run_id, proposal_id, event)
);
CREATE TRIGGER IF NOT EXISTS audit_events_no_update BEFORE UPDATE ON audit_events
BEGIN SELECT RAISE(ABORT, 'audit_events is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_events_no_delete BEFORE DELETE ON audit_events
BEGIN SELECT RAISE(ABORT, 'audit_events is append-only'); END;
CREATE VIEW IF NOT EXISTS repair_log AS
SELECT run_id, seq, ts,
       proposal_id || ' ' || rule || COALESCE(' ' || column_name, '') || ' rows=' || row_keys ||
       ' old=' || old_values || ' new=' || new_values || ' -> ' || event ||
       COALESCE(' ' || decision, '') || COALESCE(' by ' || reviewer, '') ||
       COALESCE(' (' || detail || ')', '') AS line
FROM audit_events ORDER BY seq;
"""


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


class AuditLog:
    def __init__(self, path: Optional[str] = None):
        self.path = Path(path or audit_db_path())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as con:
            con.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=30)
        con.row_factory = sqlite3.Row
        return con

    def record(
        self,
        run_id: str,
        proposal: Dict[str, Any],
        event: str,
        input_hash: str,
        decision: Optional[str] = None,
        reviewer: Optional[str] = None,
        ts: Optional[str] = None,
        detail: Optional[str] = None,
    ) -> bool:
        """Insert one event. Returns False when it was already recorded."""
        with self._connect() as con:
            cur = con.execute(
                """INSERT OR IGNORE INTO audit_events
                (run_id, proposal_id, event, decision, reviewer, ts, input_hash, rule, column_name,
                 row_keys, old_values, new_values, reason, risk, detail, proposal)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id,
                    proposal["id"],
                    event,
                    decision,
                    reviewer,
                    ts or now_iso(),
                    input_hash,
                    proposal["rule"],
                    proposal.get("column"),
                    json.dumps(proposal["row_keys"]),
                    json.dumps(proposal["old_values"]),
                    json.dumps(proposal["new_values"]),
                    proposal["reason"],
                    proposal["risk"],
                    detail,
                    json.dumps(proposal, sort_keys=True),
                ),
            )
            return cur.rowcount == 1

    def events(self, run_id: str, event: Optional[str] = None) -> List[Dict[str, Any]]:
        q = "SELECT * FROM audit_events WHERE run_id = ?"
        args: List[Any] = [run_id]
        if event:
            q += " AND event = ?"
            args.append(event)
        with self._connect() as con:
            rows = con.execute(q + " ORDER BY seq", args).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            for k in ("row_keys", "old_values", "new_values", "proposal"):
                d[k] = json.loads(d[k])
            out.append(d)
        return out

    def decisions(self, run_id: str) -> Dict[str, Dict[str, Any]]:
        return {e["proposal_id"]: e for e in self.events(run_id, "decision")}


def repair_log(run_id: str, path: Optional[str] = None) -> List[str]:
    log = AuditLog(path)
    with log._connect() as con:
        return [r["line"] for r in con.execute("SELECT line FROM repair_log WHERE run_id = ?", (run_id,))]
