"""Paths and settings read from the environment.

SHDI_HOME             working folder for state (default ./.shdi)
SHDI_CHECKPOINT_DB    LangGraph SQLite checkpoint file (default $SHDI_HOME/checkpoints.sqlite)
SHDI_AUDIT_DB         audit SQLite file (default $SHDI_HOME/audit.sqlite)
SHDI_SINK_DIR         where CSV sources get their staging/quarantine files (default $SHDI_HOME/out)
SHDI_LLM              off | plan | explain (default off)
SHDI_OTEL             1 to export OpenTelemetry spans (default off)
SHDI_PG_DSN           Postgres DSN for reading sources (use a SELECT-only role)
SHDI_PG_SINK_DSN      Postgres DSN for writing staging/quarantine (default SHDI_PG_DSN)
SHDI_ALLOWED_SOURCES  MCP server allowlist, comma separated fnmatch patterns
SHDI_CONTRACT_DIR     MCP server contract folder (default ./contracts)
"""
from __future__ import annotations

import os
from pathlib import Path


def home() -> Path:
    return Path(os.environ.get("SHDI_HOME", ".shdi"))


def checkpoint_db_path() -> str:
    return os.environ.get("SHDI_CHECKPOINT_DB") or str(home() / "checkpoints.sqlite")


def audit_db_path() -> str:
    return os.environ.get("SHDI_AUDIT_DB") or str(home() / "audit.sqlite")


def sink_dir() -> Path:
    return Path(os.environ.get("SHDI_SINK_DIR") or (home() / "out"))


def llm_mode() -> str:
    mode = os.environ.get("SHDI_LLM", "off").strip().lower()
    if mode not in ("off", "plan", "explain"):
        raise ValueError(f"SHDI_LLM must be off, plan or explain, not {mode!r}")
    return mode
