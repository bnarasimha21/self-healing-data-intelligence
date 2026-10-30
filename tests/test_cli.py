"""The CLI review loop across separate processes: each command is a new
Python process, so the pause genuinely survives a restart."""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys

from shdi.contract import load_contract

from conftest import CONTRACT, FIXTURES, ROOT


def shdi(*args, env_home):
    env = {**os.environ, "SHDI_HOME": str(env_home), "SHDI_LLM": "off"}
    env.pop("SHDI_OTEL", None)
    return subprocess.run([sys.executable, "-m", "shdi", *args], capture_output=True, text=True, env=env, cwd=ROOT)


def test_cli_run_review_decide_audit(dirty_csv, tmp_path):
    home = tmp_path / "home"
    sink = tmp_path / "out"
    r = shdi("run", str(dirty_csv), "--contract", str(CONTRACT), "--run-id", "cli1", "--sink-dir", str(sink), "--json", env_home=home)
    assert r.returncode == 0, r.stderr
    st = json.loads(r.stdout)
    assert st["status"] == "awaiting_approval"
    assert not sink.exists()

    r = shdi("review", "cli1", env_home=home)
    assert r.returncode == 0 and "need approval" in r.stdout
    assert "quarantine_row sales" in r.stdout or "quarantine_row" in r.stdout

    rescale = next(p["id"] for p in st["proposals"] if p["rule"] == "rescale")
    r = shdi("reject", "cli1", "--ids", rescale, "--reviewer", "erin", "--json", env_home=home)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["status"] == "awaiting_approval"

    r = shdi("approve", "cli1", "--all", "--reviewer", "erin", env_home=home)
    assert r.returncode == 0, r.stderr
    assert "Run cli1:" in r.stdout and "Before/after" in r.stdout

    r = shdi("status", "cli1", "--json", env_home=home)
    done = json.loads(r.stdout)
    assert done["status"] == "completed"
    assert done["decisions"][rescale]["decision"] == "rejected"
    assert (sink / "dirty_sales__shdi_staging.csv").exists()
    assert (sink / "dirty_sales__shdi_quarantine.csv").exists()

    r = shdi("audit", "cli1", env_home=home)
    events = [json.loads(line) for line in r.stdout.splitlines()]
    assert {e["event"] for e in events} == {"decision", "applied"}
    assert sum(1 for e in events if e["event"] == "decision") == len(st["proposals"])
    assert all(e["reviewer"] == "erin" for e in events if e["event"] == "decision" and e["proposal_id"] == rescale)

    r = shdi("approve", "cli1", "--ids", rescale, env_home=home)
    assert r.returncode == 0 and "already decided, ignored" in r.stdout


def test_cli_errors_are_clean(tmp_path):
    home = tmp_path / "home"
    r = shdi("status", "nope", env_home=home)
    assert r.returncode == 2 and "unknown run" in r.stderr
    r = shdi("run", "postgres://user:pw@host/db", "--contract", str(CONTRACT), env_home=home)
    assert r.returncode == 2 and "error:" in r.stderr
    assert "pw" not in r.stderr


def test_main_demo_runs_offline(tmp_path):
    env = {**os.environ, "SHDI_HOME": str(tmp_path / "home"), "SHDI_LLM": "off"}
    for k in ("GRADIENT_MODEL_ACCESS_KEY", "DIGITALOCEAN_INFERENCE_KEY", "SHDI_OTEL"):
        env.pop(k, None)
    r = subprocess.run([sys.executable, "main.py"], capture_output=True, text=True, env=env, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "SELF-HEALING SUMMARY" in r.stdout and ": passed" in r.stdout


def test_required_columns_and_null_literal_removed_from_code():
    files = [ROOT / "main.py", *sorted((ROOT / "shdi").rglob("*.py"))]
    for f in files:
        src = f.read_text()
        assert "REQUIRED_COLUMNS" not in src, f
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Constant) and isinstance(node.value, float):
                assert node.value != 0.3, f"hard-coded 0.3 in {f}"


def test_infer_contract_cli_writes_loadable_yaml(tmp_path):
    out = tmp_path / "draft.yaml"
    proc = subprocess.run(
        [sys.executable, "-m", "shdi", "infer-contract", str(FIXTURES / "dirty_sales.csv"), "-o", str(out)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
    draft = load_contract(out)
    assert set(draft.columns) == {"id", "date", "sales", "region", "notes"}
    assert draft.columns["date"].type == "date"
    assert draft.primary_key == ["id"]
    assert draft.columns["sales"].measure and not draft.columns["id"].measure
