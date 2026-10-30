"""Demo entry point: run the bundled dirty sample end to end.

`python main.py` auto-approves every proposal so it finishes in one go. For the
real review flow (pause, inspect, approve or reject, resume later) use the CLI:

    shdi run examples/dirty_sales.csv --contract contracts/sales.yaml
"""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from shdi import workflow
from shdi.graph import build_graph  # noqa: F401  (kept importable as main.build_graph)

HERE = Path(__file__).resolve().parent
SAMPLE = HERE / "examples" / "dirty_sales.csv"
CONTRACT = HERE / "contracts" / "sales.yaml"


def sample_dirty_csv() -> str:
    return SAMPLE.read_text()


def run_demo() -> None:
    load_dotenv()
    st = workflow.start_run(str(SAMPLE), CONTRACT, auto_approve=True, reviewer="demo:auto-approve")
    print("===== SELF-HEALING SUMMARY =====")
    print(st.get("summary", ""))


if __name__ == "__main__":
    run_demo()
