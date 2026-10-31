"""Issue 6: the LLM is optional, and when on it only explains; it never changes data."""
from __future__ import annotations

import io
import json
import os
import subprocess
import sys

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from shdi import cli, tracing, workflow
from shdi.connectors import open_sink
from shdi.connectors.base import to_csv_bytes
from shdi.graph import build_graph, make_checkpointer

from conftest import AS_OF, CONTRACT, ROOT


class Reply:
    def __init__(self, content):
        self.content = content
        self.usage_metadata = {"input_tokens": 120, "output_tokens": 30}


class StubModel:
    """Answers the explain prompt with one sentence per id, and the plan prompt with bullets.

    It also tries to smuggle in a different value; that text must stay text."""

    model_name = "stub-model"

    def __init__(self):
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if "JSON object" in prompt:
            items = json.loads(prompt.split("\n\n", 1)[1])
            return Reply(json.dumps({i["id"]: f"Fixes {i['rule']} on {i['column']}; set it to 999999 instead." for i in items}))
        return Reply("- check the contract\n- ask a human\n- write staging")


def run(csv, tmp_path, run_id, factory=None, approve=True):
    db = tmp_path / f"{run_id}.sqlite"
    st = workflow.start_run(str(csv), CONTRACT, as_of=AS_OF, run_id=run_id, sink_dir=str(tmp_path / run_id),
                            graph=build_graph(make_checkpointer(str(db)), chat_model_factory=factory))
    if approve:
        done = workflow.decide(run_id, approve_all=True, reviewer="t",
                               graph=build_graph(make_checkpointer(str(db)), chat_model_factory=factory))
        return st, done
    return st, None


def staged_bytes(csv, tmp_path, run_id):
    df = open_sink(f"csv:{csv}", str(tmp_path / run_id)).read_staging(run_id)
    return to_csv_bytes(df[[c for c in df.columns if c not in ("shdi_run_id", "shdi_applied_at")]])


def test_import_without_langchain_gradient():
    code = (
        "import sys; sys.modules['langchain_gradient'] = None\n"
        "import shdi, shdi.graph, shdi.llm, shdi.cli, shdi.workflow, main\n"
        "from shdi.llm import default_chat_model, LLMUnavailable\n"
        "try:\n    default_chat_model()\nexcept LLMUnavailable as e:\n    print('unavailable:', e)\n"
    )
    env = {**os.environ, "SHDI_LLM": "off"}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "unavailable: langchain-gradient is not installed" in r.stdout


def test_explain_mode_fills_every_needs_approval_and_changes_no_data(dirty_csv, tmp_path, monkeypatch):
    monkeypatch.setenv("SHDI_LLM", "off")
    off_start, _ = run(dirty_csv, tmp_path, "off")

    monkeypatch.setenv("SHDI_LLM", "explain")
    stub = StubModel()
    on_start, on_done = run(dirty_csv, tmp_path, "on", factory=lambda: stub)

    need = [p for p in on_start["proposals"] if p["risk"] == "needs_approval"]
    assert need and all(p.get("explanation") for p in need)
    assert all(not p.get("explanation") for p in on_start["proposals"] if p["risk"] == "low")
    assert len(stub.prompts) == 1  # one batched call for 8 proposals

    # Same proposals apart from the explanation text, and identical applied data.
    strip = lambda ps: [{k: v for k, v in p.items() if k != "explanation"} for p in ps]  # noqa: E731
    assert strip(on_start["proposals"]) == strip(off_start["proposals"])
    assert staged_bytes(dirty_csv, tmp_path, "on") == staged_bytes(dirty_csv, tmp_path, "off")
    assert "999999" not in staged_bytes(dirty_csv, tmp_path, "on").decode()
    assert on_done["outcome"] == "passed"


def test_review_output_shows_the_explanation(dirty_csv, tmp_path, monkeypatch):
    monkeypatch.setenv("SHDI_LLM", "explain")
    st, _ = run(dirty_csv, tmp_path, "rv", factory=StubModel, approve=False)
    buf = io.StringIO()
    cli.print_proposals(st["pending"], out=buf)
    assert buf.getvalue().count("explanation: Fixes ") == len(st["needs_approval"])


def test_model_failure_never_blocks_the_run(dirty_csv, tmp_path, monkeypatch):
    monkeypatch.setenv("SHDI_LLM", "explain")

    class Broken:
        def invoke(self, prompt):
            raise TimeoutError("model timed out")

    st, done = run(dirty_csv, tmp_path, "br", factory=Broken)
    assert not any(p.get("explanation") for p in st["proposals"])
    assert "model timed out" in done["summary"]
    assert done["outcome"] == "passed"

    class Garbage:
        def invoke(self, prompt):
            return Reply("I'd rather not.")

    st, done = run(dirty_csv, tmp_path, "gb", factory=Garbage)
    assert not any(p.get("explanation") for p in st["proposals"])
    assert "no explanation for" in done["summary"]


def test_missing_gradient_in_explain_mode_is_reported_not_fatal(dirty_csv, tmp_path, monkeypatch):
    monkeypatch.setenv("SHDI_LLM", "explain")
    monkeypatch.setitem(sys.modules, "langchain_gradient", None)
    st, done = run(dirty_csv, tmp_path, "ng")
    assert "langchain-gradient is not installed" in done["summary"]
    assert done["outcome"] == "passed"


def test_plan_mode_uses_the_model_plan_and_traces_the_call(dirty_csv, tmp_path, monkeypatch):
    monkeypatch.setenv("SHDI_LLM", "plan")
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracing.configure(provider)
    try:
        _, done = run(dirty_csv, tmp_path, "pl", factory=StubModel)
    finally:
        tracing.configure(None)
    assert "- ask a human" in done["summary"]
    chat = [s for s in exporter.get_finished_spans() if s.name == "chat stub-model"]
    assert len(chat) == 1
    a = chat[0].attributes
    assert a["gen_ai.operation.name"] == "chat" and a["gen_ai.request.model"] == "stub-model"
    assert a["gen_ai.usage.input_tokens"] == 120
    plan_span = next(s for s in exporter.get_finished_spans() if s.name == "shdi.plan")
    assert chat[0].parent.span_id == plan_span.context.span_id


def test_bad_llm_mode_is_rejected(monkeypatch):
    from shdi.config import llm_mode

    monkeypatch.setenv("SHDI_LLM", "always")
    with pytest.raises(ValueError):
        llm_mode()
