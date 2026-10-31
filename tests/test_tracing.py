"""Issue 6: one OpenTelemetry span per executed node, through the real graph."""
from __future__ import annotations

import os
import subprocess
import sys

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from shdi import tracing, workflow
from shdi.audit import AuditLog
from shdi.graph import NODE_ORDER, build_graph, make_checkpointer

from conftest import AS_OF, CONTRACT, ROOT


@pytest.fixture
def spans():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracing.configure(provider)
    yield exporter
    tracing.configure(None)
    provider.shutdown()


def node_spans(exporter):
    return [s for s in exporter.get_finished_spans() if s.attributes.get("shdi.node")]


def test_one_span_per_node_in_order_with_counts_matching_the_audit(dirty_csv, tmp_path, spans):
    graph = build_graph(make_checkpointer(str(tmp_path / "cp.sqlite")))
    st = workflow.start_run(str(dirty_csv), CONTRACT, as_of=AS_OF, graph=graph, run_id="tr1", auto_approve=True)
    assert st["outcome"] == "passed"

    nodes = node_spans(spans)
    assert [s.name for s in nodes] == [f"shdi.{n}" for n in NODE_ORDER]
    assert all(s.attributes["shdi.run_id"] == "tr1" for s in nodes)
    assert all(s.status.status_code != StatusCode.ERROR for s in nodes)
    by_node = {s.attributes["shdi.node"]: s for s in nodes}

    audit = AuditLog()
    decided = {e["proposal_id"] for e in audit.events("tr1", "decision")}
    applied = audit.events("tr1", "applied")
    assert by_node["propose"].attributes["shdi.proposals.count"] == len(decided) == len(st["proposals"])
    assert by_node["approve"].attributes["shdi.decisions.approved"] == len(decided)
    assert by_node["validate"].attributes["shdi.checks.failed"] == 9
    assert by_node["apply"].attributes["shdi.checks.failed"] == 0
    assert by_node["load"].attributes["shdi.rows"] == 12
    events = [e for e in by_node["apply"].events if e.name == "shdi.apply.written"]
    assert len(events) == 1 and events[0].attributes["shdi.applied"] == len(applied)

    # All node spans sit under one root span, so the run is one trace.
    root = [s for s in spans.get_finished_spans() if s.name == "shdi.run"]
    assert len(root) == 1
    assert {s.context.trace_id for s in nodes} == {root[0].context.trace_id}
    assert all(s.parent is not None and s.parent.span_id == root[0].context.span_id for s in nodes)


def test_pause_is_an_attribute_not_an_error(dirty_csv, tmp_path, spans):
    db = tmp_path / "cp.sqlite"
    workflow.start_run(str(dirty_csv), CONTRACT, as_of=AS_OF, graph=build_graph(make_checkpointer(str(db))), run_id="tr2")
    first = node_spans(spans)
    assert [s.attributes["shdi.node"] for s in first] == ["plan", "load", "validate", "propose", "approve"]
    paused = first[-1]
    assert paused.attributes["shdi.interrupted"] is True
    assert paused.status.status_code != StatusCode.ERROR

    spans.clear()
    workflow.decide("tr2", approve_all=True, reviewer="otel", graph=build_graph(make_checkpointer(str(db))))
    second = node_spans(spans)
    assert [s.attributes["shdi.node"] for s in second] == ["approve", "apply", "summarize"]
    assert "shdi.interrupted" not in second[0].attributes
    resume = [s for s in spans.get_finished_spans() if s.name == "shdi.resume"]
    assert len(resume) == 1 and resume[0].attributes["shdi.reviewer"] == "otel"


def test_tracing_is_off_by_default():
    assert not tracing.enabled()
    with tracing.tracer().start_as_current_span("x") as span:
        assert not span.is_recording()


def test_console_exporter_via_env(dirty_csv, tmp_path):
    env = {**os.environ, "SHDI_HOME": str(tmp_path / "home"), "SHDI_LLM": "off", "SHDI_OTEL": "1"}
    env.pop("OTEL_EXPORTER_OTLP_ENDPOINT", None)
    env.pop("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", None)
    r = subprocess.run(
        [sys.executable, "-m", "shdi", "run", str(dirty_csv), "--contract", str(CONTRACT), "--auto-approve",
         "--sink-dir", str(tmp_path / "out")],
        capture_output=True, text=True, env=env, cwd=ROOT,
    )
    assert r.returncode == 0, r.stderr
    for n in NODE_ORDER:
        assert f'"name": "shdi.{n}"' in r.stderr
