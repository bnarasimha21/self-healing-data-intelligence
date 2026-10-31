"""OpenTelemetry tracing for runs. Off unless SHDI_OTEL=1.

    SHDI_OTEL=1                                  console exporter (spans printed to stderr)
    SHDI_OTEL=1 OTEL_EXPORTER_OTLP_ENDPOINT=...  OTLP over HTTP; needs
                                                 opentelemetry-exporter-otlp-proto-http

Every graph node runs inside a span named `shdi.<node>` with `shdi.run_id`,
`shdi.node` and, where the node knows them, `shdi.proposals.count`,
`shdi.checks.failed` and decision counts. A node that pauses for approval ends
its span with `shdi.interrupted=true`; a pause is not an error. Model calls get
a `chat <model>` span with `gen_ai.*` attributes (OpenTelemetry GenAI semantic
conventions).

Tests call `configure(provider)` to send spans to an in-memory exporter.
"""
from __future__ import annotations

import functools
import os
import sys
from typing import Any, Callable, Dict, Optional

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

try:  # LangGraph raises this to pause a run; it is control flow, not a failure.
    from langgraph.errors import GraphInterrupt
except ImportError:  # pragma: no cover
    GraphInterrupt = ()  # type: ignore

TRACER_NAME = "shdi"
_provider: Any = None
_env_provider: Any = None


def configure(provider: Any) -> None:
    """Use this TracerProvider for shdi spans (None turns tracing back to the env setting)."""
    global _provider
    _provider = provider


def _from_env() -> Any:
    global _env_provider
    if _env_provider is not None:
        return _env_provider
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": os.environ.get("OTEL_SERVICE_NAME", "shdi")}))
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT") or os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"):
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        except ImportError as e:
            raise RuntimeError(
                "OTEL_EXPORTER_OTLP_ENDPOINT is set but opentelemetry-exporter-otlp-proto-http is not installed"
            ) from e
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    else:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter(out=sys.stderr)))
    _env_provider = provider
    return provider


def enabled() -> bool:
    return _provider is not None or os.environ.get("SHDI_OTEL", "").strip().lower() in ("1", "true", "yes", "on")


def tracer() -> trace.Tracer:
    if _provider is not None:
        return _provider.get_tracer(TRACER_NAME)
    if enabled():
        return _from_env().get_tracer(TRACER_NAME)
    return trace.NoOpTracer()


def shutdown() -> None:
    """Flush spans from the env-configured provider (the CLI calls this before exiting)."""
    if _env_provider is not None:
        _env_provider.force_flush()


def _count_failed(checks: Any) -> Optional[int]:
    if not isinstance(checks, list):
        return None
    return sum(1 for c in checks if not c.get("passed", True))


def _result_attributes(state: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    attrs: Dict[str, Any] = {}
    proposals = result.get("proposals", state.get("proposals"))
    if proposals is not None:
        attrs["shdi.proposals.count"] = len(proposals)
        attrs["shdi.proposals.needs_approval"] = sum(1 for p in proposals if p.get("risk") == "needs_approval")
    checks = result.get("checks_after") if "checks_after" in result else result.get("checks_before")
    failed = _count_failed(checks)
    if failed is not None:
        attrs["shdi.checks.failed"] = failed
    decisions = result.get("decisions")
    if isinstance(decisions, dict):
        approved = sum(1 for d in decisions.values() if d.get("decision") == "approved")
        attrs["shdi.decisions.approved"] = approved
        attrs["shdi.decisions.rejected"] = len(decisions) - approved
    if "row_count" in result:
        attrs["shdi.rows"] = result["row_count"]
    if "status" in result:
        attrs["shdi.status"] = result["status"]
    return attrs


def traced_node(name: str, fn: Callable[[Dict[str, Any]], Dict[str, Any]]) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    @functools.wraps(fn)
    def wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
        with tracer().start_as_current_span(
            f"shdi.{name}", record_exception=False, set_status_on_exception=False
        ) as span:
            span.set_attribute("shdi.run_id", state.get("run_id", ""))
            span.set_attribute("shdi.node", name)
            try:
                result = fn(state)
            except GraphInterrupt:
                span.set_attribute("shdi.interrupted", True)
                raise
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
            for k, v in _result_attributes(state, result or {}).items():
                span.set_attribute(k, v)
            return result

    return wrapper
