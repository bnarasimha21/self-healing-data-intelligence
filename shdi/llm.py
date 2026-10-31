"""Optional model calls. Controlled by SHDI_LLM:

    off      (default) no model is ever called or imported
    plan     the model writes the run plan shown in the summary (informational)
    explain  the model writes a one-line plain-English reason for each
             needs_approval proposal, shown to the reviewer in `shdi review`

The model never produces values: explanations are stored as text on the
proposal and `apply` does not read them. Any model failure is reported in the
run messages and the run continues without explanations.

The default model is DigitalOcean Gradient through `langchain-gradient`
(install requirements-llm.txt), imported only when a mode needs it.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from .config import llm_mode
from .tracing import tracer

ChatModelFactory = Callable[[], Any]
DEFAULT_MODEL = "llama3.3-70b-instruct"
BATCH = 20
MAX_EXPLANATION = 240


class LLMUnavailable(RuntimeError):
    pass


def model_name() -> str:
    return os.environ.get("SHDI_LLM_MODEL", DEFAULT_MODEL)


def default_chat_model() -> Any:
    try:
        from langchain_gradient import ChatGradient
    except ImportError as e:
        raise LLMUnavailable("langchain-gradient is not installed (pip install -r requirements-llm.txt)") from e
    key = os.environ.get("DIGITALOCEAN_INFERENCE_KEY")
    if not key:
        raise LLMUnavailable("DIGITALOCEAN_INFERENCE_KEY is not set")
    return ChatGradient(model=model_name(), api_key=key)


def _invoke(model: Any, prompt: str, operation: str) -> str:
    name = getattr(model, "model_name", None) or getattr(model, "model", None) or model_name()
    with tracer().start_as_current_span(f"chat {name}") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.provider.name", "digitalocean.gradient")
        span.set_attribute("gen_ai.request.model", str(name))
        span.set_attribute("shdi.llm.purpose", operation)
        response = model.invoke(prompt)
        usage = getattr(response, "usage_metadata", None) or {}
        if isinstance(usage, dict):
            if "input_tokens" in usage:
                span.set_attribute("gen_ai.usage.input_tokens", int(usage["input_tokens"]))
            if "output_tokens" in usage:
                span.set_attribute("gen_ai.usage.output_tokens", int(usage["output_tokens"]))
        content = getattr(response, "content", response)
        return content if isinstance(content, str) else str(content)


def write_plan(user_request: str, factory: Optional[ChatModelFactory] = None) -> Tuple[Optional[List[str]], Optional[str]]:
    """Return (plan lines, error). Only called when SHDI_LLM=plan."""
    try:
        model = (factory or default_chat_model)()
        text = _invoke(
            model,
            "You are planning a data quality run that validates a dataset against a data contract, "
            "proposes repairs, waits for a human to approve them and writes the result to staging tables. "
            "Return ONLY 5-8 bullet points, each starting with '- '. No code, no headings.\n\n"
            f"User request: {user_request}\n",
            "plan",
        )
    except Exception as e:
        return None, f"LLM plan unavailable, using the static plan: {e}"
    lines = [ln.strip()[2:].strip() for ln in text.splitlines() if ln.strip().startswith("- ")]
    return (lines or None), (None if lines else "LLM plan was empty, using the static plan")


def _clean(text: Any) -> str:
    s = " ".join(str(text).split())
    return s[:MAX_EXPLANATION]


def _parse_json_object(text: str) -> Dict[str, Any]:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def explain_proposals(
    proposals: List[Dict[str, Any]], factory: Optional[ChatModelFactory] = None
) -> Tuple[Dict[str, str], Optional[str]]:
    """Return ({proposal id: explanation}, error) for needs_approval proposals."""
    targets = [p for p in proposals if p["risk"] == "needs_approval"]
    if not targets:
        return {}, None
    try:
        model = (factory or default_chat_model)()
    except Exception as e:
        return {}, f"LLM explanations unavailable: {e}"
    out: Dict[str, str] = {}
    errors: List[str] = []
    for i in range(0, len(targets), BATCH):
        batch = targets[i : i + BATCH]
        brief = [
            {
                "id": p["id"],
                "rule": p["rule"],
                "column": p.get("column"),
                "rows": p["row_keys"][:5],
                "old": p["old_values"][:5],
                "new": p["new_values"][:5],
                "reason": p["reason"],
            }
            for p in batch
        ]
        prompt = (
            "A data reviewer must approve or reject each proposed change below. For each one, write ONE short "
            "plain-English sentence saying what it changes and why, for someone who is not a data engineer. "
            "Do not suggest different values. Answer with only a JSON object mapping each id to its sentence.\n\n"
            + json.dumps(brief, default=str)
        )
        try:
            data = _parse_json_object(_invoke(model, prompt, "explain"))
        except Exception as e:
            errors.append(str(e))
            continue
        for p in batch:
            text = data.get(p["id"])
            if text and _clean(text):
                out[p["id"]] = _clean(text)
    missing = len(targets) - len(out)
    err = None
    if errors:
        err = f"LLM explanations failed for some proposals: {errors[0]}"
    elif missing:
        err = f"LLM gave no explanation for {missing} of {len(targets)} proposals"
    return out, err


__all__ = ["llm_mode", "write_plan", "explain_proposals", "default_chat_model", "LLMUnavailable"]
