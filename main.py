import io
import re
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import operator
from typing_extensions import Annotated

from dotenv import load_dotenv
from langchain_gradient import ChatGradient
import os

import numpy as np
import pandas as pd
from dateutil import parser as dateparser

from langgraph.graph import StateGraph, START, END

load_dotenv()

# Columns the demo treats as required. Validation reports any that are missing and
# repair adds them (as NaN, then filled by the NA rules). The sample CSV has all three.
REQUIRED_COLUMNS = ["id", "date", "sales"]

# Repair passes allowed before giving up and reporting what is still failing.
MAX_REPAIR_PASSES = 2

def coerce_numeric(df: pd.DataFrame, cols: List[str], errors: str = "coerce") -> Tuple[pd.DataFrame, List[str]]:
    logs = []
    for c in cols:
        if c in df.columns:
            before_non_numeric = df[c].astype(str).str.contains(r"[^0-9\.\-eE]", regex=True, na=False).sum()
            df[c] = pd.to_numeric(df[c], errors=errors)
            after_na = df[c].isna().sum()
            logs.append(f"Coerced numeric for '{c}': non-numeric_before={before_non_numeric}, na_after={after_na}")
        else:
            logs.append(f"Column '{c}' not found for numeric coercion")
    return df, logs

def parse_dates(df: pd.DataFrame, cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    logs = []
    for c in cols:
        if c not in df.columns:
            logs.append(f"Date column '{c}' missing; skipped parse")
            continue
        def try_parse(x):
            if pd.isna(x) or (isinstance(x, float) and np.isnan(x)):
                return pd.NaT
            try:
                return dateparser.parse(str(x), dayfirst=False, yearfirst=False)
            except Exception:
                return pd.NaT
        before_na = df[c].isna().sum()
        # First, permissive parse to Python datetimes
        df[c] = df[c].apply(try_parse)
        # Then, coerce entire column to pandas datetime64[ns]
        df[c] = pd.to_datetime(df[c], errors="coerce")
        after_na = df[c].isna().sum()
        logs.append(f"Parsed dates for '{c}': na_before={before_na}, na_after={after_na}")
    return df, logs

def add_missing_columns(df: pd.DataFrame, required: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
    logs = []
    for col, default in required.items():
        if col not in df.columns:
            df[col] = default
            logs.append(f"Added missing column '{col}' with default={default!r}")
    return df, logs

def fill_na_rules(df: pd.DataFrame, rules: Dict[str, Any]) -> Tuple[pd.DataFrame, List[str]]:
    logs = []
    for col, default in rules.items():
        if col in df.columns:
            n_before = df[col].isna().sum()
            df[col] = df[col].fillna(default)
            n_after = df[col].isna().sum()
            if n_after < n_before:
                logs.append(f"Filled NA in '{col}': {n_before} -> {n_after} using default={default!r}")
    return df, logs

def clamp_numeric_ranges(df: pd.DataFrame, ranges: Dict[str, Tuple[Optional[float], Optional[float]]]) -> Tuple[pd.DataFrame, List[str]]:
    logs = []
    for col, (lo, hi) in ranges.items():
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            s = df[col]
            is_integer = pd.api.types.is_integer_dtype(s.dtype)
            # Prepare dtype-safe bounds
            lower = lo
            upper = hi
            if is_integer:
                if lo is not None:
                    lower = int(np.floor(lo))
                if hi is not None:
                    upper = int(np.ceil(hi))
            # Compute out-of-range mask before clipping
            before_out = ((s < lower) if lower is not None else False) | ((s > upper) if upper is not None else False)
            out_count = int(before_out.sum()) if hasattr(before_out, "sum") else 0
            # Apply clipping
            df[col] = s.clip(lower=lower, upper=upper)
            logs.append(f"Clamped '{col}' to [{lower}, {upper}] and adjusted {out_count} values")
    return df, logs

@dataclass
class ValidationResult:
    passed: bool
    issues: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

def validate(df: pd.DataFrame, expected_schema: Dict[str, str], required_columns: List[str]) -> ValidationResult:
    issues = []
    # Check columns
    for col in required_columns:
        if col not in df.columns:
            issues.append(f"Missing required column: {col}")

    # Check simple dtypes
    dtype_map = {
        "int": np.integer,
        "float": np.floating,
        "number": (np.integer, np.floating),
        "string": object,
        "date": "datetime64[ns]",
        "bool": np.bool_
    }

    for col, etype in expected_schema.items():
        if col not in df.columns:
            continue
        if etype == "date":
            if not np.issubdtype(df[col].dtype, np.datetime64):
                issues.append(f"Column '{col}' not parsed as datetime")
        elif etype in ("int", "float", "number"):
            if not pd.api.types.is_numeric_dtype(df[col]):
                issues.append(f"Column '{col}' not numeric as expected")
        elif etype == "bool":
            if df[col].dropna().map(lambda x: isinstance(x, (bool, np.bool_))).mean() < 1.0:
                issues.append(f"Column '{col}' not boolean-like")
        elif etype == "string":
            # pandas often uses 'object' for strings
            pass

    # Null ratios
    null_stats = {c: float(df[c].isna().mean()) for c in df.columns}
    for c, r in null_stats.items():
        if r > 0.3:
            issues.append(f"High null ratio in '{c}': {r:.0%}")

    return ValidationResult(passed=len(issues) == 0, issues=issues, stats={"null_ratios": null_stats})

@dataclass
class AnomalyReport:
    findings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

def detect_anomalies(df: pd.DataFrame, numeric_cols: Optional[List[str]] = None, date_cols: Optional[List[str]] = None) -> AnomalyReport:
    findings = []
    details = {}

    # Duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        findings.append(f"Found {dup_count} duplicate rows")
        details["duplicates"] = int(dup_count)

    # Simple z-score anomalies for numeric-like cols
    if numeric_cols is None:
        # Consider all columns; we'll coerce to numeric per-column
        numeric_cols = list(df.columns)
    z_anoms = {}
    p99_anoms = {}
    for c in numeric_cols:
        # Try coercing to numeric; skip if too few numeric values
        s = pd.to_numeric(df[c], errors="coerce").dropna().astype(float)
        if len(s) < 5:
            continue
        mean = float(s.mean())
        std = float(s.std(ddof=0))
        if std == 0.0:
            continue
        z = (s - mean) / std
        idx = s.index[(np.abs(z) > 3.0)]
        if len(idx) > 0:
            z_anoms[c] = idx.tolist()
            findings.append(f"Potential outliers in '{c}': {len(idx)} rows | |z|>3")
        # Percentile-based high-end outliers (robust for small N)
        hi = float(np.percentile(s, 99))
        hi_idx = s.index[s > hi]
        if len(hi_idx) > 0:
            p99_anoms[c] = hi_idx.tolist()
            findings.append(f"High values in '{c}' above P99: {len(hi_idx)} rows")
    if z_anoms:
        details["zscore_outliers"] = z_anoms
    if p99_anoms:
        details["p99_outliers"] = p99_anoms

    # Date sanity checks
    if date_cols is None:
        date_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]
    date_issues = {}
    for c in date_cols:
        s = df[c]
        # Ensure datetimelike before using .dt; coerce invalid to NaT
        if not np.issubdtype(s.dtype, np.datetime64):
            s = pd.to_datetime(s, errors="coerce")
        s = s.dropna()
        if s.empty:
            continue
        # absurdly old or future dates
        too_old = s[s.dt.year < 1970].index.tolist()
        now_naive = pd.Timestamp.now(tz=None)
        future = s[s > now_naive].index.tolist()
        if too_old or future:
            date_issues[c] = {"too_old": too_old, "future": future}
            findings.append(f"Suspicious dates in '{c}': old={len(too_old)}, future={len(future)}")
    if date_issues:
        details["date_anomalies"] = date_issues

    return AnomalyReport(findings=findings, details=details)

@dataclass
class PipelineState:
    # Inputs
    user_request: str
    csv_text: str
    # Working artifacts
    df: Optional[pd.DataFrame] = None
    validation: Optional[ValidationResult] = None
    anomalies: Optional[AnomalyReport] = None
    repair_log: List[str] = field(default_factory=list)
    iteration: int = 0
    done: bool = False
    summary: str = ""
    messages: Annotated[List[str], operator.add] = field(default_factory=list)

def node_plan(state: PipelineState) -> Dict[str, Any]:
    # Prefer LLM-generated plan when available; otherwise fallback to static list
    llm_plan_text = None
    try:
        llm = ChatGradient(
            model="llama3.3-70b-instruct",
            api_key=os.getenv("DIGITALOCEAN_INFERENCE_KEY")
        )
        # Constrain the model to produce only a brief bullet-point plan
        plan_prompt = (
            "You are a data quality pipeline planner. "
            "Return ONLY a concise plan as 5-8 bullet points, each starting with '- '. "
            "No code blocks, no CSV rewrites, no narrative, no markdown headings.\n\n"
            f"User request: {state.user_request}\n"
        )
        response = llm.invoke(plan_prompt)
        llm_plan_text = getattr(response, "content", "")
        if isinstance(llm_plan_text, str):
            llm_plan_text = llm_plan_text.strip()
        else:
            llm_plan_text = None
    except Exception as e:
        # If LLM fails, fall back to static plan and surface a brief note
        err_note = f"LLM plan generation failed: {e}"
        fallback_msg = [err_note]
    else:
        fallback_msg = []

    if llm_plan_text:
        # Post-process: keep only bullet lines, cap to 8 items
        lines = [l.strip() for l in llm_plan_text.splitlines()]
        bullets = [l for l in lines if l.startswith("- ")]
        if not bullets:
            # Fallback: convert non-empty lines to bullets
            bullets = [f"- {l}" for l in lines if l and not l.startswith(("#", "`"))]
        bullets = bullets[:8]
        msg = "Plan:\n" + "\n".join(bullets)
        return {"messages": [msg]}

    # Fallback static plan
    plan = [
        "Load CSV into DataFrame",
        "Infer candidate schema and required columns",
        "Validate required columns/types/nulls",
        "Apply self-healing repairs if validation fails",
        "Re-validate; if still failing and repair passes < 2, repeat",
        "Detect anomalies (duplicates, outliers, date issues) on the final data",
        "Summarize results"
    ]
    msg = "Plan:\n- " + "\n- ".join(plan)
    return {"messages": (fallback_msg + [msg])}

def node_load(state: PipelineState) -> Dict[str, Any]:
    df = pd.read_csv(io.StringIO(state.csv_text))
    msg = f"Loaded dataset with shape {df.shape}"
    return {"df": df, "messages": [msg]}

def infer_schema_and_requirements(df: pd.DataFrame) -> Tuple[Dict[str, str], List[str], List[str]]:
    # Heuristic expectations for the demo
    expected_schema = {}
    date_cols = []
    numeric_cols = []
    for c in df.columns:
        if re.search(r"date|time|timestamp", c, re.I):
            expected_schema[c] = "date"
            date_cols.append(c)
        elif pd.api.types.is_numeric_dtype(df[c]):
            expected_schema[c] = "number"
            numeric_cols.append(c)
        else:
            expected_schema[c] = "string"
    required_columns = list(REQUIRED_COLUMNS)
    return expected_schema, required_columns, date_cols

def node_validate(state: PipelineState) -> Dict[str, Any]:
    assert state.df is not None
    expected_schema, required_columns, _ = infer_schema_and_requirements(state.df)
    validation = validate(state.df, expected_schema, required_columns)
    if validation.passed:
        msg = "Validation: PASSED"
    else:
        msg = "Validation: FAILED\nIssues:\n- " + "\n- ".join(validation.issues)
    return {"validation": validation, "messages": [msg]}

def node_anomaly(state: PipelineState) -> Dict[str, Any]:
    assert state.df is not None
    _, _, date_cols = infer_schema_and_requirements(state.df)
    numeric_cols = [c for c in state.df.columns if pd.api.types.is_numeric_dtype(state.df[c])]
    anomalies = detect_anomalies(state.df, numeric_cols=numeric_cols, date_cols=date_cols)
    if anomalies.findings:
        msg = "Anomalies:\n- " + "\n- ".join(anomalies.findings)
    else:
        msg = "Anomalies: none detected"
    return {"anomalies": anomalies, "messages": [msg]}

def node_repair(state: PipelineState) -> Dict[str, Any]:
    assert state.df is not None
    expected_schema, required_columns, date_cols = infer_schema_and_requirements(state.df)

    df = state.df.copy()
    repair_log: List[str] = list(state.repair_log)

    # 1) Ensure required columns
    df, logs = add_missing_columns(df, {c: np.nan for c in required_columns if c not in df.columns})
    repair_log.extend(logs)

    # 2) Date parse
    df, logs = parse_dates(df, date_cols)
    repair_log.extend(logs)

    # 3) Numeric coercion (attempt on likely numeric object columns too)
    numeric_cols = [c for c, t in expected_schema.items() if t in ("int", "float", "number")]
    # add object columns that look mostly numeric
    obj_cols = [c for c in df.columns if df[c].dtype == object and c not in date_cols]
    mostly_numeric = []
    for c in obj_cols:
        s = df[c].astype(str)
        numeric_like = s.str.match(r"^[\s]*[-+]?((\d+\.?\d*)|(\.\d+))([eE][-+]?\d+)?[\s]*$").fillna(False)
        if numeric_like.mean() >= 0.5:
            mostly_numeric.append(c)
    numeric_targets = list(dict.fromkeys(numeric_cols + mostly_numeric))
    df, logs = coerce_numeric(df, numeric_targets)
    repair_log.extend(logs)

    # 4) Recompute schema after coercion and fill NA for common fields
    expected_schema, _, _ = infer_schema_and_requirements(df)
    fill_rules = {}
    for c, t in expected_schema.items():
        if t in ("int", "float", "number"):
            fill_rules[c] = 0
        elif t == "string":
            fill_rules[c] = ""
        elif t == "date":
            fill_rules[c] = pd.NaT
    df, logs = fill_na_rules(df, fill_rules)
    repair_log.extend(logs)

    # 5) Clamp numeric ranges
    rng = {}
    for c in numeric_cols:
        s = df[c].dropna().astype(float)
        if len(s) >= 5:
            lo = np.percentile(s, 1)
            hi = np.percentile(s, 99)
            rng[c] = (lo, hi)
    df, logs = clamp_numeric_ranges(df, rng)
    repair_log.extend(logs)

    # Log
    if logs:
        msg = "Repairs applied:\n- " + "\n- ".join(repair_log[-10:])
    else:
        msg = "No repairs necessary"

    return {"df": df, "repair_log": repair_log, "messages": [msg]}

def node_decide(state: PipelineState) -> Dict[str, Any]:
    # Decide to loop or finish
    next_iteration = state.iteration + 1
    done = False
    if state.validation and state.validation.passed:
        done = True
    elif next_iteration > MAX_REPAIR_PASSES:  # iteration 1 is the pre-repair validation
        done = True
    # Append a compact snapshot of the DataFrame after this iteration
    df_preview = ""
    if state.df is not None:
        try:
            df_preview = state.df.head(10).to_string(index=False)
        except Exception:
            df_preview = "<unable to render df preview>"
    msg = f"DataFrame after iteration {next_iteration}:\n" + df_preview
    return {"iteration": next_iteration, "done": done, "messages": [msg]}

def node_summarize(state: PipelineState) -> Dict[str, Any]:
    assert state.df is not None
    shape = state.df.shape
    cols = ", ".join(state.df.columns)
    val = state.validation
    an = state.anomalies

    summary_lines: List[str] = []
    summary_lines.append(f"Final shape: {shape}")
    summary_lines.append(f"Columns: {cols}")
    if val:
        summary_lines.append(f"Validation passed: {val.passed}")
        if val.issues:
            summary_lines.append("Remaining issues:")
            summary_lines.extend([f"- {i}" for i in val.issues])
    if an and an.findings:
        summary_lines.append("Anomaly notes:")
        summary_lines.extend([f"- {f}" for f in an.findings])
    if state.repair_log:
        summary_lines.append(f"Total repairs logged: {len(state.repair_log)}")
    # Include accumulated node messages at the top
    if state.messages:
        summary_lines = state.messages + [""] + summary_lines
    new_summary = "\n".join(summary_lines) + "\n"
    return {"summary": new_summary}

def build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("plan", node_plan)
    graph.add_node("load", node_load)
    graph.add_node("validate", node_validate)
    graph.add_node("anomaly", node_anomaly)
    graph.add_node("repair", node_repair)
    graph.add_node("decide", node_decide)
    graph.add_node("summarize", node_summarize)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "load")
    graph.add_edge("load", "validate")
    graph.add_edge("repair", "validate")
    graph.add_edge("validate", "decide")
    # Repair/validate loop until validation passes or the pass limit is hit,
    # then run anomaly detection once on the final data.
    graph.add_conditional_edges(
        "decide",
        lambda s: "anomaly" if s.done else "repair",
        {"anomaly": "anomaly", "repair": "repair"}
    )
    graph.add_edge("anomaly", "summarize")
    graph.add_edge("summarize", END)
    return graph.compile()

# Sample dirty CSV data
# Columns: id, date, sales, region, notes
def sample_dirty_csv() -> str:
    return """id,date,sales,region,notes
1,2024-01-05,100,North,ok
2,01/06/2024,200,North,ok
3,2024/07/01,three hundred,South,ok
4,2050-12-01,1000,East,ok
5,1960-01-01,50,West,ok
5,1960-01-01,50,West,ok
6,2025-01-01,400000,North,ok
7,,700,South,ok
8,2024-02-29,80,East,ok
9,2024-03-10,100,West,ok
"""

def run_demo():
    state = PipelineState(
        user_request="Clean the file, detect anomalies, fix common issues, and summarize.",
        csv_text=sample_dirty_csv()
    )
    graph = build_graph()
    final_state = graph.invoke(state)
    print("===== SELF-HEALING SUMMARY =====")
    if isinstance(final_state, dict):
        print(final_state.get("summary", ""))
    else:
        print(getattr(final_state, "summary", ""))

if __name__ == "__main__":
    run_demo()