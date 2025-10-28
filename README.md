# Self-Healing Data Intelligence

## Overview
Self‑Healing Data Intelligence is a small, demonstrative pipeline that ingests CSV data, validates schema and types, detects anomalies, applies deterministic repairs, and summarizes results. It is orchestrated with LangGraph (graph-based state transitions) and can optionally use an LLM just to propose a short execution plan (no data mutation by the LLM). The system distinguishes between structural validation (schema, dtypes, nulls) and anomaly detection (duplicates, outliers, suspicious dates), allowing clean separation of “must‑fix” issues versus “review” items.

---

## difference between validation and anomaly detection?

| **Validation** | **Anomaly Detection** |
|----------------|----------------------|
| Structural checks | Quality checks |
| Does schema match? | Does data look normal? |
| Are types correct? | Are there outliers? |
| Are required columns present? | Are there duplicates? |
| Is null ratio acceptable? | Are dates suspicious? |
| **Pass/Fail with issues list** | **Findings report with details** |

Example:
- Validation fails: "Column 'date' not parsed as datetime"
- Anomaly detection finds: "Found 1 duplicate rows", "Outlier in 'sales': row 6"

They complement each other in the pipeline.

---

## Pipeline Execution Flow

```
START
  ↓
Plan (optional LLM-generated steps)
  ↓
Load CSV into DataFrame
  ↓
[LOOP START - up to 2 iterations]
  ↓
Repair:
  - Coerce types (dates, numerics)
  - Fill missing values
  - Clamp ranges
  ↓
Validate:
  - Check schema compliance
  - Type correctness
  - Null ratios
  ↓
If validation passes → EXIT LOOP
If still failing and iterations < 2 → CONTINUE LOOP
  ↓
[LOOP END]
  ↓
Anomaly Detection:
  - Find duplicates
  - Detect outliers (z-score > 3)
  - Flag suspicious dates
  ↓
Decide:
  - Check if done (validation passed OR max iterations reached)
  ↓
Summarize:
  - Final shape
  - Columns
  - Validation result
  - Anomaly findings
  - Repair log count
  ↓
END
```

---