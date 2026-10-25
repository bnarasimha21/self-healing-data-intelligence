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
- Anomaly detection finds: "Found 1 duplicate rows", "High values in 'sales' above P99: 1 rows"

They complement each other in the pipeline.

---

## Pipeline Execution Flow

```
START
  ↓
Plan (LLM bullet list if DIGITALOCEAN_INFERENCE_KEY works, otherwise a static list; informational only)
  ↓
Load CSV into DataFrame
  ↓
Validate (first pass, on the raw data)
  ↓
Decide:
  - validation passed        → go to Anomaly Detection
  - repair passes used = 2   → go to Anomaly Detection (issues are reported, not hidden)
  - otherwise                → Repair
  ↓
Repair (then back to Validate, then Decide again):
  - Add missing required columns (id, date, sales; see REQUIRED_COLUMNS in main.py)
  - Parse date columns, coerce numeric columns
  - Fill missing values (0 for numbers, "" for text)
  - Clamp numeric columns to their 1st to 99th percentile
  ↓
Anomaly Detection (once, on the final data):
  - Duplicate rows
  - Outliers: |z-score| > 3, plus values above the 99th percentile
  - Suspicious dates (before 1970 or in the future)
  ↓
Summarize:
  - Node messages, final shape, columns
  - Validation result and remaining issues
  - Anomaly findings
  - Repair log count
  ↓
END
```

### Behaviors worth knowing

- **Clamping changes data.** Repair clips each numeric column to its [P1, P99] range, which silently overwrites extreme values. Because anomaly detection runs after repair, it scans the clamped data, so the z-score check rarely fires on heavy outliers; the "above P99" check is the one that usually catches them. Clamping only applies to columns that were already numeric before coercion and have at least 5 values, so a text-polluted column (like `sales` with "three hundred") is clamped on the second repair pass, not the first.
- **Required columns** are the fixed list `REQUIRED_COLUMNS = ["id", "date", "sales"]` in `main.py`. Edit it for your own data. A missing one fails validation, and repair adds it as NaN and then fills it.
- **The LLM never touches the data.** It only writes the plan text.
- A CSV that already passes validation skips repair entirely.

## Running

```bash
pip install -r requirements.txt
export DIGITALOCEAN_INFERENCE_KEY=...   # optional; without it the static plan is used
python main.py
```

`python main.py` runs the built-in dirty sample CSV and prints the summary.
