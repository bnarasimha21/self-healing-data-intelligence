# Self-Healing Data Intelligence

A LangGraph workflow that checks a dataset against a declared **data contract**,
proposes repairs for what fails, **pauses for a human to approve or reject each
proposed change**, then writes the approved result to a staging table and the
rows it could not fix to a quarantine table. The source is never modified.
Every decision and every applied change is recorded in an append-only audit
log, keyed by row and carrying the old and new values.

## How a run flows

```
plan -> load -> validate -> propose -> approve (pause) -> apply -> summarize
```

| Node | What it does |
|---|---|
| plan | Records the static run plan. |
| load | Reads the source as text and fingerprints it (sha256 over the content). |
| validate | Runs every check in the contract, plus review-only outlier detection on `measure` columns. |
| propose | Turns each failing check into proposed changes, each with row keys, old values, new values, a reason and a risk level. Nothing is changed yet. |
| approve | Pauses the graph with `langgraph.types.interrupt()` until every `needs_approval` proposal has a decision. State is checkpointed to SQLite, so the run can be resumed from a different process, hours later. |
| apply | Re-reads the source and refuses to write if its fingerprint changed during the pause. Applies only approved proposals, writes staging and quarantine, records the audit trail, and re-runs every check on the output. |
| summarize | A before/after report: failing checks before and after, cells changed, rows quarantined, duplicates dropped. |

Proposals marked `low` risk (a type cast that changes no text) are approved by
policy if a reviewer has not rejected them by the time the last `needs_approval`
proposal is decided. Rejecting everything produces an output that is
byte-identical to the input.

## The data contract

Rules live in YAML, not in code. `contracts/sales.yaml` describes the bundled sample:

```yaml
apiVersion: shdi/v1
name: sales
primary_key: [id]
max_null_ratio: 0.3
columns:
  id:     {type: integer, nullable: false}
  date:   {type: date, min: "1970-01-01", max: today}
  sales:
    type: number
    nullable: false
    min: 0
    max: 100000
    measure: true          # only measure columns are scanned for outliers
    unit_factors: [1000]   # 400000 -> 400 is proposed as a unit fix, not a clamp
  region: {type: string, nullable: false, default: Unknown,
           accepted_values: [North, South, East, West, Unknown]}
  notes:  {type: string}
```

Each check has a severity (`error` or `warn`) and an `on_violation` action
(`propose_fix`, `quarantine` or `report_only`), with defaults per check kind and
overrides per column or for the whole dataset. Unknown keys are rejected, so a
typo in a contract fails loudly.

What each failure turns into:

| Failure | Proposed change |
|---|---|
| Value that is not the declared type (`three hundred` in a number column) | Quarantine the row. A text value is never silently turned into 0. |
| Number outside `min`/`max` | Clamp to the bound, or rescale when a `unit_factors` entry brings it back in range. |
| Date outside `min`/`max` in a nullable column | Set it to null. |
| Non-ISO date (`01/06/2024`) | Cast to ISO; flagged as ambiguous and always `needs_approval`. |
| Missing value in a non-nullable column | Fill the column's `default`, or quarantine when it has none. |
| Exact duplicate row | Drop the extra copy. |
| Same primary key, different content | Quarantine every row with that key. |

Outliers use the median absolute deviation (modified z-score above 3.5), only on
columns marked `measure`, so an `id` column is never reported. They are
reported for review and never changed automatically.

To start a contract for your own data, draft one and edit it:

```bash
shdi infer-contract my_data.csv -o contracts/my_data.yaml
```

## Running

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e .                 # installs the `shdi` command
python main.py                   # demo: runs the sample with every proposal auto-approved
```

The review loop:

```bash
shdi run examples/dirty_sales.csv --contract contracts/sales.yaml   # prints proposals and pauses
shdi review <run_id>
shdi reject <run_id> --ids p8
shdi approve <run_id> --all
shdi status <run_id>
shdi audit <run_id>
```

For CSV sources the output goes to `$SHDI_SINK_DIR` (default `.shdi/out/`) as
`<name>__shdi_staging.csv` and `<name>__shdi_quarantine.csv`. Checkpoints and
the audit log live in `$SHDI_HOME` (default `.shdi/`).

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests drive the real compiled graph end to end: pausing, resuming from a new
process, partial approval, reject-all byte identity, the audit trail and a
source that changes during the pause.
