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
| plan | Records the run plan (written by the model when `SHDI_LLM=plan`). |
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

## Sources and sinks

| Source URI | Staging and quarantine go to |
|---|---|
| `data.csv` or `csv:path/data.csv` | `<name>__shdi_staging.csv`, `<name>__shdi_quarantine.csv` in `$SHDI_SINK_DIR` |
| `duckdb:///abs/path/file.duckdb#table` or `duckdb:rel/path.duckdb#schema.table` | tables `<table>__shdi_staging`, `<table>__shdi_quarantine` in the same file |
| `postgres:schema.table` | tables `<schema>.<table>__shdi_staging`, `<schema>.<table>__shdi_quarantine` |

The source is read through a read-only connection and is never written. Sinks
refuse any table name without one of the two suffixes. Each staging and
quarantine row carries `shdi_run_id`, `shdi_applied_at` and `shdi_source_row`;
quarantine rows also carry `shdi_proposal_id` and `shdi_reason`, and keep their
raw values as text. Re-applying a run replaces that run's rows instead of
appending duplicates.

Postgres connection strings come only from the environment, never from the URI
or the command line: `SHDI_PG_DSN` reads the source (give it a role with
`SELECT` and nothing else) and `SHDI_PG_SINK_DSN` writes staging and quarantine
(defaults to `SHDI_PG_DSN`). A URI that contains credentials is rejected.
Postgres tables have no insertion order, so the source is read sorted on every
column; `shdi_source_row` is the position in that order, and row keys in
proposals and the audit log always include the primary key.

In-memory DuckDB is not accepted, because a paused run must be able to re-open
the source later. To try DuckDB with the sample:

```bash
python examples/seed_duckdb.py examples/dirty_sales.csv .shdi/demo.duckdb sales
shdi run duckdb:.shdi/demo.duckdb#sales --contract contracts/sales.yaml
```

## Tracing

Set `SHDI_OTEL=1` to trace runs with OpenTelemetry. Spans print to stderr by
default; set `OTEL_EXPORTER_OTLP_ENDPOINT` (and install
`opentelemetry-exporter-otlp-proto-http`) to send them to any OTLP backend.

Each start or resume is one trace: a `shdi.run` or `shdi.resume` root span with
one child span per node (`shdi.plan`, `shdi.load`, ... `shdi.summarize`). Node
spans carry `shdi.run_id`, `shdi.node`, `shdi.proposals.count`,
`shdi.checks.failed` and decision counts. The node that pauses for approval
ends its span with `shdi.interrupted=true`, not an error. `apply` adds a
`shdi.apply.written` event with what it wrote. Model calls get a
`chat <model>` span with `gen_ai.*` attributes.

## The optional LLM

The pipeline runs fully without a model. `SHDI_LLM` turns one on:

| `SHDI_LLM` | What the model does |
|---|---|
| `off` (default) | Nothing. `langchain-gradient` is never imported. |
| `plan` | Writes the run plan shown in the summary. Informational only. |
| `explain` | Writes a one-line plain-English reason for each `needs_approval` proposal, shown by `shdi review`. |

The model never produces values: explanations are stored as text on the
proposal, and `apply` does not read them, so the output is identical with the
model on or off. The prompt includes cell values from the source, so treat an
explanation as a hint written from untrusted data, not as a decision. A model
failure is reported in the summary and the run continues without explanations.

```bash
pip install -r requirements-llm.txt
export DIGITALOCEAN_INFERENCE_KEY=...      # set it in your shell, never in a file in the repo
SHDI_LLM=explain shdi run examples/dirty_sales.csv --contract contracts/sales.yaml
```

`SHDI_LLM_MODEL` picks the Gradient model (default `llama3.3-70b-instruct`).

## MCP server

`shdi-mcp` serves the workflow over MCP (stdio) with six tools:
`profile_source`, `validate`, `propose_repairs`, `list_pending`, `decide` and
`get_audit`. Only `decide` writes, and only to staging and quarantine; a run
started by `propose_repairs` always pauses first. Sources must match
`SHDI_ALLOWED_SOURCES` (nothing is allowed when it is unset), contracts are
names inside `SHDI_CONTRACT_DIR`, and connection strings come from the server's
environment, never from tool arguments. Setup and a client config example:
[docs/mcp.md](docs/mcp.md).

## Benchmark

`make bench` generates 10,000 clean sales rows from a fixed seed, injects 1,050
labelled defects (150 of each type, one per row, duplicates appended as exact
copies), runs the real graph on the dirty DuckDB table with every proposal
auto-approved, and scores staging and quarantine against the clean data. Full
output in [bench/RESULTS.md](bench/RESULTS.md).

| Defect | Injected | Detection precision | Detection recall | Restored | Quarantined | Neutralized | Wrong | Missed | Fix accuracy |
|---|---|---|---|---|---|---|---|---|---|
| type_pollution | 150 | 100.0% | 100.0% | 0 | 150 | 0 | 0 | 0 | 100.0% |
| null | 150 | 100.0% | 100.0% | 0 | 75 | 75 | 0 | 0 | 50.0% |
| duplicate | 150 | 100.0% | 100.0% | 150 | 0 | 0 | 0 | 0 | 100.0% |
| out_of_range | 150 | 100.0% | 100.0% | 0 | 150 | 0 | 0 | 0 | 100.0% |
| unit_error | 150 | 100.0% | 100.0% | 150 | 0 | 0 | 0 | 0 | 100.0% |
| future_date | 150 | 100.0% | 100.0% | 0 | 0 | 150 | 0 | 0 | 0.0% |
| pre_1970_date | 150 | 100.0% | 100.0% | 0 | 0 | 150 | 0 | 0 | 0.0% |
| **all** | 1050 | | | 300 | 375 | 375 | 0 | 0 | 64.3% |

On the 45,500 clean cells of the dirty table nothing was changed, no clean row
was lost and no outlier was reported. The clean data run on its own proposes no
value changes. Fix accuracy counts a defect as handled when the clean value is
back in staging or the row is in quarantine. Out-of-range dates are set to null
because the true date cannot be known, so they count as neutralized, not fixed;
a missing region gets the contract default `Unknown`, which counts the same way.
`tests/test_benchmark_smoke.py` runs the same pipeline on 500 rows and fails if
any of these numbers drop.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests drive the real compiled graph end to end: pausing, resuming from a new
process, partial approval, reject-all byte identity, the audit trail and a
source that changes during the pause. DuckDB tests always run. The Postgres
tests are marked `postgres` and skip unless `SHDI_PG_DSN` is set; to run them
against a throwaway server:

```bash
docker compose -p shdi-test -f docker-compose.test.yml up -d --wait
export SHDI_PG_DSN=postgresql://shdi_reader:reader@127.0.0.1:55433/shdi
export SHDI_PG_SINK_DSN=postgresql://shdi_writer:writer@127.0.0.1:55433/shdi
pytest -m postgres
docker compose -p shdi-test -f docker-compose.test.yml down -v
```
