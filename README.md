# Self-Healing Data Intelligence

A LangGraph workflow that checks a dataset against a declared **data contract**,
proposes repairs for what fails, **pauses for a human to approve or reject each
proposed change**, then writes the approved result to a staging table and the
rows it could not fix to a quarantine table. The source is never modified.
Every decision and every applied change is recorded in an append-only audit
log, keyed by row and carrying the old and new values. That includes type casts
and added columns, which list every row they touch.

## How a run flows

```
plan -> load -> validate -> propose -> approve (pause) -> apply -> summarize
```

| Node | What it does |
|---|---|
| plan | Records the run plan (written by the model when `SHDI_LLM=plan`). |
| load | Reads the source with its column types and fingerprints it: sha256 over the column types and the database's own text of every cell, rendered in UTC. |
| validate | Runs every check in the contract, plus review-only outlier detection on `measure` columns. |
| propose | Turns each failing check into proposed changes, each with row keys, old values, new values, a reason and a risk level. Nothing is changed yet. |
| approve | Pauses the graph with `langgraph.types.interrupt()` until every `needs_approval` proposal has a decision. State is checkpointed to SQLite, so the run can be resumed from a different process, hours later. |
| apply | Re-reads the source and refuses to write if its fingerprint changed during the pause. Applies only approved proposals, writes staging and quarantine, records the audit trail, and re-runs every check on the output. |
| summarize | A before/after report: failing checks before and after, cells changed, rows quarantined, duplicates dropped. |

Proposals marked `low` risk (a type cast where every value, stored as the
contract type, reads back as exactly its source text) are approved by policy if a reviewer has not rejected them by the time the last `needs_approval`
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

Types are parsed strictly. Dates must be complete (`2024-05` or `May 2024` is
not a date, rather than being filled in from today), timestamps with a time zone
are converted to UTC, integers must fit in 64 bits, and only ASCII digits count
as digits. A date written with an offset (`2024-01-02T01:00:00+05:30`) is its
UTC date, `2024-01-01`. Numbers are compared as exact decimals, never as
binary floats: `5` and `5.0` are the same value, and a DECIMAL(20,4) or
HUGEINT value too long for a double keeps every digit. DuckDB connections run with `TimeZone` set to UTC and Postgres reads
set the session to UTC, so the machine's time zone never changes a result. A contract that could not work is rejected when it loads: `min` or
`max` on a string or boolean column, a fractional bound on an integer column, a
`default` that does not parse or is outside `accepted_values`, and a unit
factor that is zero, negative or 1.

Each check has a severity (`error` or `warn`) and an `on_violation` action
(`propose_fix`, `quarantine` or `report_only`), with defaults per check kind and
overrides per column or for the whole dataset. Unknown keys are rejected, so a
typo in a contract fails loudly.

What each failure turns into:

| Failure | Proposed change |
|---|---|
| Value that is not the declared type (`three hundred` in a number column) | Quarantine the row. A text value is never silently turned into 0. |
| Number outside `min`/`max` | Clamp to the bound, or rescale when a `unit_factors` entry brings it back in range (in an integer column only when the result is a whole number). |
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
the audit log live in `$SHDI_HOME` (default `.shdi/` in the current folder, so
run every command for one run from the same folder, or set `SHDI_HOME` to an
absolute path). The sink folder is stored as an absolute path when the run
starts, so `approve` writes to the same place wherever it runs from.

Exit codes: 0 when the run passed or is waiting for a reviewer, 1 when it
completed with checks still failing (`failed_checks`), when the source changed
during the pause (`stale_source`), or when it was interrupted, and 2 for a usage
error such as an unknown run or a contract that does not load.

`start` and `approve`/`reject` take a per-run lock, so two reviewers deciding
the same run at once apply it once. If applying fails part way (the source file
was moved, the database was down), the run stays `interrupted` with its
decisions saved; fix the cause and run `shdi approve <run_id>` again to retry.
A run id that already has audit records cannot be reused.

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

A run's staging and quarantine rows are written together: in one transaction
for DuckDB and Postgres, and under one file lock for CSV, so two runs writing
the same CSV sink at once keep each other's rows. If the source changed during
the pause, `apply` also deletes anything an earlier failed attempt of the run
wrote, so neither table holds rows from a stale read.

Values keep their source types. DuckDB is read through Arrow, so DECIMAL,
HUGEINT, DATE and TIMESTAMPTZ columns arrive exactly and a DuckDB staging
column keeps the source column's type when every value fits it. A NaN in a
DuckDB DOUBLE column stays NaN; it is not turned into null. Before writing,
`apply` checks every stored cell against the value in its proposal. A clamp to
99.5 in a BIGINT column widens the column rather than storing 99, and the
write is refused if a cell would still differ.

A CSV whose header repeats a column name, or with a row that has more fields
than the header, is rejected with an error instead of being read with columns
shifted.

Postgres connection strings come only from the environment, never from the URI
or the command line: `SHDI_PG_DSN` reads the source (give it a role with
`SELECT` and nothing else) and `SHDI_PG_SINK_DSN` writes staging and quarantine
(defaults to `SHDI_PG_DSN`). A URI that contains credentials is rejected.
Postgres tables have no insertion order, so the source is read sorted on every
column; `shdi_source_row` is the position in that order, and row keys in
proposals and the audit log always include the primary key.

Table names must be plain identifiers, `table` or `schema.table` with ASCII
letters, digits and underscores. DuckDB would otherwise read a file named in the
table position (`#"/etc/passwd"`), so that is refused, and every DuckDB
connection is also opened with file and network access turned off.

When a later run writes a column the sink table cannot hold, the column is
widened instead of failing or rounding, and new columns are added. In DuckDB,
integers and decimals widen to a DECIMAL that holds both (VARCHAR past 38
digits), an integer meeting a float gives DOUBLE, and anything else becomes
VARCHAR. In Postgres, exact decimals are written as `numeric` and timestamps
with a zone as `timestamptz`; an integer column meeting a decimal becomes
`numeric`, one meeting a float becomes `double precision`, and anything else
becomes `text`. Postgres columns outside the contract keep
their values but not always their type: integers, numbers, booleans, dates and
timestamps keep it, while json and array columns land in staging as JSON text and other types
(point, xml) as their text form.

Two CSV sources with the same file name in different folders share a sink file
name. Give them different `--sink-dir` folders.

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

`make bench` generates 10,000 clean sales rows from a fixed seed, injects
labelled defects (150 of each type, one per row, duplicates appended as exact
copies), runs the real graph on the dirty DuckDB table with every proposal
auto-approved, and scores staging and quarantine against the clean data. It
runs two scenarios: `text`, where every column is VARCHAR, and `typed`, where
the table has BIGINT, DATE, DECIMAL(10,2) and TIMESTAMPTZ columns (it has no
type pollution, since a typed column cannot hold the word `three hundred`).
Full output in [bench/RESULTS.md](bench/RESULTS.md).

Each defect ends in exactly one outcome, and only one of them means fixed:

* **Restored**: the clean value is back in staging.
* **Quarantined**: the row was held back for a person to fix.
* **Neutralized**: the bad value became null or the column default, so the
  real value is gone.
* **Wrong**: staging holds a value that is neither clean nor neutral.
* **Missed**: the defect is still in staging untouched.

Three defect types are deliberately outside the contract: a x10 slip that stays
inside the allowed range, a wrong but plausible value, and a date with day and
month swapped. They measure what a contract cannot see.

| Defect | In contract | Detection recall | Restored | Quarantined | Neutralized | Missed |
|---|---|---|---|---|---|---|
| type_pollution (text only) | yes | 100% | 0 | 150 | 0 | 0 |
| null | yes | 100% | 0 | 75 | 75 | 0 |
| duplicate | yes | 100% | 150 | 0 | 0 | 0 |
| out_of_range | yes | 100% | 0 | 150 | 0 | 0 |
| unit_error | yes | 100% | 150 | 0 | 0 | 0 |
| future_date | yes | 100% | 0 | 0 | 150 | 0 |
| pre_1970_date | yes | 100% | 0 | 0 | 150 | 0 |
| x10_error | no | 100% | 0 | 0 | 0 | 150 |
| in_range_wrong | no | 0% | 0 | 0 | 0 | 150 |
| day_month_swap | no | 0% | 0 | 0 | 0 | 150 |

The counts are the same in both scenarios. Restored rate:

| Scenario | Defects | Restored | Quarantined | Neutralized | Missed | Restored rate (all) | Restored rate (in contract) |
|---|---|---|---|---|---|---|---|
| text | 1500 | 300 | 375 | 375 | 450 | 20.0% | 28.6% |
| typed | 1350 | 300 | 225 | 375 | 450 | 22.2% | 33.3% |

Only duplicates and unit errors come back with their true value. Everything
else the contract describes is contained (held for review or blanked) but not
repaired, and the three undescribed types pass straight through. The outlier
rule flags every x10 slip for review but changes nothing, so those count as
missed. Detection precision is 100% for every described type; it is not
defined for the undescribed ones, where recall counts any failing check or
outlier finding on the row.

In both scenarios no clean cell was changed, no clean row was lost, no outlier
was reported on a clean row, and the repaired output passes every check. The
clean data run on its own proposes no value changes and stages every cell
equal to the clean value. In the typed scenario staging keeps the BIGINT, DATE,
DECIMAL(10,2) and TIMESTAMPTZ types, and a rescaled amount is the exact decimal.
`tests/test_benchmark_smoke.py` runs both scenarios on 500 rows and fails if any
outcome share moves.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests drive the real compiled graph end to end: pausing, resuming from a new
process, partial approval, reject-all byte identity, the audit trail and a
source that changes during the pause. `tests/test_review_fixes.py` covers exact
values through typed DuckDB and Postgres columns, transactional writes, the CSV
lock with concurrent writers, and results under three process time zones. DuckDB tests always run. The Postgres
tests are marked `postgres` and skip unless `SHDI_PG_DSN` is set; to run them
against a throwaway server:

```bash
docker compose -p shdi-test -f docker-compose.test.yml up -d --wait
export SHDI_PG_DSN=postgresql://shdi_reader:reader@127.0.0.1:55433/shdi
export SHDI_PG_SINK_DSN=postgresql://shdi_writer:writer@127.0.0.1:55433/shdi
pytest -m postgres
docker compose -p shdi-test -f docker-compose.test.yml down -v
```
