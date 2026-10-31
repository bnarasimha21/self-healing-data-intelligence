# MCP server

`shdi-mcp` (or `python -m shdi.mcp_server`) exposes the workflow to any MCP
client (Claude Desktop, Claude Code, Cursor, a Gradient agent) over stdio. The
approval gate and the audit log stay on the server: a client can propose and
decide, but every write still goes through the same paused graph run.

## Tools

| Tool | Writes | What it returns |
|---|---|---|
| `profile_source(uri)` | no | row count; per column the inferred type, null count, distinct count and up to 5 sample values |
| `validate(uri, contract, as_of?)` | no | every check with `passed`, `failing_count` and failing rows and row keys, plus outlier findings |
| `propose_repairs(uri, contract, as_of?)` | no | a paused run: `thread_id` (same as `run_id`), the proposals, `pending` and `needs_approval` ids |
| `list_pending(thread_id)` | no | the same view of an existing run, or its outcome once applied |
| `decide(thread_id, approved_ids, rejected_ids, reviewer)` | staging and quarantine only | the run after the decisions: still `awaiting_approval`, or `completed` with `apply` and `before_after` counts |
| `get_audit(run_id)` | no | every decision and applied change with row keys, old and new values |

Results are structured JSON. Lists of rows, row keys and values are cut to 50
entries per item; `row_count` / `failing_count` give the full size and
`truncated` says when the list was cut.

`propose_repairs` always pauses, even when nothing needs a decision, so no run
writes until a client calls `decide`. `decide` records new decisions and
resumes; the run applies once every `needs_approval` proposal has one.
Low-risk proposals (type casts that change no text) that nobody rejected are
approved by policy at that point, and calling `decide` with no ids on a run
with nothing left to approve applies it. Ids that already have a decision are
ignored and listed under `ignored`, so a repeated call writes no second audit
record and applies nothing twice.

## Configuration (server environment only)

| Variable | Meaning |
|---|---|
| `SHDI_ALLOWED_SOURCES` | Required. Comma-separated `fnmatch` patterns matched against the canonical URI. Nothing is allowed when unset. |
| `SHDI_CONTRACT_DIR` | Folder of contract YAML files (default `./contracts`). Tools take a contract name such as `sales`, never a path. |
| `SHDI_HOME` | Checkpoints and the audit log. Use an absolute path, since clients start the server from varying folders. |
| `SHDI_PG_DSN`, `SHDI_PG_SINK_DSN` | Postgres connections for `postgres:schema.table` sources. Never passed as tool arguments; a URI with credentials is rejected. |
| `SHDI_SINK_DIR` | Where CSV sources get their staging and quarantine files. |

Canonical URIs have resolved absolute paths: `csv:/data/in/sales.csv`,
`duckdb:///data/warehouse.duckdb#sales`, `postgres:public.sales`. Paths are
resolved (including `..` and symlinks) before matching, so
`duckdb:///data/../etc/x.duckdb#t` is checked as `duckdb:///etc/x.duckdb#t`.
In `fnmatch`, `*` also matches `/`, so `csv:/data/*` covers subfolders too.

## Client config

Claude Desktop (`claude_desktop_config.json`) or any client with the same shape:

```json
{
  "mcpServers": {
    "shdi": {
      "command": "/path/to/self-healing-data-intelligence/.venv/bin/shdi-mcp",
      "env": {
        "SHDI_ALLOWED_SOURCES": "duckdb:///data/landing.duckdb#*,postgres:raw.*",
        "SHDI_CONTRACT_DIR": "/path/to/self-healing-data-intelligence/contracts",
        "SHDI_HOME": "/path/to/shdi-state",
        "SHDI_PG_DSN": "postgresql://reader@db.internal/warehouse"
      }
    }
  }
}
```

Claude Code: `claude mcp add shdi --env SHDI_ALLOWED_SOURCES='duckdb:///data/*.duckdb#*' -- /path/to/.venv/bin/shdi-mcp`.

## A typical session

1. `validate("duckdb:///data/landing.duckdb#sales", "sales")`: what fails.
2. `propose_repairs(...)`: returns `thread_id` and proposals; nothing written.
3. The assistant shows the `needs_approval` proposals to the user.
4. `decide(thread_id, approved_ids=["p4", "p7"], rejected_ids=["p5"], reviewer="narsi")`.
5. `get_audit(thread_id)`: who approved what, and what changed.

The test `tests/test_mcp_server.py` runs exactly this over stdio with the SDK's
`ClientSession`.
