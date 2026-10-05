# IT Incident Response Agent (AcmeCloud simulation)

A simulated AI SRE agent. A user reports an incident; the agent investigates a
fake production environment through tools, diagnoses the cause, asks a human
before dangerous remediation, executes, verifies recovery, and records the
workflow. The design is described in `stuff.md`.

## Status

| Milestone | Scope | State |
| --- | --- | --- |
| 1 | Fake infrastructure + query functions + tests (no LLM) | **done** |
| 2 | Gemini reasoning loop over the tools | **done** |
| 3 | LangGraph state machine | **done** |
| 4 | Tools served over MCP, agent as MCP client | **done** |
| 5 | Pre-tool and post-tool hooks | **done** |
| 6 | Human approval interrupt for high-risk tools | **done** |
| 7 | Compaction and SQLite checkpointing | **done** |
| 8 | Langfuse observability of the agent lifecycle | **done** |
| 9 | LLM-as-judge Error Recovery on 10 traces | **done** |
| 10 | FastAPI and Docker | **done** |

## Layout

```text
data/                    JSON seed files (generated, committed)
scripts/generate_seed_data.py   deterministic generator for data/
mcp_server/data_store.py        loads data/ into SQLite; all reads/writes go through it
mcp_server/server.py            MCP server (stdio) over the tool functions
mcp_server/tools.py             the 17 tools + ToolError + execute router
app/graph.py                    LangGraph: discover, compact, reason, guard, approve, execute
app/hooks.py                    deterministic pre/post tool hooks
app/compaction.py               transcript summary past 10 messages
app/checkpoint.py               SQLite graph checkpoints
app/telemetry.py                Langfuse v4 observations (agent, generation, tool, retriever, guardrail)
app/api.py                      FastAPI: /health, /webhook, /runs/{id}, /approval
app/reports.py                  opens an INC-NNN record for a natural-language report
evaluation/traces.json          10 historical traces for Error Recovery scoring
evaluation/judge.py             Gemini LLM-as-judge for Error Recovery (1-5)
evaluation/scenarios.json       hidden ground truth per incident (never exposed via tools)
tests/                   seed, tools, scripted graph, hooks, compaction, MCP stdio, telemetry
```

## Setup

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt     # Windows
# source .venv/bin/activate && pip install -r requirements.txt   # macOS/Linux
.venv/Scripts/python -m pytest
```

Regenerate seed data after editing the scenario design:

```bash
.venv/Scripts/python scripts/generate_seed_data.py
```

## The simulated environment

Simulated clock: `2026-09-30T10:20:00Z`. Ten services, seven databases,
twelve deployments, nine alerts, 196 log lines and 480 per-minute metric
samples (09:30-10:20).

| Incident | Service | Scenario | Hidden truth |
| --- | --- | --- | --- |
| INC-001 | orders-api | Bad deployment | `deploy-8472` pointed the DB client at PgBouncer port 6432, which doesn't exist |
| INC-002 | payments-api | DB pool exhausted | v4.2.0 retry wrapper leaks connections |
| INC-003 | users-api | Redis down | `redis-cache` OOM-killed, corrupt AOF; users-api is only a victim |
| INC-004 | api-gateway | False alarm | 3-minute traffic burst, already recovered |
| INC-005 | auth-service | Stuck rollout | new pods miss `JWT_SIGNING_KEY_V2`, readiness fails |
| INC-006 | notification-service | Tool-error recovery | infra healthy; agent must fix a bad `search_logs` time range |
| INC-007 | orders-api | Historical, resolved | approved rollback fixed a bad release |
| INC-008 | payments-api | Historical, resolved | restart was rejected; cause was an expired cert |
| INC-009 | inventory-api | Remediation fails | rollback of `deploy-8501` fails (image pruned); feature flag fixes it |
| INC-010 | search-api | Contradictory signals | logs say DB timeout, DB is healthy; config points at a dead host |

## Tools

```text
read-only : get_incident, list_incidents, get_service_health, get_recent_deployments,
            get_deployment, search_logs, get_metrics, get_database_status, get_alerts,
            get_dependencies
low-risk  : add_incident_note, create_incident, update_incident
high-risk : restart_service, rollback_deployment, scale_service, disable_feature
```

Every tool returns a dict. Bad input and simulated failures raise `ToolError`
with an HTTP-like status; the router turns them into
`{"status": 400, "error": "Invalid time range: ..."}` so the model always
receives a structured, recoverable error.

```python
from mcp_server.tools import ToolKit

tk = ToolKit()                                   # fresh in-memory copy of the seed data
tk.execute("get_service_health", {"service": "orders-api"})
# {'status': 200, 'result': {'status': 'degraded', 'replicas': 4, 'healthy_replicas': 2, 'error_rate': 0.34, ...}}

tk.execute("search_logs", {"start_time": "10 minutes ago"})
# {'status': 400, 'error': "Invalid time range: 'start_time' is not a valid ISO 8601 timestamp ...", ...}
```

High-risk tools do not fix anything by magic. Each service carries a hidden
`simulation` block in `data/services.json` that decides which action actually
heals it. Restarting `orders-api` leaves it degraded; rolling back
`deploy-8472` heals it, writes a fresh baseline metric sample, resolves its
alerts and records an audit event. Remediations advance the simulated clock
by one minute so a follow-up `get_service_health` sees new data.

`DataStore(db_path="incident.db")` gives a file-backed store that persists
mutations across processes; the default `:memory:` store is reseeded per
instance, which is what the tests use.

## Run the agent

Put your key in `.env`:

```text
GEMINI_API_KEY=your_key_here
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://us.cloud.langfuse.com
```

Optional: `GEMINI_MODEL` (default `gemini-3.5-flash-lite`), `GEMINI_SUMMARY_MODEL`
(defaults to the same model; used only for compaction), `GEMINI_THINKING`
(`minimal`, `low`, `medium`, `high`; default `low`), `AGENT_MAX_STEPS` (default 12).
Langfuse is optional: without the two keys the agent still runs, and tests stay
offline. With keys, each `graph.invoke` is one `investigate-incident` (or
`resume-after-approval`) agent trace; `session_id` is the checkpoint thread id
so a HITL pause and resume stay in the same session. Thought signatures are
dropped. Observation names are stable (`propose-next-action`, `search_logs`,
`check-tool-policy`, `await-human-approval`) so dashboards can filter on them.

Send a scripted INC-006 tree without calling Gemini:

```bash
.venv/Scripts/python -m evaluation.send_sample_trace
```

```bash
.venv/Scripts/python -m app.main --incident INC-001
.venv/Scripts/python -m app.main --resume --thread <id>
```

The process prints a thread id. Checkpoints go to `var/checkpoints.sqlite`.
The simulated world for that thread is `var/runs/<id>.db`, so a restarted
process sees the same services and the same paused approval.

```text
discover tools (MCP tools/list)
        ↓
compact if the transcript is longer than 10 messages
        ↓
Gemini proposes a tool call
        ↓
pre-hook
   ├── read-only or low-risk → MCP tools/call
   ├── delete*               → 403, never called
   └── high-risk             → interrupt, wait for approve/reject
                                      ↓
                                 MCP tools/call or a rejection error
        ↓
post-hook writes an audit record (outcome, latency)
        ↓
Gemini again
```

Compaction and checkpointing do different jobs:

```text
Compaction    = controls how much transcript is sent to the model.
Checkpointing = controls whether the workflow survives a process restart.
```

A report passed with `--message` and no `--incident` is stored first. The next
id comes from `var/incident_seq.sqlite` (or `INCIDENT_SEQ_FILE`), so each
report gets its own `INC-NNN` even though every run database starts from the
same seed. The process prints that id, and the model is told to read it with
`get_incident`.

A tool error is an ordinary result. The model sees the status and can retry
with different arguments. If a human rejects an action, the same call is
refused afterwards instead of asking again. `--once` exits at the first answer
or at an approval pause, and prints the command that resumes it.

## HTTP API

```bash
.venv/Scripts/python -m uvicorn app.api:app --host 0.0.0.0 --port 8000
```

`POST /webhook` returns immediately with a `run_id` and an `incident_id`. The graph runs in the background. Poll `GET /runs/{run_id}` until `status` is `waiting_for_approval` or `final`. `POST /approval` resumes the same checkpoint.

```bash
curl -s -X POST localhost:8000/webhook -H "content-type: application/json" -d "{\"message\":\"orders-api is returning HTTP 500 errors.\"}"
curl -s localhost:8000/runs/<run_id>
curl -s -X POST localhost:8000/approval -H "content-type: application/json" -d "{\"run_id\":\"<run_id>\",\"decision\":\"approve\"}"
```

A message with no `incident_id` opens a new incident first (`INC-011`, then `INC-012`, and so on). Passing `"incident_id": "INC-001"` investigates that seeded ticket instead.

`GET /health` is the process check. Checkpoints default to `var/checkpoints.sqlite` and run databases to `var/runs/`. Override them with `CHECKPOINT_PATH` and `RUNS_DIR` when the disk must outlive a container, and keep a single worker so SQLite is not shared across processes.

The `Dockerfile` starts the same app. Render sets `PORT`.

## Error Recovery evaluation

`evaluation/traces.json` holds ten historical investigations (not live Langfuse
exports). Gemini grades only Error Recovery on the assignment 1–5 rubric:

```bash
.venv/Scripts/python -m evaluation.judge
```

That writes `evaluation/scores.json` and prints the average Error Recovery
score. The latest run scored **4.2 / 5** across the ten traces (eight clean
recoveries at 5; two failure cases at 1). The judge is a separate model call
from the investigator and is itself traced as an `evaluate-error-recovery`
evaluator observation when Langfuse keys are set.
