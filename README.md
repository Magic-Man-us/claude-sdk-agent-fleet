# agent-fleet

Assembles a minimal Claude Agent SDK agent from a problem statement and a capability corpus, then
runs and resumes it. The generation path is deterministic — no LLM and no network in it, so the
same request and corpus produce a byte-identical agent. Running the agent is the part that talks to
the SDK.

## Repository layout

Python only. Three workspace members, layered so each depends on the ones below it.

| Path | What it is | Run or import |
|---|---|---|
| `src/agent-fleet/` | `agent_fleet` — core engine: pipeline, router, pool | imported |
| `src/agent-fleet-api/` | `agent_fleet_api` — FastAPI service over the core | run |
| `src/agent-fleet-mcp/` | `agent_fleet_mcp` — MCP server exposing the pool as tools (`pool-mcp`) | run |

```
agent_fleet_api ─┐
                 ├─imports──▶ agent_fleet ──imports──▶ capdisc
agent_fleet_mcp ─┘
```

The two front-ends carry the web/MCP dependencies so the engine carries none. Environment scanning
lives in [capdisc](https://github.com/Magic-Man-us/capability-discovery), a
separate repo consumed as a pinned git dependency. That repo is currently private — building this
workspace requires access to it until it's published separately.

## Pipeline

```
ProblemRequest ─▶ recall ─▶ select ─▶ compose ─▶ score ─▶ render
                  (source)  (budget)  (AgentSpec) (efficiency) (SDK program)
```

- **recall** — a `CatalogSource` ranks the corpus by lexical relevance (a pluggable `Ranker`) and trims to a limit.
- **select** — keep candidates above a relevance threshold (plus pinned), capped by tool/skill budgets.
- **compose** — map the selected refs into an `AgentSpec` with a templated system prompt.
- **score** — check the spec against tool/skill/prompt budgets (`efficiency`).
- **render** — emit a runnable Claude Agent SDK program.

## Pool

`AgentPool` (SQLite) keys each pooled agent by a stable `AgentKey` and stores the `AgentSpec` and
session id that built it, so a run can be retrieved, resumed against the same live SDK
conversation, or found fuzzily. `run_with_capture` observes the live message stream to record the
real, resumable session id of every agent a run involves — the top-level agent and each dispatched
subagent. Runs, per-agent runs, and findings are persisted alongside the entry.

## Develop

```
uv run pytest
uv run ruff check
uv run mypy src/agent-fleet/src src/agent-fleet-api/src src/agent-fleet-mcp/src
```

Details: [docs/OVERVIEW.md](docs/OVERVIEW.md) · [docs/pipeline.md](docs/pipeline.md) ·
[docs/catalog-boundary.md](docs/catalog-boundary.md)
