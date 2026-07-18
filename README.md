# agent-fleet

[![PyPI](https://img.shields.io/pypi/v/claude-sdk-agent-fleet)](https://pypi.org/project/claude-sdk-agent-fleet/)
[![Python versions](https://img.shields.io/pypi/pyversions/claude-sdk-agent-fleet)](https://pypi.org/project/claude-sdk-agent-fleet/)
[![License](https://img.shields.io/pypi/l/claude-sdk-agent-fleet)](LICENSE)
[![CI](https://github.com/Magic-Man-us/claude-sdk-agent-fleet/actions/workflows/ci.yml/badge.svg)](https://github.com/Magic-Man-us/claude-sdk-agent-fleet/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/Magic-Man-us/claude-sdk-agent-fleet/graph/badge.svg)](https://codecov.io/gh/Magic-Man-us/claude-sdk-agent-fleet)

Assembles a minimal Claude Agent SDK agent from a problem statement and a capability corpus, then
runs and resumes it. The generation path is deterministic — no LLM and no network in it, so the
same request and corpus produce a byte-identical agent. Running the agent is the part that talks to
the SDK.

## Repository layout

Python only. One PyPI package (`claude-sdk-agent-fleet`), three subpackages layered so each depends on the ones
below it. `pip install claude-sdk-agent-fleet` gets the core engine and MCP pool server;
`claude-sdk-agent-fleet[api]` (or `[all]`) adds the FastAPI service.

| Path | What it is | Run or import | Install |
|---|---|---|---|
| `src/agent_fleet/` | core engine: pipeline, router, pool | imported | `claude-sdk-agent-fleet` |
| `src/agent_fleet_api/` | FastAPI service over the core | run | `claude-sdk-agent-fleet[api]` |
| `src/agent_fleet_mcp/` | MCP server exposing the pool as tools (`pool-mcp`) | run | `claude-sdk-agent-fleet` |

```
agent_fleet_api ─┐
                 ├─imports──▶ agent_fleet ──imports──▶ capdisc
agent_fleet_mcp ─┘
```

The API front-end carries the web dependencies as an optional extra so the core engine carries
none by default. Environment scanning lives in
[capdisc](https://github.com/Magic-Man-us/capability-discovery), a separate, public repo consumed
as a pinned git dependency.

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
uv sync --extra api
uv run pytest
uv run ruff check
uv run mypy src
make coverage       # test coverage, printed to the terminal
```

Details: [docs/OVERVIEW.md](docs/OVERVIEW.md) · [docs/pipeline.md](docs/pipeline.md) ·
[docs/catalog-boundary.md](docs/catalog-boundary.md)
