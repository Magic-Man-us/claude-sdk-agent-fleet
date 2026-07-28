# agent-fleet

[![PyPI](https://img.shields.io/pypi/v/claude-sdk-agent-fleet)](https://pypi.org/project/claude-sdk-agent-fleet/)
[![Python versions](https://img.shields.io/pypi/pyversions/claude-sdk-agent-fleet)](https://pypi.org/project/claude-sdk-agent-fleet/)
[![License](https://img.shields.io/pypi/l/claude-sdk-agent-fleet)](LICENSE)
[![CI](https://github.com/Magic-Man-us/claude-sdk-agent-fleet/actions/workflows/publish.yml/badge.svg)](https://github.com/Magic-Man-us/claude-sdk-agent-fleet/actions/workflows/publish.yml)
[![codecov](https://codecov.io/gh/Magic-Man-us/claude-sdk-agent-fleet/graph/badge.svg)](https://codecov.io/gh/Magic-Man-us/claude-sdk-agent-fleet)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

Assembles a minimal Claude Agent SDK agent from a problem statement and a capability corpus, then
runs and resumes it. The generation path is deterministic — no LLM and no network in it, so the
same request and corpus produce a byte-identical agent. Running the agent is the part that talks to
the SDK.

The runtime remains Claude Agent SDK-specific: Claude is the supervisor and conversation owner.
Live and generated Claude agents also receive one guarded in-process MCP tool,
`mcp__codex__codex_run`, for delegating bounded implementation, debugging, and independent-review
work to the local Codex CLI. Codex is a worker capability, not a second fleet runtime or an
authoritative workflow store.

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

## Teammates

The pool's primary MCP surface: a hardcoded roster of templated teammates
(`src/agent_fleet/engine/teammates.py`), addressed by name. `spawn_teammate` stands one up from
its template (toolkits can pin its capabilities) and runs it in the background; `check_teammate`
reads the derived status and persisted outcome; `message_teammate` revives the standing session.
The pool key is `teammate.{name}`, so the same name always resumes the same conversation.

Consumption is push-style, not poll-in-conversation: set a harness Monitor on `check_teammate`
rather than re-checking it from inside the conversation. `AGENT_FLEET_NOTIFY_COMMAND` configures a
Stop-hook shell command that runs when a teammate's run finishes, receiving Claude Code's
hook-input JSON on stdin — the payload identifies the finished session.

## Guarded Codex worker

Install and authenticate the Codex CLI once:

```bash
codex --version
codex login
```

Every Claude fleet run—including named subagents and emitted agent programs—gets `codex_run` by
default. The tool accepts a self-contained prompt, an absolute canonical Git worktree root, a
`read-only` or `workspace-write` sandbox, a supported Codex model, reasoning effort,
timeout, and an optional prior thread UUID. It returns the final message, thread UUID, commands
observed in Codex JSONL, token usage, duration, and bounded failure detail.

Fresh prompts travel over stdin. Codex's documented non-interactive resume form requires a
positional prompt, so resumed prompts are capped at 64,000 UTF-8 bytes for macOS process-argument
portability; avoid putting secrets in a resume prompt because local process listings can expose
command arguments.

The secure defaults are intentional:

- `cwd` must stay under an absolute, existing operator-allowlisted directory. An empty root list
  uses the fleet process working directory, except when that directory is the filesystem root or
  user home; set an explicit narrower allowlist in those cases.
- `read-only` is the default and Codex receives `--ask-for-approval never`.
- `workspace-write` is disabled until explicitly enabled, and then accepts only a clean, detached
  linked Git worktree.
- the wrapper never uses `--skip-git-repo-check`, never invokes a shell, ignores user Codex config
  for the automated run, explicitly treats the worktree as untrusted so project `.codex/` config,
  hooks, rules, and MCP servers do not load, and pins adjacent controls: no command network, web
  search, extra writable or temporary roots, hooks, apps, remote plugins, nested agents, login
  shells, or broad command-environment inheritance.
- runtime/output are bounded, the whole child process group is terminated on macOS and Linux, and
  common API/service credentials are removed from the Codex process environment.
- the pool MCP does not expose a direct Codex endpoint; the running Claude agent is the caller.

Configure the boundary with environment variables:

```bash
AGENT_FLEET_CODEX_ENABLED=true
AGENT_FLEET_CODEX_ALLOWED_ROOTS='["/absolute/path/to/repos","/absolute/path/to/worktrees"]'
AGENT_FLEET_CODEX_ALLOW_WORKSPACE_WRITE=false
AGENT_FLEET_CODEX_MAX_TIMEOUT_SECONDS=3600
```

For implementation workers, set `AGENT_FLEET_CODEX_ALLOW_WORKSPACE_WRITE=true`, create a detached
linked worktree under an allowed root, and pass that exact worktree root to `codex_run`. Disabling
`AGENT_FLEET_CODEX_ENABLED` removes the server and tool grant entirely.

## Develop

```
uv sync --extra api
uv run pytest
uv run ruff check
uv run mypy src
make coverage       # test coverage, printed to the terminal
```

Details: [docs/OVERVIEW.md](docs/OVERVIEW.md) · [docs/pipeline.md](docs/pipeline.md) ·
[docs/catalog-boundary.md](docs/catalog-boundary.md) · [docs/live-smoke.md](docs/live-smoke.md)

## Coverage

[![codecov sunburst](https://codecov.io/gh/Magic-Man-us/claude-sdk-agent-fleet/graphs/sunburst.svg)](https://codecov.io/gh/Magic-Man-us/claude-sdk-agent-fleet)
