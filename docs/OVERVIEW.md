# agent-fleet — Codebase Overview

Assembles a minimal Claude agent from a problem statement and a capability corpus, then runs and
resumes it. The **generation** path is deterministic and offline: the same request and corpus
produce byte-identical output, enforced by tests (stable sort keys, frozen models, pure functions,
no clock/random). The **run** path is where the SDK, the network, and the LLM enter.

## Three faces

| Face | Entry point | What it does |
|------|-------------|--------------|
| Generator pipeline | `assemble()` → `generate()` in [src/agent_fleet/engine/pipeline.py](../src/agent-fleet/src/agent_fleet/engine/pipeline.py) | problem → `AgentSpec` → runnable SDK program |
| Capability router | `CapabilityRouter` in [src/agent_fleet/router/capability.py](../src/agent-fleet/src/agent_fleet/router/capability.py), served over MCP by [router/mcp_server.py](../src/agent-fleet/src/agent_fleet/router/mcp_server.py) | `find_skills`/`find_tools`/`find_mcp`/`load_skill` as deferred tools |
| Agent pool | `AgentPool` in [src/agent_fleet/engine/pool.py](../src/agent-fleet/src/agent_fleet/engine/pool.py), served over MCP by [agent_fleet_mcp/pool_server.py](../src/agent-fleet-mcp/src/agent_fleet_mcp/pool_server.py) | persist, resume, and fan out named agent sessions |

The pipeline builds an agent; the router is how an agent finds capability at its own runtime; the
pool is how a run survives past one conversation.

## Packages

Three Python workspace members plus one external dependency:

| Package | Holds |
|---------|-------|
| `agent_fleet` (`src/agent-fleet/`) | the pipeline, the router, the pool, and the agent domain models |
| `agent_fleet_api` (`src/agent-fleet-api/`) | FastAPI service — catalog, generate, render, orchestrate, pool routes |
| `agent_fleet_mcp` (`src/agent-fleet-mcp/`) | MCP server exposing the pool as tools (`pool-mcp` console script) |
| `capdisc` ([separate repo](https://github.com/Magic-Man-us/capability-discovery)) | environment scanning — skills, tools, plugins, MCP servers, hooks — into typed catalogs and a report; also owns the `Catalog`/`CatalogEntry` models and the shared Pydantic presets |

`capdisc` is consumed as a **pinned git dependency** (`{ git = "...", rev = "..." }`
in the root `pyproject.toml`). That repo is currently private, so building this workspace requires
access to it until it's published separately.

The context explorer is a separate project
([Magic-Man-us/cc-session-explorer](https://github.com/Magic-Man-us/cc-session-explorer)) with its
own app, API, and MCP server. It has no dependency relationship with agent-fleet in either
direction.

There is no frontend. The web layer was removed; the API still serves an OpenAPI schema, but ships
no first-party client.

What does and does not belong in the catalog is fixed by
[docs/catalog-boundary.md](catalog-boundary.md): the build-time surface is kept separate from the
run-time surface of any agent it emits, and every `Ref` must map to a `find_`/`load_` entry point.

## Pipeline

```
ProblemRequest ─▶ recall ──▶ select ───▶ compose ──▶ score ─────▶ generate
                  (source)   (budget)    (AgentSpec)  (efficiency)  (render)
```

`assemble()` runs recall→select→compose→score into an `AssemblyResult{spec, selection,
efficiency}`. `generate(spec)` is a separate dispatch that renders the spec to a target's native
format. Full diagram: [docs/pipeline.md](pipeline.md).

- recall — [engine/source.py](../src/agent-fleet/src/agent_fleet/engine/source.py): rank every entry by a pluggable `Ranker`, sort by `(-score, id)`, trim to `limit`. `BM25Ranker` (Okapi BM25, normalized to [0,1]) is the default; `TwoStageRanker` narrows by domain/tags first, then reranks.
- select — [engine/select.py](../src/agent-fleet/src/agent_fleet/engine/select.py): drop below `RELEVANCE_THRESHOLD` (unless pinned), equip skills and MCP servers into de-duplicated buckets via `singledispatch`. Tools aren't recalled — every agent gets the fixed `DEFAULT_TOOLS`.
- compose — [engine/compose.py](../src/agent-fleet/src/agent_fleet/engine/compose.py): map the selection into an `AgentSpec`, generating an XML-tagged system prompt unless the caller supplied one.
- score — [engine/efficiency.py](../src/agent-fleet/src/agent_fleet/engine/efficiency.py): three pass/fail dimensions (tool_count, skill_count, prompt_size) against budgets. `EfficiencyReport.passed` is a `computed_field`.
- render — [engine/render.py](../src/agent-fleet/src/agent_fleet/engine/render.py): `render_claude_sdk` → a runnable `claude_agent_sdk` program; `to_options` / `to_agent_definition` / `with_subagents` build the live SDK objects.

## File map

`models/agent/` — the agent domain ([src/agent_fleet/models/agent/](../src/agent-fleet/src/agent_fleet/models/agent/))
- `types.py` — the domain-primitive aliases, the `ModelId`/`AgentEffort` `StrEnum`s, and the default constants (read this first).
- `request.py` — `ProblemRequest` (flat: task · name · domain · tags · team · model · pinned · system_prompt).
- `spec.py` — `AgentSpec` + `SubagentFrontmatter`. `prompt.py` — the templated prompt. `thinking.py` — the `ThinkingConfig` union.
- `pool.py` — the pool's records: `PoolEntry`, `RunRecord`, `AgentRunRecord`, `Finding`, `RunOutcome`.

`engine/` — assembly, execution, and the tools an agent is handed
- [pipeline.py](../src/agent-fleet/src/agent_fleet/engine/pipeline.py) — `assemble()` and `generate()`.
- [source.py](../src/agent-fleet/src/agent_fleet/engine/source.py) · [select.py](../src/agent-fleet/src/agent_fleet/engine/select.py) · [compose.py](../src/agent-fleet/src/agent_fleet/engine/compose.py) · [efficiency.py](../src/agent-fleet/src/agent_fleet/engine/efficiency.py) · [render.py](../src/agent-fleet/src/agent_fleet/engine/render.py) — the five pipeline stages above.
- [pool.py](../src/agent-fleet/src/agent_fleet/engine/pool.py) — `AgentPool` (SQLite) and `AsyncAgentPool` (an `asyncio.to_thread` wrapper for async consumers).
- [dispatch.py](../src/agent-fleet/src/agent_fleet/engine/dispatch.py) — `run_with_capture()`: run an agent live and record the real, resumable session id of the top-level agent *and* every subagent it dispatches, by observing the streamed messages.
- [run.py](../src/agent-fleet/src/agent_fleet/engine/run.py) — `run_agent()`: execute a spec in-process via the SDK's `query()`.
- [orchestrate.py](../src/agent-fleet/src/agent_fleet/engine/orchestrate.py) — `Orchestrator`: reviews the ranked slates, proposes an `AgentSpec`, and spawns it. Review/propose are synchronous and testable without the SDK; only spawn calls into it.
- [acquire_tool.py](../src/agent-fleet/src/agent_fleet/engine/acquire_tool.py) — the `acquire_capability` MCP tool: an agent finds, equips, and runs a capability on demand, mid-run.
- [findings_tool.py](../src/agent-fleet/src/agent_fleet/engine/findings_tool.py) — the `write_finding` MCP tool: a concurrency-safe shared sink for fan-out lens agents.
- [emit.py](../src/agent-fleet/src/agent_fleet/engine/emit.py) — `write_agent()`: source → file on disk. [naming.py](../src/agent-fleet/src/agent_fleet/engine/naming.py) — `slugify_name()`.

`router/` — the deferred-tool capability product
- [capability.py](../src/agent-fleet/src/agent_fleet/router/capability.py) — `CapabilityRouter` + `SkillCard`/`ToolCard`/`McpCard`/`PluginCard`: index the environment once, answer `find_*` with a top-k BM25 slate, load a skill body on demand.
- [mcp_server.py](../src/agent-fleet/src/agent_fleet/router/mcp_server.py) — `FastMCP("skill-router")`; `main()` is the `skill-router` console script.

Also: [settings.py](../src/agent-fleet/src/agent_fleet/settings.py) — `AgentFleetSettings(DiscoverySettings)`, adding the agent/skill directories. [main.py](../src/agent-fleet/src/agent_fleet/main.py) — the CLI.

Environment scanning (`scan_environment`, `BUILTIN_TOOLS`, `parse_mcp_servers`), the `Catalog`
models, and the Pydantic presets (`FrozenModel`, `InputModel`) all live in `capdisc`
and are re-exported from `agent_fleet.__init__`.

## Type system and patterns

Pydantic v2 discipline throughout:
- Domain primitives — no bare `str`/`int` in any model or signature; each is a named `Annotated` alias in a `types.py` carrying constraint + `title`/`description`/`examples`.
- Discriminated union for catalog entries (`Field(discriminator="kind")`), dispatched on parse.
- Centralized configs (`FrozenModel`/`InputModel` from `capdisc.base`); no inline `ConfigDict`.
- Boundary vs interior — `InputModel` only at filesystem/CLI/MCP edges; internals are frozen and strict.
- `computed_field` for derived output (`EfficiencyReport.passed`).
- Polymorphic dispatch via `singledispatch` (`select._equip`) and structural `match`; not dict-poking.
- Protocols as swap seams — `CatalogSource`, `Ranker`.

## Swap seams

1. `CatalogSource` — `InMemoryCatalogSource` → a real index, same `recall()`.
2. The ranker — `BM25Ranker` (Okapi BM25) → semantic. `TwoStageRanker` is the current intermediate.
3. `select` — the deterministic rule → an LLM precision pick plus a decomposition front-end.

## HTTP surface

[src/agent_fleet_api/routes.py](../src/agent-fleet-api/src/agent_fleet_api/routes.py) — `GET
/healthz` (open) plus, behind auth: `GET /catalog`, `GET /report`, `POST /generate`, `POST
/render`, `POST /orchestrate`, and the pool routes (`POST|GET|DELETE /pool/{agent_key}`, `GET
/pool`, `GET /pool/find`, `GET /pool/{agent_key}/runs`, `POST /pool/{agent_key}/run`).

## Known issues

1. No `BaseSettings` in the core beyond `AgentFleetSettings`; the pipeline's budgets and thresholds are module-level constants.
2. `find_mcp` recall is name-only. MCP server cards carry a name-derived description, so recall matches only server-name tokens — a query like "drive a browser" won't surface `playwright`. Indexing each connected server's tools is the fix.
