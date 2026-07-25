# Teammates: named, revivable, background-run pooled agents

The primary MCP surface for agent-fleet: a hardcoded roster of templated teammates, addressed by
name, spawned into background runs, and revived against their standing pool session. The existing
key-addressed tools (`create_agent`, `run_agent`, …) remain beneath it unchanged.

## Context

The pool already provides the substance of a teammate:

- `create_agent` assembles a skill-scoped agent and stores it under a stable `AgentKey` with a
  session UUID ([pool_server.py](../../../src/agent_fleet_mcp/pool_server.py)).
- `run_agent` resumes that session on every run after the first
  ([dispatch.py](../../../src/agent_fleet/engine/dispatch.py)), including reviving a specific
  dispatched subagent via `resume_agent_id`.
- `get_agent` / `list_agents` / `find_agents` reference entries exactly or fuzzily.

What is missing: name-based addressing, templated capability bundles, and background execution —
`run_agent` blocks the MCP caller, and `RunOutcome.output` is never persisted
(the `runs` table stores only task + timestamps).

## Toolkits and templates

New module `src/agent_fleet/models/agent/teammate.py`:

```python
class Toolkit(FrozenModel):
    name: ToolkitName
    entries: list[CatalogEntryId]

class TeammateTemplate(FrozenModel):
    name: AgentName
    brief: TaskBrief
    toolkits: list[ToolkitName] = []
    tags: list[Tag] = []
    model: ModelId = ModelId.inherit
    system_prompt: PromptBody | None = None
```

- `ToolkitName` is a new domain alias in
  [types.py](../../../src/agent_fleet/models/agent/types.py) (slug, same shape discipline as
  `AgentName`).
- Toolkits and the roster are hardcoded tuples in `src/agent_fleet/engine/teammates.py`.
- `CatalogEntryId` is a deterministic slug of `{prefix}.{ref}` (capdisc `catalog_id`), so
  hardcoded toolkit entries stay valid across environment rescans while the capability keeps its
  name.
- Template resolution flattens `toolkits` into the pinned-id list handed to the existing
  `create_agent` path. Nothing below the template layer changes.
- A template's `agent_key` is derived deterministically: `teammate.{name}`. Name-based addressing
  is therefore key derivation, not a new lookup; `find_agents` remains the fuzzy fallback.
  Spawning is idempotent — same name, same pool entry, same resumed session.

## Pool changes

- `runs` gains three nullable columns: `output TEXT`, `structured_output_json TEXT`,
  `total_cost_usd REAL`. Guarded `ALTER TABLE` statements in pool init migrate existing DBs in
  place.
- `RunRecord` gains the matching optional fields; `finish_run` accepts the outcome to stamp them.
- Run status is derived, never stored: `finished_at` set → `finished`; unset and the run id is in
  the server's live registry → `running`; unset and unknown → `stale`. A stored status column
  would claim `running` forever after a crash and need a repair path; the derivation is
  self-healing because a crash empties the registry. A `stale` run's session is intact — a
  subsequent `message_teammate` resumes the conversation.

## Background runner

In `agent_fleet_mcp`: a `TeammateRunner` holding `dict[RunId, asyncio.Task]`.

- `spawn` calls the existing `prepare_run`, wraps `run_with_capture` in an `asyncio.Task`, and
  returns immediately with the started `RunRecord`.
- On completion the task persists the `RunOutcome` into the run row via the extended
  `finish_run`.
- Spawning a teammate whose latest run is still `running` returns that run's status instead of
  opening a second concurrent run against the same session.

## MCP tools

| Tool | Behavior |
| --- | --- |
| `roster()` | Every template plus live pool state: spawned or not, latest run status, session id. |
| `spawn_teammate(name, task, fresh_session=False)` | Resolve template → `create_agent` if the entry is absent → background run. Returns `{run_id, status: "running"}` immediately. `fresh_session=True` passes `reset_session` through, minting a new session UUID. |
| `check_teammate(name)` | Latest run's derived status; `output` / `structured_output` / `total_cost_usd` once finished; `unspawned` when the entry does not exist. |
| `message_teammate(name, text, wait=False, resume_agent_id=None)` | Revive the standing session; creates the entry from the template first if absent (same idempotent path as spawn). Backgrounded by default; `wait=True` blocks and returns the full outcome like `run_agent` today. `resume_agent_id` carries the dispatched-subagent revival path. |

Unknown teammate name → `ValueError` listing the valid roster names.

## Completion notification

- Persistence happens in-server when the run's coroutine finishes; no hook is required for
  correctness.
- Optional `AGENT_FLEET_NOTIFY_COMMAND` setting: when set, a `Stop` command hook is folded into
  each teammate run through the existing `with_hooks` settings-file path. Filesystem hooks fire in
  the CLI harness, so this works despite the Python SDK lacking programmatic
  `TeammateIdle`/`Stop` events.
- Docs show the push-style consumption pattern: the calling agent sets a harness `Monitor` on the
  check surface and is woken on completion, instead of polling in conversation.

## Testing

Using the repo's existing fake-`query()` stream patterns:

- `spawn_teammate` returns before the stream ends; completion persists output.
- Emptying the registry turns an unfinished run `stale`; re-messaging resumes the same session id.
- Pool init migrates a pre-existing DB (columns added, rows preserved).
- Roster resolution, toolkit flattening into pinned ids, `teammate.{name}` key derivation, and
  the unknown-name error.
- Gates: `uv run pytest`, `uv run ruff check`, `uv run mypy src`.

## Out of scope

- Detached per-teammate processes (runs do not survive an MCP-server restart; the session does).
- A stored teammate lifecycle model beyond `PoolEntry` + `RunRecord`.
- Dynamic (non-hardcoded) roster editing tools.
