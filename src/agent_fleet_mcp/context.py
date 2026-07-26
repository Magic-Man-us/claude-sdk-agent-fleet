"""Shared runtime for the MCP servers: pooled state, roster resolution, and run plumbing.

Both servers drive the same pool and the same roster, so this holds the one implementation of
each. Callers reach these through the module (`context.pool()`, not `from .context import pool`)
so a test can substitute one and have every server see the substitution.
"""

from __future__ import annotations

import logging
from collections.abc import Coroutine
from functools import cache

from pydantic import TypeAdapter

from agent_fleet.engine.pool import AgentPool
from agent_fleet.engine.pool import create_agent as pool_create_agent
from agent_fleet.engine.source import CatalogSource, InMemoryCatalogSource
from agent_fleet.engine.teammates import build_teammate, current_roster, resolve_template
from agent_fleet.models.agent import (
    AgentKey,
    AgentName,
    PoolEntry,
    RosterFile,
    RunError,
    RunId,
    RunOutcome,
    RunRecord,
    TeammateBuild,
    TeammateRunStatus,
    TeammateStatus,
    teammate_key,
)
from agent_fleet.router.capability import CapabilityRouter
from agent_fleet.settings import AgentFleetSettings, current_discovery_scope
from capdisc.discovery import scan_environment
from capdisc.hooks import CommandHook, HookConfig, HookEvent, MatcherGroup

from .runner import TeammateRunner, run_status

logger = logging.getLogger(__name__)

_RUN_ERROR_ADAPTER: TypeAdapter[RunError] = TypeAdapter(RunError)


@cache
def pool() -> AgentPool:
    """Build the process-wide pool, once, on first use.

    Lazily built (and cached) so importing this module does not touch disk. Reads
    `AgentFleetSettings().pool_db`.

    Returns:
        The SQLite-backed pool of problem-keyed, resumable agent sessions.
    """
    return AgentPool(AgentFleetSettings().pool_db)


@cache
def source() -> CatalogSource:
    """Build the process-wide catalog source `create_agent` recalls against, once, on first use.

    Shares `current_discovery_scope()` with `mcp_server.py`'s `_router()` and this module's own
    `capability_router()`, so a pool-created agent's skills and MCP servers are chosen the same
    way the rest of this codebase already chooses them.

    Returns:
        The in-memory catalog source wrapping the environment scan of the discovered scope roots.
    """
    roots = current_discovery_scope().roots()
    return InMemoryCatalogSource(scan_environment(roots))


@cache
def capability_router() -> CapabilityRouter:
    """Build the process-wide capability router `run_agent` grants the acquire tool from, once,
    on first use.

    Shares `current_discovery_scope()` with `source()` above, without the MCP-cache/plugin-harvest
    complexity `mcp_server.py`'s own router needs (this server's acquire tool only needs
    skill/tool/mcp recall, not the full capability-router product).

    Returns:
        The capability router wired from the discovered scope roots.
    """
    roots = current_discovery_scope().roots()
    return CapabilityRouter.from_environment(roots)


@cache
def runner() -> TeammateRunner:
    """The process-wide registry of live background teammate runs, once, on first use."""
    return TeammateRunner()


@cache
def roster_in_force() -> RosterFile:
    """The roster in force, once, on first use.

    Cached like the other process-wide accessors, so a long-lived server does not re-read and
    re-validate the file on every call. Restart the server to pick up an edited roster.
    """
    return current_roster()


def build(name: AgentName) -> TeammateBuild:
    """Expand teammate `name` into its request and subagent list.

    Raises:
        ValueError: When `name` is not on the roster.
    """
    roster = roster_in_force()
    return build_teammate(
        resolve_template(name, roster),
        roster,
        subagent_model=AgentFleetSettings().subagent_model,
    )


def ensure_teammate(name: AgentName, *, fresh_session: bool = False) -> PoolEntry:
    """The pool entry for teammate `name`, creating it from its roster template when absent.

    Raises:
        ValueError: When `name` is not on the roster.
    """
    key = teammate_key(name)
    entry = pool().get_by_key(key)
    if entry is not None and not fresh_session:
        return entry
    return pool_create_agent(
        key, build(name).request, source(), pool(), reset_session=fresh_session
    )


def subagent_keys(name: AgentName) -> dict[AgentName, AgentKey]:
    """The teammate's toolkit-granted agents, as the subagent name → pool key mapping a run wants.

    Each named agent is itself a roster teammate, so its entry is stood up on demand — dispatching
    to a teammate that has never been spawned works without the caller sequencing it.
    """
    keys: dict[AgentName, AgentKey] = {}
    for agent in build(name).agents:
        keys[agent] = ensure_teammate(agent).agent_key
    return keys


def notify_hooks() -> HookConfig | None:
    """The Stop-notification hook for teammate runs, from `AGENT_FLEET_NOTIFY_COMMAND`; None
    when unconfigured.

    Deliberately NOT `@cache`d unlike the sibling accessors above: the env var must be re-read on
    every run, not frozen at first use — a cached `HookConfig` would also survive across the
    `monkeypatch.setenv` calls the tests use to toggle it.
    """
    command = AgentFleetSettings().notify_command
    if command is None:
        return None
    return HookConfig({HookEvent.stop: [MatcherGroup(hooks=[CommandHook(command=command)])]})


async def record_failure(run_id: RunId, coro: Coroutine[object, object, RunOutcome]) -> RunOutcome:
    """Stamp the run `failed` with the exception text before re-raising.

    `run_with_capture`'s own `finish_run` call sits after its message loop, so a stream that
    raises mid-run never reaches it — the row stays open, and once the task deregisters
    `run_status` would report it `stale` rather than surfacing the failure. This wraps every
    dispatched run (background and `wait=True` alike) so a raised run is still visible via
    `check_teammate`.

    The composed message is validated through `RunError` before it reaches `finish_run` —
    `finish_run` itself is a plain function, so its `RunError` parameter annotation does not
    enforce anything on the way in. Bounding it here means the stored value is already within
    `RunError`'s length limit, matching what its own truncating validator would produce on read —
    write and read cannot disagree.

    Known limit: a mid-stream failure discards the partial transcript — this wrapper sees only the
    exception, not whatever text the stream had already produced. Capturing partial output would
    require `run_with_capture` itself to stamp its own failure path (it already collects `parts` as
    it streams); deliberately not done here.

    The teammate's entry can be dismissed (`delete_agent`) while its run is still live — deleting
    it cascades to its `runs` row, so `finish_run` below finds nothing to stamp and raises its own
    `KeyError`. That would otherwise replace the run's real failure as this task's terminal
    exception (Python's implicit exception chaining keeps the original in `__context__`, but the
    propagating exception — and what `check_teammate`/log tooling see as `task.exception()` — would
    be the `KeyError`, not the actual reason the run failed). Caught and logged here instead;
    `finish_run`'s own `KeyError` contract for a genuinely-unknown run id is unchanged elsewhere.
    """
    try:
        return await coro
    except Exception as exc:
        error = _RUN_ERROR_ADAPTER.validate_python(f"{type(exc).__name__}: {exc}")
        try:
            pool().finish_run(run_id, error=error)
        except KeyError:
            logger.exception("teammate run %s: row deleted while live — failure unrecorded", run_id)
        raise


def live_run(key: AgentKey) -> RunRecord | None:
    """The teammate's latest run, when it is still live (registered and unfinished); None when
    it is safe to queue a new one. Checked before any session mutation (`fresh_session`) so a
    live run's session can't be silently reset out from under it."""
    latest = pool().latest_run(key)
    if latest is not None and run_status(latest, runner()) is TeammateRunStatus.running:
        return latest
    return None


def teammate_status(name: AgentName) -> TeammateStatus:
    """The derived status of `name`'s latest run, with persisted outcome fields once finished (or
    its captured error once failed)."""
    key = teammate_key(name)
    entry = pool().get_by_key(key)
    if entry is None:
        return TeammateStatus(name=name, agent_key=key, status=TeammateRunStatus.unspawned)
    latest = pool().latest_run(key)
    return TeammateStatus(
        name=name,
        agent_key=key,
        status=run_status(latest, runner()),
        run_id=latest.run_id if latest is not None else None,
        session_id=entry.session_id,
        output=latest.output if latest is not None else None,
        structured_output=latest.structured_output if latest is not None else None,
        total_cost_usd=latest.total_cost_usd if latest is not None else None,
        error=latest.error if latest is not None else None,
    )
