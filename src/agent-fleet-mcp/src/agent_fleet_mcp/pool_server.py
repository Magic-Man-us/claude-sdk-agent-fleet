from __future__ import annotations

from functools import cache

from fastmcp import FastMCP

from agent_fleet.engine.dispatch import prepare_run, run_with_capture
from agent_fleet.engine.pool import AgentPool
from agent_fleet.engine.pool import create_agent as pool_create_agent
from agent_fleet.engine.source import CatalogSource, InMemoryCatalogSource
from agent_fleet.models.agent import (
    DEFAULT_TEAM,
    AgentId,
    AgentKey,
    AgentName,
    AgentRunRecord,
    Finding,
    ModelId,
    PoolEntry,
    ProblemRequest,
    PromptBody,
    RunId,
    RunOutcome,
    RunRecord,
    TaskBrief,
    TeamSlug,
)
from agent_fleet.router.capability import CapabilityRouter
from agent_fleet.settings import AgentFleetSettings, current_discovery_scope
from capabilities_discovery.catalog import (
    DEFAULT_RECALL_LIMIT,
    CatalogEntryId,
    DomainTag,
    RecallLimit,
    Tag,
)
from capabilities_discovery.discovery import scan_environment

mcp = FastMCP("agent-pool")


@cache
def _pool() -> AgentPool:
    """Build the process-wide pool, once, on first use.

    Lazily built (and cached) so importing this module does not touch disk. Reads
    `AgentFleetSettings().pool_db`.

    Returns:
        The SQLite-backed pool of problem-keyed, resumable agent sessions.
    """
    return AgentPool(AgentFleetSettings().pool_db)


@cache
def _source() -> CatalogSource:
    """Build the process-wide catalog source `create_agent` recalls against, once, on first use.

    Shares `current_discovery_scope()` with `mcp_server.py`'s `_router()` and this module's own
    `_capability_router()`, so a pool-created agent's skills and MCP servers are chosen the same
    way the rest of this codebase already chooses them.

    Returns:
        The in-memory catalog source wrapping the environment scan of the discovered scope roots.
    """
    roots = current_discovery_scope().roots()
    return InMemoryCatalogSource(scan_environment(roots))


@cache
def _capability_router() -> CapabilityRouter:
    """Build the process-wide capability router `run_agent` grants the acquire tool from, once,
    on first use.

    Shares `current_discovery_scope()` with `_source()` above, without the MCP-cache/plugin-harvest
    complexity `mcp_server.py`'s own router needs (this server's acquire tool only needs
    skill/tool/mcp recall, not the full capability-router product).

    Returns:
        The capability router wired from the discovered scope roots.
    """
    roots = current_discovery_scope().roots()
    return CapabilityRouter.from_environment(roots)


@mcp.tool
def create_agent(
    agent_key: AgentKey,
    task: TaskBrief,
    name: AgentName | None = None,
    domain: DomainTag | None = None,
    tags: list[Tag] = [],  # noqa: B006 — read-only; an empty-list default reads best in the tool schema
    team: TeamSlug = DEFAULT_TEAM,
    model: ModelId = ModelId.inherit,
    pinned: list[CatalogEntryId] = [],  # noqa: B006 — read-only; an empty-list default reads best in the tool schema
    system_prompt: PromptBody | None = None,
    reset_session: bool = False,
) -> PoolEntry:
    """Assemble an agent for a task and store it in the pool under `agent_key`, returning the
    stored entry — the create/update entry point an orchestrating agent uses to stand up a new
    pooled agent (or re-assemble an existing one).

    Args:
        agent_key: The stable agent key that identifies the stored entry; re-using a key
            overwrites it, preserving the existing session UUID unless `reset_session` is True.
        task: The task to build the agent for; also the text `find_agents` ranks against.
        name: Optional display name; auto-slugged from the task when omitted.
        domain: Optional domain to route recall; None recalls across every domain.
        tags: Optional routing tags to narrow recall and to label the stored entry.
        team: The team that owns the assembled agent.
        model: Which model the agent runs on; `inherit` defers to the caller's model.
        pinned: Capability ids to force into the selection regardless of relevance score.
        system_prompt: A full system-prompt override; when omitted, one is templated from the task
            and the selected tools/skills.
        reset_session: When True, mint a fresh session UUID even if the id already exists.

    Returns:
        The persisted pool entry, carrying the assembled spec, its session UUID, and timestamps.
    """
    request = ProblemRequest(
        task=task,
        name=name,
        domain=domain,
        tags=tags,
        team=team,
        model=model,
        pinned=pinned,
        system_prompt=system_prompt,
    )
    return pool_create_agent(agent_key, request, _source(), _pool(), reset_session=reset_session)


@mcp.tool
def get_agent(agent_key: AgentKey) -> PoolEntry | None:
    """Return the pooled entry stored under `agent_key`, or None when the pool holds no such entry.

    A safe probe: returns None rather than raising when the id is absent, so an orchestrating agent
    can check existence without handling an error.

    Args:
        agent_key: The agent key to look up.

    Returns:
        The stored pool entry, or None when there is none.
    """
    return _pool().get_by_key(agent_key)


@mcp.tool
def list_agents() -> list[PoolEntry]:
    """Return every pooled agent, most-recently-updated first.

    Returns:
        Every stored pool entry, newest-updated first.
    """
    return _pool().list()


@mcp.tool
def find_agents(query: TaskBrief, limit: RecallLimit = DEFAULT_RECALL_LIMIT) -> list[PoolEntry]:
    """Rediscover pooled agents by re-describing the problem, ranked most-relevant first —
    the fuzzy path for when the exact `agent_key` isn't at hand.

    Args:
        query: The re-described problem text to rank stored entries against.
        limit: The most entries to return.

    Returns:
        Up to `limit` entries, highest relevance first.
    """
    return _pool().find(query, limit)


@mcp.tool
def delete_agent(agent_key: AgentKey) -> bool:
    """Remove the pooled entry stored under `agent_key`.

    Args:
        agent_key: The agent key to remove.

    Returns:
        True if an entry existed and was removed, False if the id was absent.
    """
    return _pool().delete(agent_key)


@mcp.tool
def list_runs(agent_key: AgentKey) -> list[RunRecord]:
    """Return every run recorded for `agent_key`, most-recently-started first.

    Args:
        agent_key: The pooled agent whose runs to list.

    Returns:
        Every run record for the pooled agent, newest-started first; empty before any run.
    """
    return _pool().list_runs(agent_key)


@mcp.tool
def get_run(run_id: RunId) -> RunRecord | None:
    """Return one run record by its `run_id`, or None when the pool holds no such run.

    A safe probe, mirroring `get_agent`: returns None rather than raising when the id is
    absent, so an orchestrating agent can check existence without handling an error.

    Args:
        run_id: The run id to look up (from a `RunOutcome`/`RunRecord` returned by a prior call).

    Returns:
        The run record, or None when there is none.
    """
    return _pool().get_run(run_id)


@mcp.tool
def list_agent_runs(run_id: RunId) -> list[AgentRunRecord]:
    """List every agent that ran within one run and the real, resumable session id each was given.

    This is how an orchestrating agent discovers the independently-resumable `session_id` of every
    agent — the run's main/supervisor agent and any subagents it dispatched — that participated in a
    specific `run_id` (obtained from `list_runs`, `get_run`, or a
    `RunOutcome`). Each captured session id can then be used to resume that one agent's conversation
    on its own, independently of the others, later.

    Args:
        run_id: The run whose agents to list (from a `RunOutcome`/`RunRecord` returned by a prior
            call, or from `list_runs`/`get_run`).

    Returns:
        Every agent that ran within the run, main agent first then dispatched subagents in
        `recorded_at` order; each row carries that agent's captured, resumable session id. Empty
        when the run id is unknown.
    """
    return _pool().list_agent_runs(run_id)


@mcp.tool
def list_findings(agent_key: AgentKey, run_id: RunId | None = None) -> list[Finding]:
    """Return the pooled agent's findings oldest-first — the assembled-document reading order.

    Args:
        agent_key: The pooled agent whose findings to read.
        run_id: When given, return only the findings recorded within that run; None returns all.

    Returns:
        Every finding for the pooled agent (optionally filtered to `run_id`), oldest-first.
    """
    return _pool().list_findings(agent_key, run_id=run_id)


@mcp.tool
async def run_agent(
    agent_key: AgentKey,
    task: TaskBrief,
    subagent_agent_keys: dict[AgentName, AgentKey] = {},  # noqa: B006 — read-only; empty-dict default reads best in the tool schema
    resume_agent_id: AgentId | None = None,
) -> RunOutcome:
    """Run the pooled agent live and record every agent that participates. The entry's first-ever
    run starts a fresh session; later runs resume it. Requires the claude CLI at runtime.

    Every run is granted the full capability set — for the main agent and, when named, for each
    dispatched subagent: fan-out subagents (resolved from their own pool entries and wired in via
    `with_subagents`), the run-scoped findings-writer (`with_findings_tool`), and dynamic capability
    acquisition (`with_acquire_tool`). The findings tool needs the run id at wiring time, so the run
    is started here and its id handed to `run_with_capture` rather than minted inside it.

    Args:
        agent_key: The pooled agent to run; must already exist in the pool.
        task: The first user turn handed to the agent, and the run's recorded task.
        subagent_agent_keys: A subagent name → pooled `agent_key` mapping; each referenced entry's
            spec is wired in as a dispatchable subagent that also carries the findings and acquire
            tools. Empty runs the agent solo.
        resume_agent_id: When set, continue one specific previously-dispatched subagent (its
            `AgentId`, captured from an earlier `RunOutcome.agent_runs[i].agent_id`) rather than
            just re-prompting the main agent: this run resumes the main session, is granted
            the harness's `SendMessage` tool (`with_agent_resume`), and its literal turn wraps
            `task` as `"Resume agent {id} and now: {task}"` — while the run record still logs
            `task`. It composes with `subagent_agent_keys`, since the resumed turn may dispatch
            further subagents too.

    Returns:
        The run's collected assistant text, the finished run record, every captured agent run, and
        — when produced — the terminal result's structured output and total cost.

    Raises:
        ValueError: When `agent_key` has no pool entry, or when any named `subagent_agent_keys`
            entry has no pool entry.
    """
    pool = _pool()
    try:
        run, options, prompt = prepare_run(
            pool,
            _capability_router(),
            agent_key,
            task,
            subagent_agent_keys=subagent_agent_keys,
            resume_agent_id=resume_agent_id,
        )
    except KeyError as exc:
        missing = exc.args[0]
        if missing == agent_key:
            raise ValueError(f"pool entry not found: {agent_key}") from exc
        raise ValueError(f"subagent pool entry not found: {missing}") from exc
    return await run_with_capture(pool, agent_key, task, options, run=run, prompt=prompt)


def main() -> None:
    """Run the agent-pool MCP server over stdio (the `pool-mcp` console entry point)."""
    mcp.run()


if __name__ == "__main__":
    main()
