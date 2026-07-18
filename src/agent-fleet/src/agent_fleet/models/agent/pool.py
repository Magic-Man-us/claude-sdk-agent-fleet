from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import JsonValue

from capabilities_discovery.base import FrozenModel

from .spec import AgentSpec
from .types import AgentId, AgentKey, AgentName, FindingContent, RunId, SessionId, TaskBrief


class PoolEntry(FrozenModel):
    """A pooled agent held in the pool: its spec plus the Claude Agent SDK session UUID
    that pins and resumes the same live conversation. `agent_key` is the logical key (a stable
    external id); `name` is the human-readable display label; `session_id` is the internally
    generated UUID — the pool maintains the mapping so a resume by agent key continues the same
    session history. `cwd` is the working directory every run of this entry is launched from: Claude
    stores session files under `~/.claude/projects/<encoded-cwd>/`, so a resume from a mismatched
    cwd silently starts a fresh session instead of continuing history — pinning it here keeps the
    resume pointed at the same on-disk session."""

    agent_key: AgentKey
    name: AgentName
    spec: AgentSpec
    session_id: SessionId
    cwd: Path
    created_at: datetime
    updated_at: datetime


class RunRecord(FrozenModel):
    """One invocation of a pooled agent — appended when a run starts and stamped when it finishes.
    Distinct from `PoolEntry`'s "current state per problem" role: an entry has many runs over its
    life. `finished_at` is None while the run is in flight."""

    run_id: RunId
    agent_key: AgentKey
    task: TaskBrief
    started_at: datetime
    finished_at: datetime | None = None


class AgentRunRecord(FrozenModel):
    """One agent that ran within a `RunRecord` — the run's main agent or a subagent it dispatched.
    `tool_use_id`, `agent_name`, and `agent_id` are all None for the run's top-level agent; all set
    for a dispatched one, where `tool_use_id` is the id of the `Agent`/`Task` tool-use block that
    spawned it and `agent_name` its `subagent_type`.

    `session_id` is the SAME session for every agent in a run — main and dispatched alike share the
    parent's session id (a dispatched subagent gets no separate resumable session). It identifies
    which `PoolEntry`'s conversation the run belongs to, not a per-subagent session. `agent_id` is
    the per-subagent identifier, captured from the dispatch's `TaskStartedMessage.task_id`; it is
    the handle to continue that one specific dispatched subagent later via the harness's
    `SendMessage` tool (see `render.with_agent_resume`)."""

    run_id: RunId
    tool_use_id: str | None
    agent_name: AgentName | None
    agent_id: AgentId | None
    session_id: SessionId
    recorded_at: datetime


class Finding(FrozenModel):
    """One deposited finding in a pooled agent's shared, append-only findings document. Every lens
    dispatched within a run — and the run's own main/supervisor agent — writes into the same table;
    a finding is never updated in place, so concurrent lenses can't clobber each other. `agent_name`
    is the lens that wrote it (None for the main/supervisor agent, mirroring `AgentRunRecord`), and
    `session_id` is the writing agent's session, so a finding traces back to exactly which run of
    exactly which agent produced it."""

    agent_key: AgentKey
    run_id: RunId
    agent_name: AgentName | None
    session_id: SessionId
    content: FindingContent
    recorded_at: datetime


class RunOutcome(FrozenModel):
    """The result of a captured run: the main agent's collected text, the finished `RunRecord`,
    every agent (main plus dispatched) whose session id was captured while it ran, and — when the
    run's options requested it — the terminal result's forced structured output and total cost.

    `structured_output`/`total_cost_usd` are None unless `options.output_format` asked the SDK to
    force a JSON-schema-validated result; `run_with_capture` populates them from the run's terminal
    `ResultMessage` when present."""

    output: str
    run: RunRecord
    agent_runs: list[AgentRunRecord]
    structured_output: JsonValue | None = None
    total_cost_usd: float | None = None
