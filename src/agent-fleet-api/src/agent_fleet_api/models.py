from __future__ import annotations

from pathlib import Path
from typing import Literal

from agent_fleet.models.agent import AgentId, AgentKey, AgentName, TaskBrief
from capabilities_discovery.base import FrozenModel

from .types import SdkSource


class Health(FrozenModel):
    """Liveness payload for `GET /healthz`."""

    status: Literal["ok"] = "ok"


class RenderedAgent(FrozenModel):
    """The emitted Claude Agent SDK program for a posted spec."""

    source: SdkSource
    path: Path | None = None
    """Where the program was persisted; None when no ``agent_dir`` is configured."""


class OrchestrateRequest(FrozenModel):
    """Input for `POST /orchestrate` — the task to run the orchestrator against."""

    task: TaskBrief


class PoolRunRequest(FrozenModel):
    """Input for `POST /pool/{agent_key}/run` — the task to run the pooled agent against, plus an
    optional subagent wiring. `subagent_agent_keys` maps a subagent display name to another pool
    entry's `agent_key`, resolved server-side to that entry's spec and wired in via
    `with_subagents`; empty means the agent runs with no subagents. `resume_agent_id`, when set,
    continues one specific previously-dispatched subagent (its `AgentId`, captured from an earlier
    `RunOutcome.agent_runs[i].agent_id`) rather than just re-prompting the main agent: the run
    resumes the main session, is granted `SendMessage` (`with_agent_resume`), and its literal turn
    wraps `task` as `"Resume agent {id} and now: {task}"` — while the run record still logs `task`.
    It composes with `subagent_agent_keys`, since the resumed turn may dispatch further subagents
    too."""

    task: TaskBrief
    subagent_agent_keys: dict[AgentName, AgentKey] = {}
    resume_agent_id: AgentId | None = None
