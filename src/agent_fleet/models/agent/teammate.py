from __future__ import annotations

from enum import StrEnum

from pydantic import JsonValue

from capdisc.base import FrozenModel
from capdisc.catalog import CatalogEntryId, Tag

from .types import (
    AgentKey,
    AgentName,
    CostUsd,
    ModelId,
    PromptBody,
    RunId,
    RunOutput,
    SessionId,
    TaskBrief,
    ToolkitName,
)


class TeammateRunStatus(StrEnum):
    """Derived state of a teammate's latest run — never stored, so it cannot lie after a crash:
    `stale` means an unfinished run unknown to the live registry (the session itself is intact)."""

    unspawned = "unspawned"
    idle = "idle"
    running = "running"
    finished = "finished"
    stale = "stale"


class Toolkit(FrozenModel):
    """A named capability bundle a template pins — the dedicated skills of a teammate."""

    name: ToolkitName
    entries: list[CatalogEntryId]


class TeammateTemplate(FrozenModel):
    """One hardcoded roster entry: the name a teammate is addressed by and what it is built from."""

    name: AgentName
    brief: TaskBrief
    toolkits: list[ToolkitName] = []
    tags: list[Tag] = []
    model: ModelId = ModelId.inherit
    system_prompt: PromptBody | None = None


class TeammateStatus(FrozenModel):
    """What `spawn_teammate`/`check_teammate`/`message_teammate` report: the derived status of the
    teammate's latest run plus its persisted outcome once finished."""

    name: AgentName
    agent_key: AgentKey
    status: TeammateRunStatus
    run_id: RunId | None = None
    session_id: SessionId | None = None
    output: RunOutput | None = None
    structured_output: JsonValue | None = None
    total_cost_usd: CostUsd | None = None


class RosterEntry(FrozenModel):
    """One `roster()` row: the template plus the live pool state of its teammate."""

    template: TeammateTemplate
    status: TeammateRunStatus
    session_id: SessionId | None = None


TEAMMATE_KEY_PREFIX = "teammate."


def teammate_key(name: AgentName) -> AgentKey:
    """The deterministic pool key for a teammate — name-based addressing is key derivation."""
    return f"{TEAMMATE_KEY_PREFIX}{name}"
