from __future__ import annotations

from enum import StrEnum

from pydantic import JsonValue, model_validator, validate_call

from capdisc.base import FrozenModel
from capdisc.catalog import McpServerRef, SkillRef, Tag, ToolRef

from .request import ProblemRequest
from .types import (
    AgentKey,
    AgentName,
    CostUsd,
    ModelId,
    PromptBody,
    RunError,
    RunId,
    RunOutput,
    SessionId,
    TaskBrief,
    ToolkitName,
)


class TeammateRunStatus(StrEnum):
    """Derived state of a teammate's latest run — never stored, so it cannot lie after a crash:
    `unspawned` means no pool entry exists yet; `idle` means the entry exists but has never run;
    `stale` means an unfinished run unknown to the live registry (the session itself is intact);
    `failed` means `finished_at` is set with a recorded error — the run raised rather than
    completed."""

    unspawned = "unspawned"
    idle = "idle"
    running = "running"
    finished = "finished"
    failed = "failed"
    stale = "stale"


class Toolkit(FrozenModel):
    """A named bundle of everything a teammate should be able to reach.

    Each kind lands somewhere different on the assembled agent, which is why they are separate
    fields rather than one opaque id list: `skills` and `mcp_servers` are pinned past the
    relevance filter so they survive selection, `tools` are granted directly (the selector does
    not score tools), and `agents` name other roster teammates to wire in as dispatchable
    subagents. Refs are the human-writable forms — `"Read"`, `"playwright"`, `"my-plugin:skill"` —
    so a hand-edited roster file reads like the thing it grants.
    """

    name: ToolkitName
    skills: list[SkillRef] = []
    mcp_servers: list[McpServerRef] = []
    tools: list[ToolRef] = []
    agents: list[AgentName] = []

    @model_validator(mode="after")
    def _grants_something(self) -> Toolkit:
        """Reject an empty toolkit: naming one that grants nothing is always a config mistake."""
        if not (self.skills or self.mcp_servers or self.tools or self.agents):
            raise ValueError(f"toolkit {self.name!r} grants nothing")
        return self


class TeammateTemplate(FrozenModel):
    """One roster entry: the name a teammate is addressed by and what it is built from."""

    name: AgentName
    brief: TaskBrief
    toolkits: list[ToolkitName] = []
    tags: list[Tag] = []
    model: ModelId = ModelId.inherit
    system_prompt: PromptBody | None = None


class RosterFile(FrozenModel):
    """A roster as written on disk: the toolkits available and the teammates built from them.

    This is the whole user-editable surface. It is validated on load, so a typo in a toolkit name
    or an unknown model is a named error at startup rather than a confusing failure at spawn.
    """

    toolkits: list[Toolkit] = []
    teammates: list[TeammateTemplate] = []

    @model_validator(mode="after")
    def _references_resolve(self) -> RosterFile:
        """Reject duplicate names and toolkit references that point at nothing.

        The same integrity the shipped roster gets, applied to a hand-written file — the errors
        name the offending entry, since the author is a person editing TOML, not a caller.
        """
        for label, names in (
            ("toolkit", [kit.name for kit in self.toolkits]),
            ("teammate", [mate.name for mate in self.teammates]),
        ):
            duplicates = sorted({name for name in names if names.count(name) > 1})
            if duplicates:
                raise ValueError(f"duplicate {label} names: {', '.join(duplicates)}")

        known = {kit.name for kit in self.toolkits}
        for mate in self.teammates:
            missing = sorted(set(mate.toolkits) - known)
            if missing:
                raise ValueError(
                    f"teammate {mate.name!r} references unknown toolkit(s): {', '.join(missing)}"
                )

        crew = {mate.name for mate in self.teammates}
        for kit in self.toolkits:
            strangers = sorted(set(kit.agents) - crew)
            if strangers:
                raise ValueError(
                    f"toolkit {kit.name!r} names agents that are not teammates: "
                    f"{', '.join(strangers)}"
                )
        return self


class TeammateStatus(FrozenModel):
    """What `spawn_teammate`/`check_teammate`/`message_teammate` report: the derived status of the
    teammate's latest run plus its persisted outcome once finished — or its captured error once
    failed."""

    name: AgentName
    agent_key: AgentKey
    status: TeammateRunStatus
    run_id: RunId | None = None
    session_id: SessionId | None = None
    output: RunOutput | None = None
    structured_output: JsonValue | None = None
    total_cost_usd: CostUsd | None = None
    error: RunError | None = None


class TeammateBuild(FrozenModel):
    """What a template expands into: the generation request, plus the teammates to wire in as
    dispatchable subagents. Agents are separate because they are applied per run rather than
    baked into the stored spec."""

    request: ProblemRequest
    agents: list[AgentName] = []


class RosterEntry(FrozenModel):
    """One `roster()` row: the template plus the live pool state of its teammate."""

    template: TeammateTemplate
    status: TeammateRunStatus
    session_id: SessionId | None = None


TEAMMATE_KEY_PREFIX = "teammate."


@validate_call(validate_return=True)
def teammate_key(name: AgentName) -> AgentKey:
    """The deterministic pool key for a teammate — name-based addressing is key derivation.

    Validated both ways: a bare annotation would let a caller pass any string and mint a key that
    is not a valid `AgentKey`, which the pool would then store and never round-trip cleanly.
    """
    return f"{TEAMMATE_KEY_PREFIX}{name}"
