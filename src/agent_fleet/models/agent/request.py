from __future__ import annotations

from capdisc.base import FrozenModel
from capdisc.catalog import CatalogEntryId, Tag, ToolRef

from .types import (
    DEFAULT_TEAM,
    AgentName,
    ModelId,
    PromptBody,
    TaskBrief,
    TeamSlug,
)


class ProblemRequest(FrozenModel):
    """The full input to agent generation: the task to build for and, optionally, the agent's
    display name (auto-slugged from the task when omitted), plus optional tag routing, a
    model choice, pinned capability ids, directly granted tools, and a system-prompt override.

    `pinned` and `tools` differ by mechanism, not preference: pinned ids are catalog entries kept
    past the relevance filter during selection, while tools are granted outright because the
    selector does not score tools."""

    task: TaskBrief
    name: AgentName | None = None
    tags: list[Tag] = []
    team: TeamSlug = DEFAULT_TEAM
    model: ModelId = ModelId.inherit
    pinned: list[CatalogEntryId] = []
    tools: list[ToolRef] = []
    system_prompt: PromptBody | None = None
