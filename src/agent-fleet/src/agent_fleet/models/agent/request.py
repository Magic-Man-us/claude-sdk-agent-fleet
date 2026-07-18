from __future__ import annotations

from capabilities_discovery.base import FrozenModel
from capabilities_discovery.catalog import CatalogEntryId, Tag

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
    model choice, pinned capability ids, and a system-prompt override."""

    task: TaskBrief
    name: AgentName | None = None
    tags: list[Tag] = []
    team: TeamSlug = DEFAULT_TEAM
    model: ModelId = ModelId.inherit
    pinned: list[CatalogEntryId] = []
    system_prompt: PromptBody | None = None
