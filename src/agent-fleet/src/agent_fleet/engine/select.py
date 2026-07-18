from __future__ import annotations

import logging
from functools import singledispatch

from capabilities_discovery.base import FrozenModel, MutableModel
from capabilities_discovery.catalog import (
    RELEVANCE_THRESHOLD,
    BuiltinTool,
    CatalogMcpServer,
    CatalogSkill,
    McpServerRef,
    SkillRef,
    ToolRef,
)

from ..models.agent import DEFAULT_SKILL_BUDGET, ProblemRequest, SkillBudget
from .source import Candidate

logger = logging.getLogger(__name__)

# The fixed tool grant every generated agent receives. Built-in tools are provisioned, not
# retrieved: their descriptions are mechanical and never match a task's goal language, so lexical
# recall can't surface them. This set covers inspect (Read/Glob/Grep), modify (Write/Edit), and
# run (Bash) — the working core of almost any task.
DEFAULT_TOOLS: list[ToolRef] = [
    BuiltinTool.read.value,
    BuiltinTool.glob.value,
    BuiltinTool.grep.value,
    BuiltinTool.write.value,
    BuiltinTool.edit.value,
    BuiltinTool.bash.value,
]


class SelectedCapabilities(FrozenModel):
    """Skills, tools, and MCP servers chosen to equip one generated agent."""

    skills: list[SkillRef] = []
    tools: list[ToolRef] = []
    mcp_servers: list[McpServerRef] = []


class _Selection(MutableModel):
    """Mutable accumulator for the recalled capabilities: budgeted, order-preserving, de-duplicated
    skill and MCP-server buckets. Tools are not recalled — they are the fixed `DEFAULT_TOOLS`."""

    skill_budget: SkillBudget
    skills: list[SkillRef] = []
    mcp_servers: list[McpServerRef] = []

    def take_skill(self, ref: SkillRef) -> str | None:
        """Add a skill ref under its budget; None on success, else the rejection reason."""
        return _take(self.skills, ref, self.skill_budget, "skill_budget")

    def add_server(self, ref: McpServerRef) -> None:
        """Add an MCP server ref, de-duplicating; servers are unbudgeted."""
        if ref not in self.mcp_servers:
            self.mcp_servers.append(ref)


def _take[T](bucket: list[T], ref: T, budget: int, reason: str) -> str | None:
    """Append `ref` to `bucket` if it fits, reporting why it didn't otherwise.

    Args:
        bucket: The accumulator list; appended to in place on success.
        ref: The ref to add.
        budget: Maximum bucket length.
        reason: The reason string to return when the budget is full.

    Returns:
        None on append, `"duplicate"` if `ref` is already present, or `reason` if the budget
        is full.
    """
    if ref in bucket:
        return "duplicate"
    if len(bucket) >= budget:
        return reason
    bucket.append(ref)
    return None


@singledispatch
def _equip(entry: object, sel: _Selection) -> str | None:  # noqa: ARG001 — dispatch base; variants use sel
    """Route one recalled catalog entry into the matching bucket of `sel`.

    Dispatches on the entry's concrete type (see the registered variants below). Only skills and
    MCP servers are recalled; tools are provisioned separately.

    Args:
        entry: The catalog entry to equip.
        sel: The selection accumulator to mutate.

    Returns:
        None when equipped, else the rejection reason from the bucket.

    Raises:
        TypeError: If the entry type has no registered handler.
    """
    raise TypeError(f"unhandled catalog entry: {type(entry).__name__}")


@_equip.register
def _(entry: CatalogSkill, sel: _Selection) -> str | None:
    return sel.take_skill(entry.ref)


@_equip.register
def _(entry: CatalogMcpServer, sel: _Selection) -> str | None:
    sel.add_server(entry.ref)
    return None


def select(candidates: list[Candidate], request: ProblemRequest) -> SelectedCapabilities:
    """Pick the capabilities to equip from ranked candidates.

    Walks candidates in rank order, equipping skills (capped at `DEFAULT_SKILL_BUDGET`) and MCP
    servers; entries below the relevance threshold are dropped unless pinned. Tools are not
    recalled — every agent gets the fixed `DEFAULT_TOOLS`. Rejections are logged.

    Args:
        candidates: Ranked candidates, highest score first.
        request: The problem request, supplying the pinned ids.

    Returns:
        The chosen skills, the fixed tool set, and the chosen MCP servers.
    """
    pinned = set(request.pinned)
    sel = _Selection(skill_budget=DEFAULT_SKILL_BUDGET)
    for c in candidates:
        reason = (
            "below_threshold"
            if c.entry.id not in pinned and c.score < RELEVANCE_THRESHOLD
            else _equip(c.entry, sel)
        )
        if reason:
            logger.debug("rejected %s (%.3f): %s", c.entry.id, c.score, reason)
    return SelectedCapabilities(skills=sel.skills, tools=DEFAULT_TOOLS, mcp_servers=sel.mcp_servers)
