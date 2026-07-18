from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import validate_call

from capabilities_discovery.base import FrozenModel
from capabilities_discovery.catalog import (
    Catalog,
    CatalogEntryId,
    CatalogMcpServer,
    CatalogPlugin,
    CatalogSkill,
    CatalogTool,
    DomainTag,
    EntryDescription,
    McpServerRef,
    McpTool,
    McpToolCount,
    McpToolName,
    PluginRef,
    RecallLimit,
    SkillRef,
    Tag,
    ToolRef,
)
from capabilities_discovery.discovery import BUILTIN_TOOLS, scan_indexed_skills
from capabilities_discovery.scope import ScopeRoots

from ..engine.source import (
    InMemoryCatalogSource,
    RecallQuery,
    TwoStageRanker,
    bm25_normalized,
    token_list,
)
from ..models.agent import TaskBrief

DEFAULT_SLATE = 5
MCP_CARD_TOOLS = 10


def relevant_tool_names(query: TaskBrief, tools: list[McpTool], limit: int) -> list[McpToolName]:
    """The `limit` tool names whose text best matches the task, by the entry-ranking BM25.

    Each tool is scored on its name, parameter names, and description — the same text the
    server-level recall ranks on. Ties and the no-overlap case fall back to harvested order so
    the result is deterministic and never empty when the server has tools.

    Args:
        query: The task to rank the server's tools against.
        tools: The server's harvested tools, in harvested order.
        limit: Maximum tool names to return.

    Returns:
        Up to `limit` tool names, most relevant first; the first `limit` in harvested order when
        no tool text overlaps the query.
    """
    docs = [token_list(f"{t.name} {' '.join(t.params)} {t.description}") for t in tools]
    scores = bm25_normalized(sorted(set(token_list(query))), docs)
    if not any(scores):
        return [t.name for t in tools[:limit]]
    ranked = sorted(zip(tools, scores, strict=True), key=lambda pair: (-pair[1], pair[0].name))
    return [tool.name for tool, _ in ranked[:limit]]


class _CardView(FrozenModel):
    """Shared agent-facing fields of one capability card from a `find_*` slate."""

    id: CatalogEntryId
    description: EntryDescription
    tags: list[Tag] = []


class SkillCard(_CardView):
    """The agent-facing view of one skill from `find_skills` — enough to decide whether to
    load it, without pulling the full SKILL.md body into context."""

    kind: Literal["skill"] = "skill"
    ref: SkillRef
    domain: DomainTag | None = None


class ToolCard(_CardView):
    """The agent-facing view of one tool from `find_tools` — the grant ref and what it does."""

    kind: Literal["tool"] = "tool"
    ref: ToolRef


class McpCard(_CardView):
    """The agent-facing view of one MCP server from `find_mcp` — the server name, its purpose, and
    the tool names most relevant to the task (full input schemas stay behind `describe_mcp`)."""

    kind: Literal["mcp_server"] = "mcp_server"
    ref: McpServerRef
    tool_count: McpToolCount
    relevant_tools: list[McpToolName] = []


class PluginCard(_CardView):
    """The agent-facing view of one installed plugin from `find_plugins` — the plugin name and
    the skills and MCP servers it bundles."""

    kind: Literal["plugin"] = "plugin"
    ref: PluginRef
    skills: list[SkillRef] = []
    mcp_servers: list[McpServerRef] = []


class CapabilityRouter:
    """The deferred-tool capability router: indexes the environment once, then answers a task
    with a small BM25-ranked slate of the relevant skills, tools, or MCP servers. The agent sees
    `find_skills`/`find_tools`/`find_mcp`, not every capability registered into context. One
    source per kind, so each slate is the top-k of that kind (skills are domain-routed too)."""

    def __init__(
        self,
        skills: list[CatalogSkill],
        tools: list[CatalogTool],
        mcp_servers: list[CatalogMcpServer],
        plugins: list[CatalogPlugin],
        paths: dict[CatalogEntryId, Path],
    ) -> None:
        """Build one ranked source per capability kind from the pre-indexed cards.

        Args:
            skills: The skill cards; ranked with the domain-routing `TwoStageRanker`.
            tools: The tool cards.
            mcp_servers: The MCP server cards.
            plugins: The plugin cards.
            paths: Map from skill id to its SKILL.md path, for `load_skill`.
        """
        self._skills = InMemoryCatalogSource(Catalog(entries=list(skills)), ranker=TwoStageRanker())
        self._tools = InMemoryCatalogSource(Catalog(entries=list(tools)))
        self._mcp = InMemoryCatalogSource(Catalog(entries=list(mcp_servers)))
        self._plugins = InMemoryCatalogSource(Catalog(entries=list(plugins)))
        self._paths: dict[CatalogEntryId, Path] = paths
        self._mcp_by_ref: dict[McpServerRef, CatalogMcpServer] = {s.ref: s for s in mcp_servers}

    @classmethod
    def from_environment(
        cls,
        roots: ScopeRoots,
        mcp_servers: Sequence[CatalogMcpServer] = (),
        plugins: Sequence[CatalogPlugin] = (),
    ) -> CapabilityRouter:
        """Build a router by indexing the skills under `roots` plus the built-in tools.

        Args:
            roots: The scope roots to scan for skills.
            mcp_servers: Connected MCP server cards to index.
            plugins: Installed plugin cards to index.

        Returns:
            A router over the indexed skills, `BUILTIN_TOOLS`, servers, and plugins.
        """
        indexed = scan_indexed_skills(roots)
        skills = [card for card, _ in indexed]
        paths = {card.id: path for card, path in indexed}
        return cls(skills, list(BUILTIN_TOOLS), list(mcp_servers), list(plugins), paths)

    def _query(
        self, text: TaskBrief, limit: RecallLimit, domain: DomainTag | None = None
    ) -> RecallQuery:
        """A find_* recall query for a task, optionally routed by domain."""
        return RecallQuery(text=text, domain=domain, limit=limit)

    @validate_call
    def find_skills(
        self, query: TaskBrief, domain: DomainTag | None = None, limit: RecallLimit = DEFAULT_SLATE
    ) -> list[SkillCard]:
        """The top-ranked skill cards for a task.

        Args:
            query: The task to match skills against.
            domain: Optional domain to route within; None searches every domain.
            limit: Maximum cards to return.

        Returns:
            Up to `limit` skill cards, highest relevance first.
        """
        candidates = self._skills.recall(self._query(query, limit, domain))
        return [SkillCard.model_validate(c.entry, from_attributes=True) for c in candidates]

    @validate_call
    def find_tools(self, query: TaskBrief, limit: RecallLimit = DEFAULT_SLATE) -> list[ToolCard]:
        """The top-ranked tool cards for a task.

        Args:
            query: The task to match tools against.
            limit: Maximum cards to return.

        Returns:
            Up to `limit` tool cards, highest relevance first.
        """
        candidates = self._tools.recall(self._query(query, limit))
        return [ToolCard.model_validate(c.entry, from_attributes=True) for c in candidates]

    @validate_call
    def find_mcp(self, query: TaskBrief, limit: RecallLimit = DEFAULT_SLATE) -> list[McpCard]:
        """The top-ranked MCP server cards for a task.

        Args:
            query: The task to match servers against.
            limit: Maximum cards to return.

        Returns:
            Up to `limit` MCP cards, highest relevance first; each card lists the tool names most
            relevant to the task.
        """
        candidates = self._mcp.recall(self._query(query, limit))
        cards: list[McpCard] = []
        for c in candidates:
            server = c.entry
            if not isinstance(server, CatalogMcpServer):
                continue
            cards.append(
                McpCard(
                    id=server.id,
                    description=server.description,
                    tags=server.tags,
                    ref=server.ref,
                    tool_count=len(server.tools),
                    relevant_tools=relevant_tool_names(query, server.tools, MCP_CARD_TOOLS),
                )
            )
        return cards

    @validate_call
    def find_plugins(
        self, query: TaskBrief, limit: RecallLimit = DEFAULT_SLATE
    ) -> list[PluginCard]:
        """The top-ranked plugin cards for a task.

        Args:
            query: The task to match plugins against.
            limit: Maximum cards to return.

        Returns:
            Up to `limit` plugin cards, highest relevance first.
        """
        candidates = self._plugins.recall(self._query(query, limit))
        return [PluginCard.model_validate(c.entry, from_attributes=True) for c in candidates]

    @validate_call
    def load_skill(self, skill_id: CatalogEntryId) -> str:
        """Read the full SKILL.md body for an indexed skill id.

        Args:
            skill_id: A skill id from a `find_skills` card.

        Returns:
            The skill's SKILL.md contents.

        Raises:
            KeyError: If no indexed skill has that id.
        """
        path = self._paths.get(skill_id)
        if path is None:
            raise KeyError(f"unknown skill id: {skill_id!r}")
        return path.read_text(encoding="utf-8")

    @validate_call
    def describe_mcp(self, server: McpServerRef) -> list[McpTool]:
        """The full tool list — names, params, and input schemas — for one connected MCP
        server returned by find_mcp; the on-demand detail behind a server card."""
        card = self._mcp_by_ref.get(server)
        return list(card.tools) if card is not None else []
