from __future__ import annotations

from functools import cache

from fastmcp import FastMCP

from capdisc.catalog import (
    CatalogEntryId,
    McpServerRef,
    McpTool,
    RecallLimit,
    Tag,
)
from capdisc.mcp_catalog import enumerate_mcp_servers
from capdisc.mcp_harvest import (
    cache_is_stale,
    read_mcp_cache,
    refresh_in_background,
)
from capdisc.plugin_catalog import enumerate_plugins
from capdisc.report import write_report_on_start

from ..models.agent import TaskBrief
from ..settings import current_discovery_scope
from .capability import DEFAULT_SLATE, CapabilityRouter, McpCard, PluginCard, SkillCard, ToolCard

mcp = FastMCP("skill-router")


@cache
def _router() -> CapabilityRouter:
    """Build the process-wide capability router, once, on first use.

    Lazily built (and cached) so importing this module does not scan the filesystem or shell out.
    This is the runtime boundary that resolves real launch paths: project scope walks up from the
    cwd, user scope from the home dir, managed scope from the OS policy dir, and plugin-bundled
    skills from the installed-plugin dirs.

    Returns:
        The router, wired from the discovered scope roots, the tool-enriched MCP cache (falling
        back to the name-only live list), and the installed plugins.
    """
    scope = current_discovery_scope()
    plugins_root = scope.plugins_root
    roots = scope.roots()
    # prefer the tool-enriched cache (built offline by `refresh_mcp_cache`) so find_mcp ranks on
    # what each server's tools do; fall back to the name-only live list when there is no cache
    mcp_servers = read_mcp_cache() or enumerate_mcp_servers()
    # stale-while-revalidate: serve the current cache now and, if it is missing or aged out, harvest
    # a fresh one in the background for the next process — never blocking this build on network I/O
    if cache_is_stale():
        refresh_in_background(plugins_root=plugins_root)
    write_report_on_start()
    return CapabilityRouter.from_environment(
        roots, mcp_servers=mcp_servers, plugins=enumerate_plugins(plugins_root)
    )


@mcp.tool
def find_skills(
    query: TaskBrief, limit: RecallLimit = DEFAULT_SLATE, tags: list[Tag] | None = None
) -> list[SkillCard]:
    """Search installed skills and return the few most relevant to a task, ranked — the
    deferred-tool alternative to registering every skill into context.

    Args:
        query: A short description of the task to match skills against.
        limit: Maximum number of cards to return.
        tags: Tags to narrow by first before ranking by description; omit to rank purely by
            description.

    Returns:
        A small slate of skill cards; call load_skill on the ids you actually need.
    """
    return _router().find_skills(query, limit, tags)


@mcp.tool
def find_tools(query: TaskBrief, limit: RecallLimit = DEFAULT_SLATE) -> list[ToolCard]:
    """Search the available tools and return the few most relevant to a task, ranked.

    Args:
        query: A short description of the task to match tools against.
        limit: Maximum number of cards to return.

    Returns:
        Tool cards, each carrying the grant ref (e.g. 'Read', 'Bash(git log:*)') used to equip
        the tool on an agent.
    """
    return _router().find_tools(query, limit)


@mcp.tool
def find_mcp(query: TaskBrief, limit: RecallLimit = DEFAULT_SLATE) -> list[McpCard]:
    """Search the connected MCP servers and return the few most relevant to a task, ranked.

    Args:
        query: A short description of the task to match servers against.
        limit: Maximum number of cards to return.

    Returns:
        MCP cards, each carrying the server name; its tools are reachable as 'mcp__<server>__*'.
    """
    return _router().find_mcp(query, limit)


@mcp.tool
def find_plugins(query: TaskBrief, limit: RecallLimit = DEFAULT_SLATE) -> list[PluginCard]:
    """Search the installed plugins and return the few most relevant to a task, ranked.

    Args:
        query: A short description of the task to match plugins against.
        limit: Maximum number of cards to return.

    Returns:
        Plugin cards, each carrying the plugin name and what it bundles (skills, commands,
        agents, MCP servers).
    """
    return _router().find_plugins(query, limit)


@mcp.tool
def load_skill(skill_id: CatalogEntryId) -> str:
    """Return the full SKILL.md body for a skill.

    Args:
        skill_id: A skill id from a find_skills card.

    Returns:
        The skill's full SKILL.md body.
    """
    return _router().load_skill(skill_id)


@mcp.tool
def describe_mcp(server: McpServerRef) -> list[McpTool]:
    """Return one connected MCP server's tools with full input schemas, for a server name
    from find_mcp. Use after find_mcp to inspect a server's exact tool contracts."""
    return _router().describe_mcp(server)


def main() -> None:
    """Run the skill-router MCP server over stdio (the `skill-router` console entry point)."""
    mcp.run()


if __name__ == "__main__":
    main()
