from __future__ import annotations

from agent_fleet import Catalog, CatalogMcpServer, InMemoryCatalogSource, RecallQuery
from capabilities_discovery.catalog import McpTool

_BROWSER_QUERY = "drive a browser to automate the web"


def _recall(catalog: Catalog, text: str) -> list:
    query = RecallQuery(text=text, limit=10)
    return InMemoryCatalogSource(catalog).recall(query)


def test_name_only_server_does_not_match_a_tool_query() -> None:
    # the old behaviour: with no harvested tools, "browser" matches nothing in name/description
    playwright = CatalogMcpServer(
        id="mcp.playwright", ref="playwright", description="An MCP server."
    )
    [card] = _recall(Catalog(entries=[playwright]), _BROWSER_QUERY)
    assert card.score == 0.0


def test_harvested_tools_make_the_server_rank() -> None:
    # the fix: folding each tool's name/params/description into the server's search text lets a
    # task match what the tools do — playwright wins on its browser_* tools over an unrelated server
    playwright = CatalogMcpServer(
        id="mcp.playwright",
        ref="playwright",
        description="An MCP server.",
        tools=[
            McpTool(
                name="browser_navigate",
                description="Navigate the browser to a url and wait for load",
                params=["url"],
            ),
            McpTool(
                name="browser_click",
                description="Click an element on the page",
                params=["element"],
            ),
        ],
    )
    notion = CatalogMcpServer(
        id="mcp.notion",
        ref="notion",
        description="An MCP server.",
        tools=[
            McpTool(name="create_page", description="Create a documentation page", params=["title"])
        ],
    )
    ranked = _recall(Catalog(entries=[notion, playwright]), _BROWSER_QUERY)
    assert ranked[0].entry.id == "mcp.playwright"
    assert ranked[0].score > 0.0
