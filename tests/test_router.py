from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_fleet import CatalogMcpServer, CatalogPlugin
from agent_fleet.router.capability import MCP_CARD_TOOLS, CapabilityRouter
from capabilities_discovery.catalog import McpTool
from capabilities_discovery.scope import ScopeRoots
from helpers import write_skill


def _playwright() -> CatalogMcpServer:
    return CatalogMcpServer(
        id="mcp.playwright",
        ref="playwright",
        description="Browser automation server driving a real browser.",
        tools=[
            McpTool(
                name="browser_navigate",
                description="Navigate the browser to a url and wait for load.",
                params=["url"],
            ),
            McpTool(
                name="browser_click",
                description="Click an element on the page.",
                params=["element", "ref"],
            ),
            McpTool(
                name="browser_take_screenshot",
                description="Take a screenshot of the current page.",
                params=["filename"],
            ),
            McpTool(
                name="browser_type",
                description="Type text into an editable element.",
                params=["element", "text"],
            ),
        ],
    )


def _corpus(root: Path) -> None:
    write_skill(
        root,
        "vuln-auditor",
        "---\nname: vuln-auditor\n"
        "description: Audit code for security vulnerabilities.\n"
        "tags: [security, audit]\n---\n\nAudit the code, then stop.",
    )
    write_skill(
        root,
        "doc-writer",
        "---\nname: doc-writer\n"
        "description: Write and publish project documentation.\n"
        "tags: [documentation, docs]\n---\n\nWrite the docs, then stop.",
    )


def _router(
    root: Path,
    mcp_servers: list[CatalogMcpServer] | None = None,
    plugins: list[CatalogPlugin] | None = None,
) -> CapabilityRouter:
    (root / ".git").mkdir(exist_ok=True)  # bound the project walk-up at this dir
    roots = ScopeRoots.discover(start=root)
    return CapabilityRouter.from_environment(
        roots, mcp_servers=mcp_servers or [], plugins=plugins or []
    )


def test_find_skills_ranks_by_relevance(tmp_path: Path) -> None:
    _corpus(tmp_path)
    cards = _router(tmp_path).find_skills("audit the code for vulnerabilities")
    assert cards[0].ref == "vuln-auditor"


def test_find_skills_respects_limit(tmp_path: Path) -> None:
    _corpus(tmp_path)
    assert len(_router(tmp_path).find_skills("write and publish documentation", limit=1)) == 1


def test_find_tools_surfaces_builtin(tmp_path: Path) -> None:
    _corpus(tmp_path)
    refs = [card.ref for card in _router(tmp_path).find_tools("read a file from disk", limit=3)]
    assert "Read" in refs


def test_find_mcp_returns_passed_servers(tmp_path: Path) -> None:
    _corpus(tmp_path)
    server = CatalogMcpServer(
        id="mcp.playwright",
        ref="playwright",
        description="Browser automation server driving a real browser.",
    )
    cards = _router(tmp_path, [server]).find_mcp("drive a browser to automate the web", limit=3)
    assert cards[0].ref == "playwright"


def test_find_mcp_card_reports_tool_count(tmp_path: Path) -> None:
    _corpus(tmp_path)
    server = _playwright()
    cards = _router(tmp_path, [server]).find_mcp("drive a browser to automate the web", limit=3)
    assert cards[0].ref == "playwright"
    assert cards[0].tool_count == len(server.tools)


def test_find_mcp_card_ranks_relevant_tools(tmp_path: Path) -> None:
    _corpus(tmp_path)
    cards = _router(tmp_path, [_playwright()]).find_mcp("click an element on the page", limit=3)
    relevant = cards[0].relevant_tools
    assert relevant.index("browser_click") < relevant.index("browser_take_screenshot")


def test_find_mcp_card_caps_relevant_tools(tmp_path: Path) -> None:
    _corpus(tmp_path)
    cards = _router(tmp_path, [_playwright()]).find_mcp("click an element on the page", limit=3)
    assert len(cards[0].relevant_tools) <= MCP_CARD_TOOLS


def test_find_mcp_is_deterministic(tmp_path: Path) -> None:
    _corpus(tmp_path)
    router = _router(tmp_path, [_playwright()])
    query = "click an element on the page"
    assert router.find_mcp(query, limit=3) == router.find_mcp(query, limit=3)


def test_find_mcp_card_falls_back_when_no_tool_overlap(tmp_path: Path) -> None:
    _corpus(tmp_path)
    server = _playwright()
    cards = _router(tmp_path, [server]).find_mcp("compile firmware bytecode quickly", limit=3)
    relevant = cards[0].relevant_tools
    assert relevant
    assert relevant == [t.name for t in server.tools[:MCP_CARD_TOOLS]]


def test_find_plugins_returns_passed_plugins(tmp_path: Path) -> None:
    _corpus(tmp_path)
    plugin = CatalogPlugin(
        id="plugin.agentforge",
        ref="agentforge",
        description="Deterministic generator for Claude Code agents, skills, and tools.",
    )
    cards = _router(tmp_path, plugins=[plugin]).find_plugins("generate an agent", limit=3)
    assert cards[0].ref == "agentforge"


def test_load_skill_returns_body(tmp_path: Path) -> None:
    _corpus(tmp_path)
    assert "Audit the code, then stop." in _router(tmp_path).load_skill("skill.vuln-auditor")


def test_load_skill_unknown_id_raises(tmp_path: Path) -> None:
    _corpus(tmp_path)
    with pytest.raises(KeyError):
        _router(tmp_path).load_skill("skill.does-not-exist")


def test_load_skill_rejects_pattern_invalid_id(tmp_path: Path) -> None:
    # @validate_call enforces the CatalogEntryId pattern before the path lookup
    _corpus(tmp_path)
    with pytest.raises(ValidationError):
        _router(tmp_path).load_skill("BAD ID WITH SPACES!!")


def test_describe_mcp_returns_tools_with_input_schema(tmp_path: Path) -> None:
    from capabilities_discovery.catalog import McpTool

    schema = {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}
    server = CatalogMcpServer(
        id="mcp.playwright",
        ref="playwright",
        description="Browser automation server driving a real browser.",
        tools=[
            McpTool(name="browser_navigate", description="Navigate to a URL.", input_schema=schema)
        ],
    )
    tools = _router(tmp_path, [server]).describe_mcp("playwright")
    assert len(tools) == 1
    assert tools[0].name == "browser_navigate"
    assert tools[0].input_schema == schema


def test_describe_mcp_returns_empty_for_unknown_ref(tmp_path: Path) -> None:
    assert _router(tmp_path).describe_mcp("does-not-exist") == []
