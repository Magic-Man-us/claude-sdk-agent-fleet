from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastmcp import Client

from agent_fleet import CatalogMcpServer, CatalogPlugin
from agent_fleet.router import mcp_server
from agent_fleet.router.capability import CapabilityRouter
from capabilities_discovery.scope import ScopeRoots
from helpers import write_skill


def _corpus(root: Path) -> None:
    write_skill(
        root,
        "vuln-auditor",
        "---\nname: vuln-auditor\n"
        "description: Audit code for security vulnerabilities.\n"
        "domain: security\ntags: [audit]\n---\n\nAudit the code, then stop.",
    )


def test_server_module_imports() -> None:
    # importing runs the @mcp.tool decorators; a clean import means the tools registered
    # without scanning the filesystem or shelling out (the router is lazy).
    assert mcp_server.mcp is not None
    assert callable(mcp_server.main)


def test_all_capability_tools_over_mcp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # a real MCP roundtrip exercising every deferred-tool entry point over the actual protocol
    _corpus(tmp_path)
    server = CatalogMcpServer(
        id="mcp.playwright",
        ref="playwright",
        description="Browser automation server driving a real browser.",
    )
    plugin = CatalogPlugin(
        id="plugin.agentforge",
        ref="agentforge",
        description="Deterministic generator for Claude Code agents, skills, and tools.",
    )
    (tmp_path / ".git").mkdir(exist_ok=True)  # bound the project walk-up at this dir
    roots = ScopeRoots.discover(start=tmp_path)
    router = CapabilityRouter.from_environment(roots, mcp_servers=[server], plugins=[plugin])
    monkeypatch.setattr(mcp_server, "_router", lambda: router)

    async def _roundtrip():
        async with Client(mcp_server.mcp) as client:
            names = {tool.name for tool in await client.list_tools()}
            skills = await client.call_tool("find_skills", {"query": "audit for vulnerabilities"})
            tools = await client.call_tool("find_tools", {"query": "read a file from disk"})
            servers = await client.call_tool("find_mcp", {"query": "drive a browser around"})
            plugins = await client.call_tool("find_plugins", {"query": "generate an agent"})
            body = await client.call_tool("load_skill", {"skill_id": "skill.vuln-auditor"})
            return names, skills.data, tools.data, servers.data, plugins.data, body.data

    names, skills, tools, servers, plugins, body = asyncio.run(_roundtrip())
    assert names == {
        "find_skills",
        "find_tools",
        "find_mcp",
        "find_plugins",
        "load_skill",
        "describe_mcp",
    }
    assert skills[0].ref == "vuln-auditor"
    assert "Read" in [card.ref for card in tools]
    assert servers[0].ref == "playwright"
    assert plugins[0].ref == "agentforge"
    assert "Audit the code, then stop." in body
