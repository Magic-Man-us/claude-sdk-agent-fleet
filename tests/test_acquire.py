from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from claude_agent_sdk import ClaudeAgentOptions, McpSdkServerConfig, Message
from claude_agent_sdk.types import AgentDefinition
from mcp.types import CallToolRequest, CallToolRequestParams, ServerResult

from agent_fleet import (
    AgentPool,
    AgentSpec,
    build_acquire_server,
    grant_acquire_to_subagent,
    with_acquire_tool,
)
from agent_fleet.engine.acquire_tool import ACQUIRE_SERVER, ACQUIRE_TOOL, acquired_agent_key
from agent_fleet.engine.render import to_agent_definition, to_options
from agent_fleet.router.capability import CapabilityRouter
from capabilities_discovery.catalog import CatalogMcpServer, CatalogSkill, CatalogTool, McpTool
from test_dispatch import _assistant, _fake_query

_AGENT_KEY = "PROJ-4821"
_NEED = "drive a headless browser to automate web pages and take screenshots"
_TASK = "open the login page, sign in, and screenshot the dashboard"


def _router() -> CapabilityRouter:
    tools = [
        CatalogTool(
            id="tool.web-fetch",
            ref="WebFetch",
            description="Fetch a web page over http and return its contents.",
        ),
        CatalogTool(
            id="tool.read",
            ref="Read",
            description="Read a file from disk.",
        ),
    ]
    skills = [
        CatalogSkill(
            id="skill.browser-automation",
            ref="browser-automation",
            description="Drive a headless browser to automate web pages and take screenshots.",
            tags=["web"],
        ),
        CatalogSkill(
            id="skill.pdf-extract",
            ref="pdf-extract",
            description="Extract text and tables from a PDF document.",
            tags=["documentation"],
        ),
    ]
    mcp_servers = [
        CatalogMcpServer(
            id="mcp.playwright",
            ref="playwright",
            description="Browser automation server driving a real browser to automate web pages.",
            tools=[
                McpTool(
                    name="browser_navigate",
                    description="Navigate the browser to a url.",
                    params=["url"],
                )
            ],
        ),
    ]
    return CapabilityRouter(skills, tools, mcp_servers, [], {})


def _pool(tmp_path: Path) -> AgentPool:
    return AgentPool(tmp_path / "pool.db")


def _fake_query_capturing(messages: list[Message], captured: list[ClaudeAgentOptions]) -> object:
    async def _query(**kwargs: object) -> AsyncIterator[Message]:
        options = kwargs["options"]
        assert isinstance(options, ClaudeAgentOptions)
        captured.append(options)
        for message in messages:
            yield message

    return _query


async def _call_acquire(server: McpSdkServerConfig, need: str, task: str) -> ServerResult:
    handler = server["instance"].request_handlers[CallToolRequest]
    request = CallToolRequest(
        method="tools/call",
        params=CallToolRequestParams(
            name="acquire_capability", arguments={"need": need, "task": task}
        ),
    )
    return await handler(request)


def test_build_acquire_server_returns_acquire_config(tmp_path: Path) -> None:
    server = build_acquire_server(_router(), _pool(tmp_path), _AGENT_KEY)
    assert server["type"] == "sdk"
    assert server["name"] == ACQUIRE_SERVER


def test_acquire_capability_equips_and_runs_matching_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool = _pool(tmp_path)
    session = str(uuid.uuid4())
    messages: list[Message] = [
        _assistant("navigating", session_id=session),
        _assistant("done: screenshot saved", session_id=session),
    ]
    captured: list[ClaudeAgentOptions] = []
    monkeypatch.setattr(
        "agent_fleet.engine.dispatch.query", _fake_query_capturing(messages, captured)
    )

    server = build_acquire_server(_router(), pool, _AGENT_KEY)
    result = asyncio.run(_call_acquire(server, _NEED, _TASK))

    assert "navigating" in result.root.content[0].text
    assert "done: screenshot saved" in result.root.content[0].text

    assert len(captured) == 1
    options = captured[0]
    assert "WebFetch" in options.allowed_tools  # matched tool granted
    assert options.skills is not None
    assert "browser-automation" in options.skills  # matched skill granted
    assert "mcp__playwright__*" in options.allowed_tools  # matched mcp server granted


def test_acquired_run_gets_its_own_resumable_pool_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool = _pool(tmp_path)
    assert pool.list_runs(_AGENT_KEY) == []
    derived = acquired_agent_key(_AGENT_KEY, _NEED)
    session = str(uuid.uuid4())
    messages: list[Message] = [_assistant("result", session_id=session)]
    monkeypatch.setattr("agent_fleet.engine.dispatch.query", _fake_query(messages))

    server = build_acquire_server(_router(), pool, _AGENT_KEY)
    asyncio.run(_call_acquire(server, _NEED, _TASK))

    entry = pool.get_by_key(derived)
    assert entry is not None  # a new, listable pool entry under the derived id
    assert entry.session_id == session  # reconciled to the session it actually ran under
    assert len(pool.list()) == 1  # the acquired agent is the only pooled entry

    runs = pool.list_runs(derived)
    assert len(runs) == 1  # a top-level run recorded under the acquired agent's own id
    assert runs[0].agent_key == derived
    assert runs[0].task == _TASK
    assert pool.list_runs(_AGENT_KEY) == []  # nothing recorded under the caller's agent key


def test_re_acquiring_same_need_resumes_the_same_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool = _pool(tmp_path)
    derived = acquired_agent_key(_AGENT_KEY, _NEED)
    server = build_acquire_server(_router(), pool, _AGENT_KEY)

    first_session = str(uuid.uuid4())
    monkeypatch.setattr(
        "agent_fleet.engine.dispatch.query",
        _fake_query([_assistant("first", session_id=first_session)]),
    )
    asyncio.run(_call_acquire(server, _NEED, _TASK))
    first_entry = pool.get_by_key(derived)
    assert first_entry is not None
    assert first_entry.session_id == first_session

    second_session = str(uuid.uuid4())
    monkeypatch.setattr(
        "agent_fleet.engine.dispatch.query",
        _fake_query([_assistant("second", session_id=second_session)]),
    )
    asyncio.run(_call_acquire(server, _NEED, _TASK))

    entry = pool.get_by_key(derived)
    assert entry is not None
    assert entry.session_id == second_session  # resumed and repinned, not duplicated
    assert len(pool.list()) == 1  # same derived id — one entry, not two
    assert len(pool.list_runs(derived)) == 2  # both runs live under the acquired agent's id


def _spec() -> AgentSpec:
    return AgentSpec(
        name="auditor",
        description="Audits code for vulnerabilities.",
        system_prompt="You are auditor. Audit the code and stop.",
        tools=("Read", "Grep"),
    )


def test_with_acquire_tool_mounts_server_and_grants_tool(tmp_path: Path) -> None:
    base = to_options(_spec())
    assert ACQUIRE_TOOL not in base.allowed_tools

    options = with_acquire_tool(base, _router(), _pool(tmp_path), _AGENT_KEY)

    assert ACQUIRE_SERVER in options.mcp_servers
    assert ACQUIRE_TOOL in options.allowed_tools
    assert isinstance(options, ClaudeAgentOptions)
    assert ACQUIRE_TOOL not in base.allowed_tools  # original untouched


def test_with_acquire_tool_preserves_existing_and_does_not_duplicate(tmp_path: Path) -> None:
    router = _router()
    pool = _pool(tmp_path)
    once = with_acquire_tool(to_options(_spec()), router, pool, _AGENT_KEY)
    twice = with_acquire_tool(once, router, pool, _AGENT_KEY)
    assert twice.allowed_tools.count(ACQUIRE_TOOL) == 1
    assert ACQUIRE_SERVER in twice.mcp_servers


def test_grant_acquire_to_subagent_appends_server_and_tool() -> None:
    definition = to_agent_definition(_spec())  # tools is a real list (has Read, Grep)
    assert definition.tools is not None
    granted = grant_acquire_to_subagent(definition)

    assert granted.mcpServers is not None
    assert ACQUIRE_SERVER in granted.mcpServers
    assert granted.tools is not None
    assert ACQUIRE_TOOL in granted.tools
    assert ACQUIRE_TOOL not in (definition.tools or [])  # original untouched


def test_grant_acquire_to_subagent_is_idempotent() -> None:
    granted_once = grant_acquire_to_subagent(to_agent_definition(_spec()))
    granted_twice = grant_acquire_to_subagent(granted_once)
    assert granted_twice.mcpServers is not None
    assert granted_twice.tools is not None
    assert granted_twice.mcpServers.count(ACQUIRE_SERVER) == 1
    assert granted_twice.tools.count(ACQUIRE_TOOL) == 1


def test_grant_acquire_to_subagent_leaves_inherit_all_tools_none() -> None:
    inherit_all = AgentDefinition(
        description="Inherits every tool.",
        prompt="You inherit everything and stop now.",
        tools=None,
    )
    granted = grant_acquire_to_subagent(inherit_all)
    assert granted.tools is None  # "inherit everything" not narrowed to an explicit list
    assert granted.mcpServers == [ACQUIRE_SERVER]
