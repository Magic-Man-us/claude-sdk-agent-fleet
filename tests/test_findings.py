from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions, McpSdkServerConfig
from claude_agent_sdk.types import AgentDefinition
from mcp.types import CallToolRequest, CallToolRequestParams, ServerResult

from agent_fleet import (
    AgentPool,
    AgentSpec,
    AsyncAgentPool,
    Finding,
    build_findings_server,
    grant_findings_to_subagent,
    with_findings_tool,
)
from agent_fleet.engine.findings_tool import FINDINGS_SERVER, WRITE_FINDING_TOOL
from agent_fleet.engine.render import to_agent_definition, to_options

_PROMPT = "You are auditor. Audit the code for vulnerabilities and stop."
_AGENT_KEY = "PROJ-4821"
_TASK = "audit the codebase for security vulnerabilities now"


def _spec() -> AgentSpec:
    return AgentSpec(
        name="auditor",
        description="Audits code for vulnerabilities.",
        system_prompt=_PROMPT,
        tools=("Read", "Grep"),
    )


def _pool(tmp_path: Path) -> AgentPool:
    return AgentPool(tmp_path / "pool.db")


def _new_run(pool: AgentPool) -> str:
    pool.save(_AGENT_KEY, _spec())
    return pool.start_run(_AGENT_KEY, _TASK).run_id


async def _call_write_finding(server: McpSdkServerConfig, content: str) -> ServerResult:
    handler = server["instance"].request_handlers[CallToolRequest]
    request = CallToolRequest(
        method="tools/call",
        params=CallToolRequestParams(name="write_finding", arguments={"content": content}),
    )
    return await handler(request)


def test_record_and_list_findings_round_trip_oldest_first(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    run_id = _new_run(pool)
    session = str(uuid.uuid4())

    first = pool.record_finding(_AGENT_KEY, run_id, session, "first finding")
    second = pool.record_finding(
        _AGENT_KEY, run_id, session, "second finding", agent_name="lens-security"
    )

    assert isinstance(first, Finding)
    assert first.agent_name is None  # main/supervisor
    assert second.agent_name == "lens-security"

    findings = pool.list_findings(_AGENT_KEY)
    assert [f.content for f in findings] == ["first finding", "second finding"]  # oldest-first


def test_list_findings_filters_by_run(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    pool.save(_AGENT_KEY, _spec())
    run_a = pool.start_run(_AGENT_KEY, _TASK).run_id
    run_b = pool.start_run(_AGENT_KEY, _TASK).run_id
    session = str(uuid.uuid4())

    pool.record_finding(_AGENT_KEY, run_a, session, "from run a")
    pool.record_finding(_AGENT_KEY, run_b, session, "from run b")

    assert [f.content for f in pool.list_findings(_AGENT_KEY, run_id=run_a)] == ["from run a"]
    assert [f.content for f in pool.list_findings(_AGENT_KEY, run_id=run_b)] == ["from run b"]
    assert len(pool.list_findings(_AGENT_KEY)) == 2


def test_concurrent_record_finding_loses_nothing(tmp_path: Path) -> None:
    async_pool = AsyncAgentPool(_pool(tmp_path))
    run_id = _new_run(async_pool.pool)
    session = str(uuid.uuid4())
    lenses = [f"lens-{i}" for i in range(24)]

    async def scenario() -> None:
        await asyncio.gather(
            *(
                async_pool.record_finding(
                    _AGENT_KEY, run_id, session, f"finding from {name}", agent_name=name
                )
                for name in lenses
            )
        )

    asyncio.run(scenario())

    findings = asyncio.run(async_pool.list_findings(_AGENT_KEY))
    assert len(findings) == len(lenses)  # every concurrent write landed
    assert {f.agent_name for f in findings} == set(lenses)  # none clobbered
    assert {f.content for f in findings} == {f"finding from {name}" for name in lenses}


def test_build_findings_server_returns_findings_config(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    run_id = _new_run(pool)
    server = build_findings_server(pool, _AGENT_KEY, run_id, str(uuid.uuid4()), None)
    assert server["type"] == "sdk"
    assert server["name"] == FINDINGS_SERVER


def test_with_findings_tool_mounts_server_and_grants_tool(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    run_id = _new_run(pool)
    base = to_options(_spec())
    assert WRITE_FINDING_TOOL not in base.allowed_tools

    options = with_findings_tool(base, pool, _AGENT_KEY, run_id, str(uuid.uuid4()))

    assert FINDINGS_SERVER in options.mcp_servers
    assert WRITE_FINDING_TOOL in options.allowed_tools
    assert isinstance(options, ClaudeAgentOptions)
    assert WRITE_FINDING_TOOL not in base.allowed_tools  # original untouched


def test_with_findings_tool_preserves_existing_servers_and_does_not_duplicate(
    tmp_path: Path,
) -> None:
    pool = _pool(tmp_path)
    run_id = _new_run(pool)
    session = str(uuid.uuid4())
    once = with_findings_tool(to_options(_spec()), pool, _AGENT_KEY, run_id, session)
    twice = with_findings_tool(once, pool, _AGENT_KEY, run_id, session)
    assert twice.allowed_tools.count(WRITE_FINDING_TOOL) == 1
    assert FINDINGS_SERVER in twice.mcp_servers


def test_write_finding_handler_inserts_a_retrievable_row(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    run_id = _new_run(pool)
    session = str(uuid.uuid4())
    server = build_findings_server(pool, _AGENT_KEY, run_id, session, "lens-performance")

    result = asyncio.run(_call_write_finding(server, "N+1 query in the report loop"))
    assert result.root.content[0].text == "recorded"

    findings = pool.list_findings(_AGENT_KEY)
    assert len(findings) == 1
    assert findings[0].content == "N+1 query in the report loop"
    assert findings[0].agent_name == "lens-performance"
    assert findings[0].session_id == session


def test_grant_findings_to_subagent_appends_server_and_tool() -> None:
    definition = to_agent_definition(_spec())  # tools is a real list (has Read, Grep)
    assert definition.tools is not None
    granted = grant_findings_to_subagent(definition)

    assert granted.mcpServers is not None
    assert FINDINGS_SERVER in granted.mcpServers
    assert granted.tools is not None
    assert WRITE_FINDING_TOOL in granted.tools
    assert WRITE_FINDING_TOOL not in (definition.tools or [])  # original untouched


def test_grant_findings_to_subagent_is_idempotent() -> None:
    granted_once = grant_findings_to_subagent(to_agent_definition(_spec()))
    granted_twice = grant_findings_to_subagent(granted_once)
    assert granted_twice.mcpServers is not None
    assert granted_twice.tools is not None
    assert granted_twice.mcpServers.count(FINDINGS_SERVER) == 1
    assert granted_twice.tools.count(WRITE_FINDING_TOOL) == 1


def test_grant_findings_to_subagent_leaves_inherit_all_tools_none() -> None:
    inherit_all = AgentDefinition(
        description="Inherits every tool.",
        prompt="You inherit everything and stop now.",
        tools=None,
    )
    granted = grant_findings_to_subagent(inherit_all)
    assert granted.tools is None  # "inherit everything" not narrowed to an explicit list
    assert granted.mcpServers == [FINDINGS_SERVER]


def test_async_record_and_list_findings_through_wrapper(tmp_path: Path) -> None:
    async_pool = AsyncAgentPool(_pool(tmp_path))
    run_id = _new_run(async_pool.pool)
    session = str(uuid.uuid4())

    async def scenario() -> list[Finding]:
        await async_pool.record_finding(_AGENT_KEY, run_id, session, "async finding")
        return await async_pool.list_findings(_AGENT_KEY, run_id=run_id)

    findings = asyncio.run(scenario())
    assert [f.content for f in findings] == ["async finding"]
