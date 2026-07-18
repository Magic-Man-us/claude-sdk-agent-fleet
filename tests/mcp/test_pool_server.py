from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from claude_agent_sdk import ClaudeAgentOptions, Message

from agent_fleet import AgentPool, RunOutcome
from agent_fleet.engine import acquire_tool, findings_tool
from agent_fleet.engine.render import SEND_MESSAGE_TOOL, SUBAGENT_TOOL
from agent_fleet.engine.source import InMemoryCatalogSource
from agent_fleet.models.agent import ModelId
from agent_fleet.router.capability import CapabilityRouter
from agent_fleet_mcp import pool_server
from capabilities_discovery.catalog import Catalog
from test_dispatch import (
    _assistant,
    _dispatch,
    _task_started,
)
from test_dispatch import (
    _fake_query_capturing as _capture_kwargs,
)

_AGENT_KEY = "PROJ-4821"
_SUB_AGENT_KEY = "PROJ-9001"
_TASK = "audit the codebase for security vulnerabilities now"
_SUB_TASK = "review the diffs for correctness and regressions"


def _fake_query(messages: list[Message]) -> object:
    async def _query(**kwargs: object) -> AsyncIterator[Message]:
        for message in messages:
            yield message

    return _query


def _fake_query_capturing(messages: list[Message], captured: list[ClaudeAgentOptions]) -> object:
    async def _query(**kwargs: object) -> AsyncIterator[Message]:
        options = kwargs["options"]
        assert isinstance(options, ClaudeAgentOptions)
        captured.append(options)
        for message in messages:
            yield message

    return _query


@pytest.fixture
def pool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AgentPool:
    built = AgentPool(tmp_path / "pool.db")
    monkeypatch.setattr(pool_server, "_pool", lambda: built)
    monkeypatch.setattr(pool_server, "_source", lambda: InMemoryCatalogSource(Catalog(entries=[])))
    monkeypatch.setattr(
        pool_server, "_capability_router", lambda: CapabilityRouter([], [], [], [], {})
    )
    return built


def test_create_then_get_round_trips(pool: AgentPool) -> None:
    created = pool_server.create_agent(_AGENT_KEY, _TASK)
    assert created.agent_key == _AGENT_KEY

    fetched = pool_server.get_agent(_AGENT_KEY)
    assert fetched is not None
    assert fetched.agent_key == _AGENT_KEY
    assert fetched.session_id == created.session_id


def test_get_absent_returns_none(pool: AgentPool) -> None:
    assert pool_server.get_agent("nope") is None


def test_list_and_find(pool: AgentPool) -> None:
    pool_server.create_agent(_AGENT_KEY, _TASK)
    pool_server.create_agent(_SUB_AGENT_KEY, _SUB_TASK)

    assert {entry.agent_key for entry in pool_server.list_agents()} == {
        _AGENT_KEY,
        _SUB_AGENT_KEY,
    }
    found = pool_server.find_agents("security vulnerabilities audit")
    assert found[0].agent_key == _AGENT_KEY


def test_delete_true_then_false(pool: AgentPool) -> None:
    pool_server.create_agent(_AGENT_KEY, _TASK)
    assert pool_server.delete_agent(_AGENT_KEY) is True
    assert pool_server.delete_agent(_AGENT_KEY) is False


def test_runs_empty_before_any_run(pool: AgentPool) -> None:
    pool_server.create_agent(_AGENT_KEY, _TASK)
    assert pool_server.list_runs(_AGENT_KEY) == []


def test_findings_empty_passthrough(pool: AgentPool) -> None:
    pool_server.create_agent(_AGENT_KEY, _TASK)
    assert pool_server.list_findings(_AGENT_KEY) == []


def test_run_solo_records_a_run(pool: AgentPool, monkeypatch: pytest.MonkeyPatch) -> None:
    entry = pool_server.create_agent(_AGENT_KEY, _TASK)
    messages: list[Message] = [_assistant("done", session_id=entry.session_id)]
    monkeypatch.setattr("agent_fleet.engine.dispatch.query", _fake_query(messages))

    outcome = asyncio.run(pool_server.run_agent(_AGENT_KEY, _TASK))
    assert isinstance(outcome, RunOutcome)
    assert "done" in outcome.output

    runs = pool_server.list_runs(_AGENT_KEY)
    assert len(runs) == 1
    assert runs[0].agent_key == _AGENT_KEY

    fetched_run = pool_server.get_run(runs[0].run_id)
    assert fetched_run == runs[0]
    assert pool_server.get_run("unknown-run-id") is None


def test_run_with_subagent_wires_its_spec(pool: AgentPool, monkeypatch: pytest.MonkeyPatch) -> None:
    entry = pool_server.create_agent(_AGENT_KEY, _TASK)
    sub_entry = pool_server.create_agent(_SUB_AGENT_KEY, _SUB_TASK)
    captured: list[ClaudeAgentOptions] = []
    messages: list[Message] = [_assistant("delegating", session_id=entry.session_id)]
    monkeypatch.setattr(
        "agent_fleet.engine.dispatch.query", _fake_query_capturing(messages, captured)
    )

    asyncio.run(
        pool_server.run_agent(_AGENT_KEY, _TASK, subagent_agent_keys={"reviewer": _SUB_AGENT_KEY})
    )

    assert len(captured) == 1
    options = captured[0]
    assert options.agents is not None
    assert "reviewer" in options.agents
    assert options.agents["reviewer"].prompt == sub_entry.spec.system_prompt
    assert SUBAGENT_TOOL in options.allowed_tools


def test_run_grants_findings_and_acquire_tools(
    pool: AgentPool, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = pool_server.create_agent(_AGENT_KEY, _TASK)
    captured: list[ClaudeAgentOptions] = []
    messages: list[Message] = [_assistant("done", session_id=entry.session_id)]
    monkeypatch.setattr(
        "agent_fleet.engine.dispatch.query", _fake_query_capturing(messages, captured)
    )

    asyncio.run(pool_server.run_agent(_AGENT_KEY, _TASK))

    assert len(captured) == 1
    options = captured[0]
    assert isinstance(options.mcp_servers, dict)
    assert findings_tool.FINDINGS_SERVER in options.mcp_servers
    assert acquire_tool.ACQUIRE_SERVER in options.mcp_servers
    assert findings_tool.WRITE_FINDING_TOOL in options.allowed_tools
    assert acquire_tool.ACQUIRE_TOOL in options.allowed_tools


def test_run_grants_capability_tools_to_each_subagent(
    pool: AgentPool, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = pool_server.create_agent(_AGENT_KEY, _TASK)
    pool_server.create_agent(_SUB_AGENT_KEY, _SUB_TASK)
    captured: list[ClaudeAgentOptions] = []
    messages: list[Message] = [_assistant("delegating", session_id=entry.session_id)]
    monkeypatch.setattr(
        "agent_fleet.engine.dispatch.query", _fake_query_capturing(messages, captured)
    )

    asyncio.run(
        pool_server.run_agent(_AGENT_KEY, _TASK, subagent_agent_keys={"reviewer": _SUB_AGENT_KEY})
    )

    options = captured[0]
    assert options.agents is not None
    definition = options.agents["reviewer"]
    assert definition.mcpServers is not None
    assert findings_tool.FINDINGS_SERVER in definition.mcpServers
    assert acquire_tool.ACQUIRE_SERVER in definition.mcpServers
    assert definition.tools is not None
    assert findings_tool.WRITE_FINDING_TOOL in definition.tools
    assert acquire_tool.ACQUIRE_TOOL in definition.tools


def test_run_missing_agent_raises(pool: AgentPool) -> None:
    with pytest.raises(ValueError, match="pool entry not found: MISSING"):
        asyncio.run(pool_server.run_agent("MISSING", _TASK))


def test_run_missing_subagent_raises(pool: AgentPool) -> None:
    pool_server.create_agent(_AGENT_KEY, _TASK)
    with pytest.raises(ValueError, match="subagent pool entry not found: GHOST"):
        asyncio.run(
            pool_server.run_agent(_AGENT_KEY, _TASK, subagent_agent_keys={"reviewer": "GHOST"})
        )


def test_run_id_generation_uses_uuid(pool: AgentPool, monkeypatch: pytest.MonkeyPatch) -> None:
    entry = pool_server.create_agent(_AGENT_KEY, _TASK)
    session = str(uuid.uuid4())
    messages: list[Message] = [_assistant("x", session_id=entry.session_id)]
    monkeypatch.setattr("agent_fleet.engine.dispatch.query", _fake_query(messages))
    outcome = asyncio.run(pool_server.run_agent(_AGENT_KEY, _TASK))
    assert outcome.run.run_id != session


def test_list_agent_runs_solo(pool: AgentPool, monkeypatch: pytest.MonkeyPatch) -> None:
    entry = pool_server.create_agent(_AGENT_KEY, _TASK)
    messages: list[Message] = [_assistant("done", session_id=entry.session_id)]
    monkeypatch.setattr("agent_fleet.engine.dispatch.query", _fake_query(messages))

    outcome = asyncio.run(pool_server.run_agent(_AGENT_KEY, _TASK))

    rows = pool_server.list_agent_runs(outcome.run.run_id)
    assert rows == outcome.agent_runs
    assert len(rows) == 1
    assert rows[0].tool_use_id is None
    assert rows[0].agent_name is None
    assert rows[0].session_id == entry.session_id


def test_list_agent_runs_with_subagent(pool: AgentPool, monkeypatch: pytest.MonkeyPatch) -> None:
    entry = pool_server.create_agent(_AGENT_KEY, _TASK)
    pool_server.create_agent(_SUB_AGENT_KEY, _SUB_TASK)
    messages: list[Message] = [
        _assistant("planning", session_id=entry.session_id),
        _dispatch(entry.session_id, tool_use_id="toolu_1", subagent_type="reviewer"),
        _task_started(tool_use_id="toolu_1", task_id="abc123def456", session_id=entry.session_id),
        _assistant("sub result", session_id=entry.session_id, parent_tool_use_id="toolu_1"),
    ]
    monkeypatch.setattr("agent_fleet.engine.dispatch.query", _fake_query(messages))

    outcome = asyncio.run(
        pool_server.run_agent(_AGENT_KEY, _TASK, subagent_agent_keys={"reviewer": _SUB_AGENT_KEY})
    )

    rows = pool_server.list_agent_runs(outcome.run.run_id)
    assert rows == outcome.agent_runs
    assert len(rows) == 2
    assert rows[0].tool_use_id is None and rows[0].session_id == entry.session_id
    assert rows[0].agent_id is None
    assert rows[1].tool_use_id == "toolu_1"
    assert rows[1].agent_name == "reviewer"
    assert rows[1].agent_id == "abc123def456"
    assert rows[1].session_id == entry.session_id  # subagent shares the main session


def test_run_agent_resume_agent_id_grants_send_message_and_wraps_prompt(
    pool: AgentPool, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry = pool_server.create_agent(_AGENT_KEY, _TASK)
    agent_id = "abc123def456"
    captured: list[dict[str, object]] = []
    messages: list[Message] = [_assistant("resumed", session_id=entry.session_id)]
    monkeypatch.setattr("agent_fleet.engine.dispatch.query", _capture_kwargs(messages, captured))

    outcome = asyncio.run(pool_server.run_agent(_AGENT_KEY, _TASK, resume_agent_id=agent_id))

    assert len(captured) == 1
    options = captured[0]["options"]
    assert isinstance(options, ClaudeAgentOptions)
    assert SEND_MESSAGE_TOOL in options.allowed_tools  # SendMessage granted for the resume
    prompt = captured[0]["prompt"]
    assert isinstance(prompt, str)
    assert f"Resume agent {agent_id}" in prompt  # the wrapper turn references the subagent id
    assert _TASK in prompt  # and carries the original task text
    assert outcome.run.task == _TASK  # the recorded run task stays the caller's original task
    assert pool_server.list_runs(_AGENT_KEY)[0].task == _TASK


def test_list_agent_runs_unknown_run_is_empty(pool: AgentPool) -> None:
    assert pool_server.list_agent_runs(str(uuid.uuid4())) == []


def test_create_agent_full_parity(
    pool: AgentPool, monkeypatch: pytest.MonkeyPatch, catalog: Catalog
) -> None:
    monkeypatch.setattr(pool_server, "_source", lambda: InMemoryCatalogSource(catalog))
    override = "You are a security auditor. Report the findings you confirm and nothing else."

    created = pool_server.create_agent(
        _AGENT_KEY,
        _TASK,
        team="security-team",
        model=ModelId.opus,
        pinned=["skill.error_handling"],
        system_prompt=override,
    )

    spec = created.spec
    assert spec.model == ModelId.opus
    assert spec.system_prompt == override
    assert "error-handling" in spec.skills  # pinned id forced into the selection


def test_schema_lists_new_tool_and_full_create_params(pool: AgentPool) -> None:
    tools = asyncio.run(pool_server.mcp.list_tools())
    by_name = {tool.name: tool for tool in tools}
    assert "list_agent_runs" in by_name

    create_schema = by_name["create_agent"].parameters
    assert set(create_schema["properties"]) == {
        "agent_key",
        "task",
        "name",
        "domain",
        "tags",
        "team",
        "model",
        "pinned",
        "system_prompt",
        "reset_session",
    }
    assert set(create_schema["required"]) == {"agent_key", "task"}
