from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, TextBlock

from agent_fleet.engine.orchestrate import (
    OrchestrateOutcome,
    Orchestrator,
    build_orchestrator_server,
    collect_orchestration,
    orchestrator_options,
)
from agent_fleet.models.agent import AgentSpec
from agent_fleet.router.capability import CapabilityRouter
from capdisc.catalog import CatalogMcpServer, McpTool
from capdisc.scope import ScopeRoots


def _write_skill(root: Path, name: str, body: str) -> None:
    skill_dir = root / ".claude" / "skills" / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(body, encoding="utf-8")


def _corpus(root: Path) -> None:
    _write_skill(
        root,
        "vuln-auditor",
        "---\nname: vuln-auditor\n"
        "description: Audit code for security vulnerabilities.\n"
        "tags: [security, audit]\n---\n\nAudit the code, then stop.",
    )


def _router(root: Path, mcp_servers: list[CatalogMcpServer] | None = None) -> CapabilityRouter:
    (root / ".git").mkdir(exist_ok=True)
    roots = ScopeRoots.discover(start=root)
    return CapabilityRouter.from_environment(roots, mcp_servers=mcp_servers or [])


def _make_orch(root: Path, mcp_servers: list[CatalogMcpServer] | None = None) -> Orchestrator:
    return Orchestrator(_router(root, mcp_servers))


_VALID_TASK = "audit the codebase for security vulnerabilities now"
_VALID_PROMPT = (
    "You are a security auditor. Examine the code for vulnerabilities and report findings."
)


def test_propose_returns_valid_spec(tmp_path: Path) -> None:
    orch = _make_orch(tmp_path)
    spec = orch.propose(
        name="security-auditor",
        description="Audits code for security vulnerabilities.",
        system_prompt=_VALID_PROMPT,
    )
    assert isinstance(spec, AgentSpec)
    assert spec.name == "security-auditor"
    assert spec.skills == []
    assert spec.tools == []
    assert spec.mcp_servers == []


def test_propose_stores_spec(tmp_path: Path) -> None:
    orch = _make_orch(tmp_path)
    first = orch.propose(
        name="first-agent",
        description="The first proposed agent.",
        system_prompt=_VALID_PROMPT,
        skills=["vuln-auditor"],
    )
    assert orch._spec is first
    second = orch.propose(
        name="second-agent",
        description="The second proposed agent replaces the first.",
        system_prompt=_VALID_PROMPT,
    )
    assert orch._spec is second
    assert orch._spec.name == "second-agent"


def test_propose_second_call_replaces_stored_spec(tmp_path: Path) -> None:
    orch = _make_orch(tmp_path)
    orch.propose(
        name="agent-one",
        description="First agent to be proposed.",
        system_prompt=_VALID_PROMPT,
        skills=["vuln-auditor"],
    )
    orch.propose(
        name="agent-two",
        description="Second agent that replaces agent-one.",
        system_prompt=_VALID_PROMPT,
        tools=["Read"],
    )
    assert orch._spec is not None
    assert orch._spec.name == "agent-two"
    assert orch._spec.tools == ["Read"]
    assert orch._spec.skills == []


def test_propose_includes_chosen_refs(tmp_path: Path) -> None:
    orch = _make_orch(tmp_path)
    spec = orch.propose(
        name="full-agent",
        description="Agent with skills tools and mcp servers.",
        system_prompt=_VALID_PROMPT,
        skills=["vuln-auditor"],
        tools=["Read"],
        mcp_servers=["playwright"],
    )
    assert "vuln-auditor" in spec.skills
    assert "Read" in spec.tools
    assert "playwright" in spec.mcp_servers


def test_spawn_no_prior_propose_raises(tmp_path: Path) -> None:
    orch = _make_orch(tmp_path)
    with pytest.raises(RuntimeError, match="no agent spec has been proposed"):
        asyncio.run(orch.spawn(_VALID_TASK))


def test_spawn_collects_assistant_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def _fake_run_agent(spec: AgentSpec, task: str) -> AsyncIterator[AssistantMessage]:
        yield AssistantMessage(content=[TextBlock(text="hello")], model="test-model")
        yield AssistantMessage(content=[TextBlock(text="world")], model="test-model")

    monkeypatch.setattr("agent_fleet.engine.orchestrate.run_agent", _fake_run_agent)

    orch = _make_orch(tmp_path)
    orch.propose(
        name="test-agent",
        description="Test agent for spawn text collection.",
        system_prompt=_VALID_PROMPT,
    )
    result = asyncio.run(orch.spawn(_VALID_TASK))
    assert "hello" in result
    assert "world" in result


def test_orchestrator_options_has_orchestrator_server(tmp_path: Path) -> None:
    orch = _make_orch(tmp_path)
    opts = orchestrator_options(orch)
    assert isinstance(opts, ClaudeAgentOptions)
    assert "orchestrator" in opts.mcp_servers
    assert "mcp__orchestrator__*" in opts.allowed_tools


def test_build_orchestrator_server_returns_config(tmp_path: Path) -> None:
    orch = _make_orch(tmp_path)
    config = build_orchestrator_server(orch)
    # McpSdkServerConfig is a TypedDict; inspect its dict contents.
    assert config["type"] == "sdk"
    assert config["name"] == "orchestrator"


def test_find_skills_delegates_to_router(tmp_path: Path) -> None:
    _corpus(tmp_path)
    orch = _make_orch(tmp_path)
    cards = orch.find_skills("audit code for security issues")
    assert any(c.ref == "vuln-auditor" for c in cards)


def test_find_tools_delegates_to_router(tmp_path: Path) -> None:
    orch = _make_orch(tmp_path)
    cards = orch.find_tools("read a file from disk")
    assert any(c.ref == "Read" for c in cards)


def test_find_mcp_delegates_to_router(tmp_path: Path) -> None:
    schema = {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}
    server = CatalogMcpServer(
        id="mcp.playwright",
        ref="playwright",
        description="Browser automation server.",
        tools=[
            McpTool(name="browser_navigate", description="Navigate to URL.", input_schema=schema)
        ],
    )
    orch = _make_orch(tmp_path, [server])
    cards = orch.find_mcp("automate a browser")
    assert any(c.ref == "playwright" for c in cards)


def test_proposed_spec_returns_none_before_propose(tmp_path: Path) -> None:
    orch = _make_orch(tmp_path)
    assert orch.proposed_spec is None


def test_proposed_spec_returns_spec_after_propose(tmp_path: Path) -> None:
    orch = _make_orch(tmp_path)
    spec = orch.propose(
        name="test-agent",
        description="Agent to verify proposed_spec accessor.",
        system_prompt=_VALID_PROMPT,
    )
    assert orch.proposed_spec is spec


def test_collect_orchestration_returns_joined_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def _fake_query(**kwargs: object) -> AsyncIterator[AssistantMessage]:
        yield AssistantMessage(content=[TextBlock(text="first")], model="test-model")
        yield AssistantMessage(content=[TextBlock(text="second")], model="test-model")

    monkeypatch.setattr("agent_fleet.engine.orchestrate.query", _fake_query)

    router = _router(tmp_path)
    outcome = asyncio.run(collect_orchestration(_VALID_TASK, router))
    assert isinstance(outcome, OrchestrateOutcome)
    assert "first" in outcome.output
    assert "second" in outcome.output
    assert outcome.spec is None


def test_describe_mcp_delegates_to_router(tmp_path: Path) -> None:
    schema = {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}
    server = CatalogMcpServer(
        id="mcp.playwright",
        ref="playwright",
        description="Browser automation server.",
        tools=[
            McpTool(name="browser_navigate", description="Navigate to URL.", input_schema=schema)
        ],
    )
    orch = _make_orch(tmp_path, [server])
    tools = orch.describe_mcp("playwright")
    assert len(tools) == 1
    assert tools[0].name == "browser_navigate"
