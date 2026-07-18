from __future__ import annotations

import pytest
from claude_agent_sdk import ClaudeAgentOptions
from pydantic import ValidationError

from agent_fleet import AgentSpec
from agent_fleet.engine.render import to_options
from agent_fleet.engine.run import run_agent
from agent_fleet.models.agent import (
    AdaptiveThinking,
    AgentEffort,
    ModelId,
    PermissionMode,
    ThinkingDisplay,
)


def test_to_options_builds_real_options() -> None:
    spec = AgentSpec(
        name="auditor",
        description="Audits code for vulnerabilities.",
        system_prompt="You are auditor. Audit the code for vulnerabilities and stop now.",
        model=ModelId.haiku,
        effort=AgentEffort.low,
        tools=("Read", "Grep"),
        skills=("appsec-audit",),
        mcp_servers=("plugin-playwright-playwright",),
    )
    options = to_options(spec)
    assert isinstance(options, ClaudeAgentOptions)
    assert options.model == "haiku"
    assert options.effort == "low"
    assert options.allowed_tools == ["Read", "Grep", "mcp__plugin-playwright-playwright__*"]
    assert options.skills == ["appsec-audit"]
    assert options.setting_sources == ["user", "project"]


def test_to_options_minimal_spec_loads_no_environment() -> None:
    spec = AgentSpec(
        name="tiny",
        description="A tiny agent.",
        system_prompt="You are tiny. Do the one job and stop now.",
        tools=("Read",),
    )
    options = to_options(spec)
    assert options.model is None
    assert options.effort is None
    assert options.skills is None
    assert options.setting_sources is None  # no skills/mcp → settings are not loaded
    assert options.disallowed_tools == []  # SDK default; the field is list[str], not optional
    assert options.max_turns is None
    assert options.permission_mode is None


def test_to_options_emits_definition_fields_when_set() -> None:
    spec = AgentSpec(
        name="auditor",
        description="Audits code for vulnerabilities.",
        system_prompt="You are auditor. Audit the code for vulnerabilities and stop now.",
        tools=("Read",),
        disallowed_tools=("Bash(rm:*)",),
        max_turns=15,
        permission_mode=PermissionMode.accept_edits,
    )
    options = to_options(spec)
    assert options.disallowed_tools == ["Bash(rm:*)"]
    assert options.max_turns == 15
    assert options.permission_mode == "acceptEdits"


def test_to_options_passes_thinking_config_through() -> None:
    spec = AgentSpec(
        name="auditor",
        description="Audits code for vulnerabilities.",
        system_prompt="You are auditor. Audit the code for vulnerabilities and stop now.",
        tools=("Read",),
        thinking=AdaptiveThinking(display=ThinkingDisplay.summarized),
    )
    options = to_options(spec)
    assert options.thinking == {"type": "adaptive", "display": "summarized"}


def test_to_options_omits_thinking_by_default() -> None:
    spec = AgentSpec(
        name="auditor",
        description="Audits code for vulnerabilities.",
        system_prompt="You are auditor. Audit the code for vulnerabilities and stop now.",
        tools=("Read",),
    )
    assert to_options(spec).thinking is None


def test_run_agent_rejects_too_short_task() -> None:
    # @validate_call enforces TaskBrief at the boundary, before the agent ever runs
    spec = AgentSpec(
        name="tiny",
        description="A tiny agent.",
        system_prompt="You are tiny. Do exactly one job, report it, and stop now.",
    )
    with pytest.raises(ValidationError):
        run_agent(spec, "hi")
