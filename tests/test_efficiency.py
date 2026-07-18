from __future__ import annotations

from agent_fleet import AgentSpec, EfficiencyConfig, score
from capabilities_discovery.catalog import SkillRef, ToolRef


def _spec(tools: list[ToolRef] | None = None, skills: list[SkillRef] | None = None) -> AgentSpec:
    return AgentSpec(
        name="worker",
        description="A small focused agent.",
        system_prompt="You are a focused agent. Do exactly one job, report it, and stop now.",
        tools=tools or [],
        skills=skills or [],
    )


def test_minimal_spec_passes() -> None:
    assert score(_spec(tools=["Read"], skills=["changelog"])).passed


def test_too_many_tools_fails() -> None:
    report = score(_spec(tools=[f"tool-{i}" for i in range(50)]))
    assert not report.passed
    assert any(not r.passed and r.dimension.value == "tool_count" for r in report.results)


def test_config_can_tighten_budget() -> None:
    assert not score(_spec(tools=["Read", "Write"]), EfficiencyConfig(tool_budget=1)).passed


def test_zero_tools_fails_as_inherit_all() -> None:
    report = score(_spec(tools=[]))
    assert not report.passed
    tool_dim = next(r for r in report.results if r.dimension.value == "tool_count")
    assert not tool_dim.passed
    assert "inherits" in tool_dim.detail
