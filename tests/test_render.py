from __future__ import annotations

from claude_agent_sdk import ClaudeAgentOptions

from agent_fleet import AgentSpec
from agent_fleet.engine.render import (
    SUBAGENT_TOOL,
    render_claude_sdk,
    to_agent_definition,
    to_options,
    tool_grant,
    with_subagents,
)
from agent_fleet.models.agent import (
    AdaptiveThinking,
    AgentEffort,
    DisabledThinking,
    MemoryScope,
    ModelId,
    PermissionMode,
    ThinkingDisplay,
)


def _load_options(code: str) -> ClaudeAgentOptions:
    # exec the emitted program — it imports claude_agent_sdk and builds OPTIONS — without
    # tripping its __main__ entrypoint, then hand back the constructed options object
    namespace: dict[str, object] = {"__name__": "generated_agent"}
    exec(compile(code, "<agent>", "exec"), namespace)
    return namespace["OPTIONS"]


def test_agent_spec_round_trips_through_its_own_dump() -> None:
    # the spec carries no derived field, so its dump re-validates cleanly under extra="forbid"
    # (the /generate -> /render flow does exactly this); the merged grant lives in tool_grant.
    spec = AgentSpec(
        name="auditor",
        description="Audits code for vulnerabilities.",
        system_prompt="You are auditor. Audit the code for vulnerabilities and stop now.",
        tools=("Read", "Grep"),
        mcp_servers=("plugin-playwright-playwright",),
    )
    dumped = spec.model_dump(mode="json")
    assert "allowed_tools" not in dumped  # tools + mcp_servers stay separate on the wire
    assert AgentSpec.model_validate(dumped) == spec


def test_tool_grant_merges_tools_and_mcp_wildcards() -> None:
    spec = AgentSpec(
        name="auditor",
        description="Audits code for vulnerabilities.",
        system_prompt="You are auditor. Audit the code for vulnerabilities and stop now.",
        tools=("Read", "Grep"),
        mcp_servers=("plugin-playwright-playwright",),
    )
    assert tool_grant(spec) == ["Read", "Grep", "mcp__plugin-playwright-playwright__*"]


def test_render_sdk_emits_constructible_options() -> None:
    spec = AgentSpec(
        name="auditor",
        description="Audits code for vulnerabilities.",
        system_prompt="You are auditor. Audit the code for vulnerabilities and stop now.",
        tools=("Read", "Grep"),
        skills=("appsec-audit",),
    )
    code = render_claude_sdk(spec)
    assert "from claude_agent_sdk import ClaudeAgentOptions, query" in code
    options = _load_options(code)
    assert isinstance(options, ClaudeAgentOptions)
    assert options.allowed_tools == ["Read", "Grep"]
    assert options.skills == ["appsec-audit"]


def test_render_sdk_emits_model_and_effort_when_set() -> None:
    spec = AgentSpec(
        name="auditor",
        description="Audits code for vulnerabilities.",
        system_prompt="You are auditor. Audit the code for vulnerabilities and stop now.",
        tools=("Read",),
        model=ModelId.haiku,
        effort=AgentEffort.low,
    )
    options = _load_options(render_claude_sdk(spec))
    assert options.model == "haiku"
    assert options.effort == "low"


def test_render_sdk_omits_model_and_effort_by_default() -> None:
    spec = AgentSpec(
        name="auditor",
        description="Audits code for vulnerabilities.",
        system_prompt="You are auditor. Audit the code for vulnerabilities and stop now.",
        tools=("Read",),
    )
    options = _load_options(render_claude_sdk(spec))
    assert options.model is None  # inherit → no model kwarg emitted
    assert options.effort is None


def test_render_sdk_emits_definition_fields_when_set() -> None:
    spec = AgentSpec(
        name="auditor",
        description="Audits code for vulnerabilities.",
        system_prompt="You are auditor. Audit the code for vulnerabilities and stop now.",
        tools=("Read",),
        disallowed_tools=("Bash(rm:*)",),
        max_turns=15,
        permission_mode=PermissionMode.accept_edits,
    )
    options = _load_options(render_claude_sdk(spec))
    assert options.disallowed_tools == ["Bash(rm:*)"]
    assert options.max_turns == 15
    assert options.permission_mode == "acceptEdits"  # SDK literal value


def test_render_sdk_omits_definition_fields_by_default() -> None:
    spec = AgentSpec(
        name="auditor",
        description="Audits code for vulnerabilities.",
        system_prompt="You are auditor. Audit the code for vulnerabilities and stop now.",
        tools=("Read",),
    )
    options = _load_options(render_claude_sdk(spec))
    assert not options.disallowed_tools  # kwarg omitted → SDK default
    assert options.max_turns is None
    assert options.permission_mode is None


def test_render_sdk_loads_mcp_from_environment_without_baking_config() -> None:
    spec = AgentSpec(
        name="browser-agent",
        description="Drives a browser via Playwright.",
        system_prompt="You are browser-agent. Drive the browser to complete the task and stop.",
        mcp_servers=("plugin-playwright-playwright",),
    )
    code = render_claude_sdk(spec)
    options = _load_options(code)
    # security model: tools are gated by name and config loads from the user's environment;
    # the server's command/url/credentials never enter generated code
    assert options.allowed_tools == ["mcp__plugin-playwright-playwright__*"]
    assert options.setting_sources == ["user", "project"]
    assert options.mcp_servers == {}  # never baked into the emitted program
    assert "mcp_servers=" not in code


def test_render_sdk_emits_thinking_when_set() -> None:
    spec = AgentSpec(
        name="auditor",
        description="Audits code for vulnerabilities.",
        system_prompt="You are auditor. Audit the code for vulnerabilities and stop now.",
        tools=("Read",),
        thinking=AdaptiveThinking(display=ThinkingDisplay.summarized),
    )
    options = _load_options(render_claude_sdk(spec))
    # the emitted dict is the SDK's ThinkingConfig TypedDict shape, display surfaced for capture
    assert options.thinking == {"type": "adaptive", "display": "summarized"}


def test_render_sdk_emits_thinking_without_optional_display() -> None:
    spec = AgentSpec(
        name="auditor",
        description="Audits code for vulnerabilities.",
        system_prompt="You are auditor. Audit the code for vulnerabilities and stop now.",
        tools=("Read",),
        thinking=DisabledThinking(),
    )
    options = _load_options(render_claude_sdk(spec))
    assert options.thinking == {"type": "disabled"}  # exclude_none drops the unset display


def test_render_sdk_omits_thinking_by_default() -> None:
    spec = AgentSpec(
        name="auditor",
        description="Audits code for vulnerabilities.",
        system_prompt="You are auditor. Audit the code for vulnerabilities and stop now.",
        tools=("Read",),
    )
    code = render_claude_sdk(spec)
    assert "thinking=" not in code
    assert _load_options(code).thinking is None


def test_to_agent_definition_maps_every_populated_field() -> None:
    spec = AgentSpec(
        name="worker",
        description="Does focused subwork.",
        system_prompt="You are worker. Do the focused subtask and stop now.",
        tools=("Read", "Grep"),
        disallowed_tools=("Bash(rm:*)",),
        model=ModelId.haiku,
        skills=("appsec-audit",),
        mcp_servers=("plugin-playwright-playwright",),
        effort=AgentEffort.high,
        max_turns=15,
        background=True,
        memory=MemoryScope.project,
        permission_mode=PermissionMode.accept_edits,
        initial_prompt="Begin with the manifest.",
    )
    definition = to_agent_definition(spec)
    assert definition.description == "Does focused subwork."
    assert definition.prompt == spec.system_prompt
    assert definition.tools == tool_grant(spec)
    assert definition.disallowedTools == ["Bash(rm:*)"]
    assert definition.model == "haiku"
    assert definition.skills == ["appsec-audit"]
    assert definition.mcpServers == ["plugin-playwright-playwright"]
    assert definition.effort == "high"
    assert definition.maxTurns == 15
    assert definition.background is True
    assert definition.memory == "project"
    assert definition.permissionMode == "acceptEdits"
    assert definition.initialPrompt == "Begin with the manifest."


def test_to_agent_definition_leaves_defaults_none_but_keeps_empty_tool_grant() -> None:
    spec = AgentSpec(
        name="worker",
        description="Does focused subwork.",
        system_prompt="You are worker. Do the focused subtask and stop now.",
    )
    definition = to_agent_definition(spec)
    assert definition.model is None
    assert definition.disallowedTools is None
    assert definition.skills is None
    assert definition.mcpServers is None
    assert definition.memory is None
    assert definition.effort is None
    assert definition.permissionMode is None
    # empty grant stays [] (a legitimate "no tools"), never collapsed to None ("inherit everything")
    assert definition.tools == []
    assert tool_grant(spec) == []


def test_with_subagents_wires_agents_and_grants_tool_leaving_rest_unchanged() -> None:
    main = AgentSpec(
        name="supervisor",
        description="Coordinates the subworkers.",
        system_prompt="You are supervisor. Delegate and stop now.",
        tools=("Read",),
        model=ModelId.haiku,
    )
    base = to_options(main)
    assert SUBAGENT_TOOL not in base.allowed_tools
    sub = AgentSpec(
        name="worker",
        description="Does focused subwork.",
        system_prompt="You are worker. Do the focused subtask and stop now.",
        tools=("Grep",),
    )
    options = with_subagents(base, {"worker": sub})
    assert set(options.agents) == {"worker"}
    assert options.agents["worker"] == to_agent_definition(sub)
    assert SUBAGENT_TOOL in options.allowed_tools
    assert options.system_prompt == base.system_prompt
    assert options.model == base.model
    assert base.agents is None  # original left untouched


def test_with_subagents_does_not_duplicate_the_tool_grant() -> None:
    main = AgentSpec(
        name="supervisor",
        description="Coordinates the subworkers.",
        system_prompt="You are supervisor. Delegate and stop now.",
        tools=("Read",),
    )
    once = with_subagents(to_options(main), {})
    twice = with_subagents(once, {})
    assert twice.allowed_tools.count(SUBAGENT_TOOL) == 1


def test_with_subagents_empty_mapping_still_sets_agents_and_grants_tool() -> None:
    main = AgentSpec(
        name="supervisor",
        description="Coordinates the subworkers.",
        system_prompt="You are supervisor. Delegate and stop now.",
        tools=("Read",),
    )
    options = with_subagents(to_options(main), {})
    assert options.agents == {}
    assert SUBAGENT_TOOL in options.allowed_tools


def test_with_subagents_supervisor_scenario_round_trips_two_workers() -> None:
    main = AgentSpec(
        name="supervisor",
        description="Coordinates the subworkers.",
        system_prompt="You are supervisor. Delegate and stop now.",
        tools=("Read",),
    )
    spec_a = AgentSpec(
        name="worker-a",
        description="Handles the first subtask.",
        system_prompt="You are worker-a. Handle the first subtask and stop now.",
        tools=("Grep",),
    )
    spec_b = AgentSpec(
        name="worker-b",
        description="Handles the second subtask.",
        system_prompt="You are worker-b. Handle the second subtask and stop now.",
        tools=("Read", "Edit"),
    )
    options = with_subagents(to_options(main), {"worker-a": spec_a, "worker-b": spec_b})
    assert set(options.agents) == {"worker-a", "worker-b"}
    assert options.agents["worker-a"].prompt == spec_a.system_prompt
    assert options.agents["worker-a"].tools == tool_grant(spec_a)
    assert options.agents["worker-b"].tools == tool_grant(spec_b)


def test_agent_spec_thinking_round_trips_through_discriminated_union() -> None:
    # the thinking union must dispatch on `type` when a dumped spec is re-validated
    spec = AgentSpec(
        name="auditor",
        description="Audits code for vulnerabilities.",
        system_prompt="You are auditor. Audit the code for vulnerabilities and stop now.",
        thinking=AdaptiveThinking(display=ThinkingDisplay.summarized),
    )
    revalidated = AgentSpec.model_validate(spec.model_dump(mode="json"))
    assert revalidated == spec
    assert isinstance(revalidated.thinking, AdaptiveThinking)
