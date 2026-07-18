from __future__ import annotations

from claude_agent_sdk import ClaudeAgentOptions

from agent_fleet import AgentSpec
from agent_fleet.engine.render import render_claude_sdk
from agent_fleet.models.agent import AgentEffort, ModelId


def test_generated_agent_constructs_against_the_real_sdk() -> None:
    """Exec the emitted program against the installed claude_agent_sdk. If a future SDK renames
    or drops a ClaudeAgentOptions parameter the renderer writes, this fails here — in our suite,
    not in a user's generated agent at runtime. The generated module only builds OPTIONS and
    defines `run`; `query` runs solely under `__main__`, so executing the body makes no calls."""
    spec = AgentSpec(
        name="smoke",
        description="A smoke-test agent that exercises every emitted option.",
        system_prompt="You verify the emitter stays compatible with the SDK constructor.",
        model=ModelId.sonnet,
        effort=AgentEffort.high,
        tools=["Read"],
        skills=["error-handling"],
        mcp_servers=["playwright"],
    )
    namespace: dict[str, object] = {}
    exec(compile(render_claude_sdk(spec), "<generated-agent>", "exec"), namespace)

    assert isinstance(namespace["OPTIONS"], ClaudeAgentOptions)
    assert callable(namespace["run"])
