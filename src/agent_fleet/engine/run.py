from __future__ import annotations

from collections.abc import AsyncIterator

from claude_agent_sdk import Message, query
from pydantic import validate_call

from ..models.agent import AgentSpec, TaskBrief
from .render import to_options


@validate_call
async def run_agent(spec: AgentSpec, task: TaskBrief) -> AsyncIterator[Message]:
    """Run a generated agent in-process via the Claude Agent SDK.

    Requires the `claude` CLI on PATH (the SDK spawns it).

    Args:
        spec: The agent spec to run, translated to options via `to_options`.
        task: The first user turn handed to the agent.

    Yields:
        Each `Message` from the SDK as the agent runs.
    """
    async for message in query(prompt=task, options=to_options(spec)):
        yield message
