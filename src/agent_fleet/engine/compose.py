from __future__ import annotations

from ..models.agent import AgentSpec, ProblemRequest, TemplatedPrompt
from .naming import slugify_name
from .select import SelectedCapabilities


def compose(request: ProblemRequest, selection: SelectedCapabilities) -> AgentSpec:
    """Assemble the final agent spec from the request and its selected capabilities.

    Args:
        request: The problem request; supplies name, task, model, tags, and an optional
            override system prompt. A missing name is auto-slugged from the task.
        selection: The chosen skills, tools, and MCP servers to equip.

    Returns:
        The agent spec, using `request.system_prompt` when given, else a templated prompt built
        from the task and selected tools/skills.
    """
    name = request.name or slugify_name(request.task)
    prompt = TemplatedPrompt(
        name=name,
        task=request.task,
        tools=selection.tools,
        skills=selection.skills,
    )
    return AgentSpec(
        name=name,
        description=request.task,  # AgentDescription summarizes the (possibly long) task
        system_prompt=request.system_prompt or prompt.body,
        model=request.model,
        tags=request.tags,
        tools=selection.tools,
        skills=selection.skills,
        mcp_servers=selection.mcp_servers,
    )
