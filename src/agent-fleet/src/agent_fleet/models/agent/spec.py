from __future__ import annotations

from capabilities_discovery.base import FrozenModel
from capabilities_discovery.catalog import McpServerRef, SkillRef, Tag, ToolRef
from capabilities_discovery.hooks import HookConfig

from .thinking import ThinkingConfig
from .types import (
    AgentColor,
    AgentDescription,
    AgentEffort,
    AgentName,
    InitialPrompt,
    Isolation,
    MaxTurns,
    MemoryScope,
    ModelId,
    PermissionMode,
    PromptBody,
)


class AgentSpec(FrozenModel):
    """The canonical agent definition the pipeline assembles and the Claude Agent SDK emitter
    renders into a runnable program. `tools` and `mcp_servers` are kept separate here; the merged
    SDK tool-grant is derived at emission by `engine.render.tool_grant`."""

    name: AgentName
    description: AgentDescription
    system_prompt: PromptBody
    model: ModelId = ModelId.inherit
    tags: list[Tag] = []

    tools: list[ToolRef] = []
    disallowed_tools: list[ToolRef] = []
    skills: list[SkillRef] = []
    mcp_servers: list[McpServerRef] = []
    effort: AgentEffort | None = None
    thinking: ThinkingConfig | None = None
    max_turns: MaxTurns | None = None
    background: bool = False
    memory: MemoryScope | None = None
    permission_mode: PermissionMode | None = None
    initial_prompt: InitialPrompt | None = None
    isolation: Isolation | None = None
    color: AgentColor | None = None
    hooks: HookConfig | None = None
