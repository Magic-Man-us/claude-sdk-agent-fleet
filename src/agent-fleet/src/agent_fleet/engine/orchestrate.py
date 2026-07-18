from __future__ import annotations

from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    McpSdkServerConfig,
    Message,
    TextBlock,
    create_sdk_mcp_server,
    query,
    tool,
)
from pydantic import TypeAdapter, validate_call

from capabilities_discovery.base import FrozenModel, InputModel
from capabilities_discovery.catalog import (
    CatalogEntryId,
    McpServerRef,
    McpTool,
    RecallLimit,
    SkillRef,
    ToolRef,
)

from ..models.agent import (
    AgentDescription,
    AgentName,
    ModelId,
    PromptBody,
    TaskBrief,
)
from ..models.agent.spec import AgentSpec
from ..router.capability import (
    DEFAULT_SLATE,
    CapabilityRouter,
    McpCard,
    SkillCard,
    ToolCard,
)
from .run import run_agent

ORCHESTRATOR_PROMPT: PromptBody = (
    "You assemble a specialized agent for a task, then run it.\n\n"
    "Workflow:\n"
    "1. Review the environment for the task: call find_skills, find_tools, and find_mcp to see "
    "the relevant skills, tools, and MCP servers. Each find_mcp result lists the tool names most "
    "relevant to the task; when you are leaning toward a server, call describe_mcp for its full "
    "input schemas. Call load_skill only when you need a skill's full contents to decide.\n"
    "2. Choose the minimal capability set that fits the task — prefer fewer tools and skills; "
    "include an MCP server only when its tools are needed.\n"
    "3. Call propose_spec with a name, a one-line description, a focused system prompt for the "
    "worker, and the chosen skills, tools, and mcp_servers.\n"
    "4. Call spawn with the task to run the proposed agent, then report what it returns.\n\n"
    "Pick capabilities from what the review surfaces; do not assume a capability exists without "
    "finding it."
)


def _extract_text(msg: Message) -> list[str]:
    """Pull text strings out of one AssistantMessage; other message types return empty."""
    if isinstance(msg, AssistantMessage):
        return [block.text for block in msg.content if isinstance(block, TextBlock)]
    return []


class OrchestrateOutcome(FrozenModel):
    """Result of running the orchestrator to completion: its final report text and the
    spec it proposed (None if it never called propose_spec)."""

    output: str
    spec: AgentSpec | None = None


# Module-level TypeAdapters — built once, used in every tool response.
_SKILL_CARDS_TA: TypeAdapter[list[SkillCard]] = TypeAdapter(list[SkillCard])
_TOOL_CARDS_TA: TypeAdapter[list[ToolCard]] = TypeAdapter(list[ToolCard])
_MCP_CARDS_TA: TypeAdapter[list[McpCard]] = TypeAdapter(list[McpCard])
_MCP_TOOLS_TA: TypeAdapter[list[McpTool]] = TypeAdapter(list[McpTool])


# ---------------------------------------------------------------------------
# Boundary models — InputModel (extra="ignore") because the LLM is external.
# ---------------------------------------------------------------------------


class _FindSkillsArgs(InputModel):
    query: TaskBrief
    limit: RecallLimit = DEFAULT_SLATE


class _FindToolsArgs(InputModel):
    query: TaskBrief
    limit: RecallLimit = DEFAULT_SLATE


class _FindMcpArgs(InputModel):
    query: TaskBrief
    limit: RecallLimit = DEFAULT_SLATE


class _DescribeMcpArgs(InputModel):
    server: McpServerRef


class _LoadSkillArgs(InputModel):
    skill_id: CatalogEntryId


class _ProposeArgs(InputModel):
    name: AgentName
    description: AgentDescription
    system_prompt: PromptBody
    model: ModelId = ModelId.inherit
    skills: list[SkillRef] = []
    tools: list[ToolRef] = []
    mcp_servers: list[McpServerRef] = []


class _SpawnArgs(InputModel):
    task: TaskBrief


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class Orchestrator:
    """Capability orchestrator: reviews the ranked slates, proposes an AgentSpec, and spawns it.
    The review/propose methods are synchronous and testable without the SDK or any LLM; only
    spawn calls into the SDK."""

    def __init__(self, router: CapabilityRouter) -> None:
        self._router = router
        self._spec: AgentSpec | None = None

    @property
    def proposed_spec(self) -> AgentSpec | None:
        return self._spec

    def find_skills(
        self,
        query: TaskBrief,
        limit: RecallLimit = DEFAULT_SLATE,
    ) -> list[SkillCard]:
        return self._router.find_skills(query, limit)

    def find_tools(self, query: TaskBrief, limit: RecallLimit = DEFAULT_SLATE) -> list[ToolCard]:
        return self._router.find_tools(query, limit)

    def find_mcp(self, query: TaskBrief, limit: RecallLimit = DEFAULT_SLATE) -> list[McpCard]:
        return self._router.find_mcp(query, limit)

    def describe_mcp(self, server: McpServerRef) -> list[McpTool]:
        return self._router.describe_mcp(server)

    def load_skill(self, skill_id: CatalogEntryId) -> str:
        return self._router.load_skill(skill_id)

    def propose(
        self,
        name: AgentName,
        description: AgentDescription,
        system_prompt: PromptBody,
        model: ModelId = ModelId.inherit,
        skills: list[SkillRef] | None = None,
        tools: list[ToolRef] | None = None,
        mcp_servers: list[McpServerRef] | None = None,
    ) -> AgentSpec:
        """Build and validate an AgentSpec from the chosen capabilities; a second call replaces
        the previously stored spec."""
        self._spec = AgentSpec(
            name=name,
            description=description,
            system_prompt=system_prompt,
            model=model,
            skills=skills or [],
            tools=tools or [],
            mcp_servers=mcp_servers or [],
        )
        return self._spec

    @validate_call
    async def spawn(self, task: TaskBrief) -> str:
        """Run the proposed agent on task and return concatenated assistant text. Raises
        RuntimeError when propose has not been called yet — callers should propose first."""
        if self._spec is None:
            raise RuntimeError("no agent spec has been proposed yet — call propose first")
        parts: list[str] = []
        async for msg in run_agent(self._spec, task):
            parts.extend(_extract_text(msg))
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# SDK tool layer
# ---------------------------------------------------------------------------

_FIND_SKILLS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "The task to find skills for."},
        "limit": {"type": "integer", "description": "Max results to return."},
    },
    "required": ["query"],
}

_FIND_TOOLS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "The task to find tools for."},
        "limit": {"type": "integer", "description": "Max results to return."},
    },
    "required": ["query"],
}

_FIND_MCP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "The task to find MCP servers for."},
        "limit": {"type": "integer", "description": "Max results to return."},
    },
    "required": ["query"],
}

_DESCRIBE_MCP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "server": {"type": "string", "description": "MCP server name from find_mcp."},
    },
    "required": ["server"],
}

_LOAD_SKILL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "skill_id": {"type": "string", "description": "Catalog id from find_skills."},
    },
    "required": ["skill_id"],
}

_PROPOSE_SPEC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "Agent slug name."},
        "description": {"type": "string", "description": "One-line agent description."},
        "system_prompt": {"type": "string", "description": "Worker system prompt."},
        "model": {"type": "string", "description": "Model id (omit to inherit)."},
        "skills": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Skill refs to grant.",
        },
        "tools": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Tool grant strings.",
        },
        "mcp_servers": {
            "type": "array",
            "items": {"type": "string"},
            "description": "MCP server names to enable.",
        },
    },
    "required": ["name", "description", "system_prompt"],
}

_SPAWN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "task": {"type": "string", "description": "Task for the proposed agent to execute."},
    },
    "required": ["task"],
}


def build_orchestrator_server(orch: Orchestrator) -> McpSdkServerConfig:
    """Assemble an in-process MCP server that exposes the orchestrator's methods as SDK tools.
    Each wrapper validates raw LLM args through the boundary model before calling the method."""

    @tool(
        "find_skills",
        "Search installed skills and return the most relevant for a task.",
        _FIND_SKILLS_SCHEMA,
    )
    async def _find_skills(args: dict[str, Any]) -> dict[str, Any]:
        parsed = _FindSkillsArgs.model_validate(args)
        cards = orch.find_skills(parsed.query, parsed.limit)
        return {"content": [{"type": "text", "text": _SKILL_CARDS_TA.dump_json(cards).decode()}]}

    @tool(
        "find_tools",
        "Search available tools and return the most relevant for a task.",
        _FIND_TOOLS_SCHEMA,
    )
    async def _find_tools(args: dict[str, Any]) -> dict[str, Any]:
        parsed = _FindToolsArgs.model_validate(args)
        cards = orch.find_tools(parsed.query, parsed.limit)
        return {"content": [{"type": "text", "text": _TOOL_CARDS_TA.dump_json(cards).decode()}]}

    @tool(
        "find_mcp",
        "Search connected MCP servers and return the most relevant for a task.",
        _FIND_MCP_SCHEMA,
    )
    async def _find_mcp(args: dict[str, Any]) -> dict[str, Any]:
        parsed = _FindMcpArgs.model_validate(args)
        cards = orch.find_mcp(parsed.query, parsed.limit)
        return {"content": [{"type": "text", "text": _MCP_CARDS_TA.dump_json(cards).decode()}]}

    @tool(
        "describe_mcp",
        "Return the full tool list and input schemas for one MCP server from find_mcp.",
        _DESCRIBE_MCP_SCHEMA,
    )
    async def _describe_mcp(args: dict[str, Any]) -> dict[str, Any]:
        parsed = _DescribeMcpArgs.model_validate(args)
        tools_list = orch.describe_mcp(parsed.server)
        return {"content": [{"type": "text", "text": _MCP_TOOLS_TA.dump_json(tools_list).decode()}]}

    @tool(
        "load_skill",
        "Return the full SKILL.md body for a skill id from find_skills.",
        _LOAD_SKILL_SCHEMA,
    )
    async def _load_skill(args: dict[str, Any]) -> dict[str, Any]:
        parsed = _LoadSkillArgs.model_validate(args)
        body = orch.load_skill(parsed.skill_id)
        return {"content": [{"type": "text", "text": body}]}

    @tool(
        "propose_spec",
        "Propose the agent spec — name, description, system prompt, and chosen capabilities.",
        _PROPOSE_SPEC_SCHEMA,
    )
    async def _propose_spec(args: dict[str, Any]) -> dict[str, Any]:
        parsed = _ProposeArgs.model_validate(args)
        spec = orch.propose(
            name=parsed.name,
            description=parsed.description,
            system_prompt=parsed.system_prompt,
            model=parsed.model,
            skills=parsed.skills,
            tools=parsed.tools,
            mcp_servers=parsed.mcp_servers,
        )
        return {"content": [{"type": "text", "text": spec.model_dump_json()}]}

    @tool(
        "spawn",
        "Run the proposed agent on the task and return what it produces.",
        _SPAWN_SCHEMA,
    )
    async def _spawn(args: dict[str, Any]) -> dict[str, Any]:
        parsed = _SpawnArgs.model_validate(args)
        output = await orch.spawn(parsed.task)
        return {"content": [{"type": "text", "text": output}]}

    return create_sdk_mcp_server(
        "orchestrator",
        tools=[
            _find_skills,
            _find_tools,
            _find_mcp,
            _describe_mcp,
            _load_skill,
            _propose_spec,
            _spawn,
        ],
    )


def orchestrator_options(orch: Orchestrator) -> ClaudeAgentOptions:
    """ClaudeAgentOptions for the orchestrator turn — in-process server only, no external MCP."""
    return ClaudeAgentOptions(
        system_prompt=ORCHESTRATOR_PROMPT,
        mcp_servers={"orchestrator": build_orchestrator_server(orch)},
        allowed_tools=["mcp__orchestrator__*"],
    )


@validate_call(config={"arbitrary_types_allowed": True})
async def collect_orchestration(task: TaskBrief, router: CapabilityRouter) -> OrchestrateOutcome:
    """Run the orchestrator to completion and return its final text + proposed spec.
    Requires the `claude` CLI at runtime (the SDK spawns it), same as run_agent."""
    orch = Orchestrator(router)
    parts: list[str] = []
    async for msg in query(prompt=task, options=orchestrator_options(orch)):
        parts.extend(_extract_text(msg))
    return OrchestrateOutcome(output="\n".join(parts), spec=orch.proposed_spec)
