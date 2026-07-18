from __future__ import annotations

from capabilities_discovery.catalog import (
    Catalog,
    CatalogEntry,
    CatalogMcpServer,
    CatalogPlugin,
    CatalogSkill,
    CatalogTool,
)
from capabilities_discovery.discovery import BUILTIN_TOOLS, scan_environment, scan_skills

from .engine.acquire_tool import (
    build_acquire_server,
    grant_acquire_to_subagent,
    with_acquire_tool,
)
from .engine.dispatch import run_with_capture
from .engine.efficiency import EfficiencyConfig, EfficiencyReport, score
from .engine.emit import write_agent
from .engine.findings_tool import (
    build_findings_server,
    grant_findings_to_subagent,
    with_findings_tool,
)
from .engine.naming import slugify_name
from .engine.pipeline import AssemblyResult, assemble, generate
from .engine.pool import AgentPool, AsyncAgentPool, create_agent
from .engine.render import render_claude_sdk, to_agent_definition, to_options, with_subagents
from .engine.run import run_agent
from .engine.select import SelectedCapabilities, select
from .engine.source import Candidate, CatalogSource, InMemoryCatalogSource, RecallQuery
from .models.agent import (
    AgentKey,
    AgentRunRecord,
    AgentSpec,
    Finding,
    FindingContent,
    PoolEntry,
    ProblemRequest,
    RunId,
    RunOutcome,
    RunRecord,
    SessionId,
)
from .router.capability import CapabilityRouter, McpCard, PluginCard, SkillCard, ToolCard

__all__ = [
    "BUILTIN_TOOLS",
    "AgentKey",
    "AgentPool",
    "AgentRunRecord",
    "AgentSpec",
    "AssemblyResult",
    "AsyncAgentPool",
    "Candidate",
    "CapabilityRouter",
    "Catalog",
    "CatalogEntry",
    "CatalogMcpServer",
    "CatalogPlugin",
    "CatalogSkill",
    "CatalogSource",
    "CatalogTool",
    "EfficiencyConfig",
    "EfficiencyReport",
    "Finding",
    "FindingContent",
    "InMemoryCatalogSource",
    "McpCard",
    "PluginCard",
    "PoolEntry",
    "ProblemRequest",
    "RecallQuery",
    "RunId",
    "RunOutcome",
    "RunRecord",
    "SelectedCapabilities",
    "SessionId",
    "SkillCard",
    "ToolCard",
    "assemble",
    "build_acquire_server",
    "build_findings_server",
    "create_agent",
    "generate",
    "grant_acquire_to_subagent",
    "grant_findings_to_subagent",
    "render_claude_sdk",
    "run_agent",
    "run_with_capture",
    "scan_environment",
    "scan_skills",
    "score",
    "select",
    "slugify_name",
    "to_agent_definition",
    "to_options",
    "with_acquire_tool",
    "with_findings_tool",
    "with_subagents",
    "write_agent",
]
