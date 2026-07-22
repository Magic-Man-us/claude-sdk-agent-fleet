"""Claude SDK agent fleet.

The fixed-agent runtime is dependency-light and imported eagerly. Discovery,
assembly, routing, and catalog exports remain available lazily when the
``discovery`` extra is installed.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .runtime import (
    RuntimeAgentPool,
    RuntimeAgentSpec,
    RuntimeRunOutcome,
    run_runtime_with_capture,
)

_EXPORTS: dict[str, tuple[str, str]] = {
    "BUILTIN_TOOLS": ("capdisc.discovery", "BUILTIN_TOOLS"),
    "Catalog": ("capdisc.catalog", "Catalog"),
    "CatalogEntry": ("capdisc.catalog", "CatalogEntry"),
    "CatalogMcpServer": ("capdisc.catalog", "CatalogMcpServer"),
    "CatalogPlugin": ("capdisc.catalog", "CatalogPlugin"),
    "CatalogSkill": ("capdisc.catalog", "CatalogSkill"),
    "CatalogTool": ("capdisc.catalog", "CatalogTool"),
    "scan_environment": ("capdisc.discovery", "scan_environment"),
    "scan_skills": ("capdisc.discovery", "scan_skills"),
    "build_acquire_server": ("agent_fleet.engine.acquire_tool", "build_acquire_server"),
    "grant_acquire_to_subagent": (
        "agent_fleet.engine.acquire_tool",
        "grant_acquire_to_subagent",
    ),
    "with_acquire_tool": ("agent_fleet.engine.acquire_tool", "with_acquire_tool"),
    "run_with_capture": ("agent_fleet.engine.dispatch", "run_with_capture"),
    "EfficiencyConfig": ("agent_fleet.engine.efficiency", "EfficiencyConfig"),
    "EfficiencyReport": ("agent_fleet.engine.efficiency", "EfficiencyReport"),
    "score": ("agent_fleet.engine.efficiency", "score"),
    "write_agent": ("agent_fleet.engine.emit", "write_agent"),
    "build_findings_server": (
        "agent_fleet.engine.findings_tool",
        "build_findings_server",
    ),
    "grant_findings_to_subagent": (
        "agent_fleet.engine.findings_tool",
        "grant_findings_to_subagent",
    ),
    "with_findings_tool": ("agent_fleet.engine.findings_tool", "with_findings_tool"),
    "slugify_name": ("agent_fleet.engine.naming", "slugify_name"),
    "AssemblyResult": ("agent_fleet.engine.pipeline", "AssemblyResult"),
    "assemble": ("agent_fleet.engine.pipeline", "assemble"),
    "generate": ("agent_fleet.engine.pipeline", "generate"),
    "AgentPool": ("agent_fleet.engine.pool", "AgentPool"),
    "AsyncAgentPool": ("agent_fleet.engine.pool", "AsyncAgentPool"),
    "create_agent": ("agent_fleet.engine.pool", "create_agent"),
    "render_claude_sdk": ("agent_fleet.engine.render", "render_claude_sdk"),
    "to_agent_definition": ("agent_fleet.engine.render", "to_agent_definition"),
    "to_options": ("agent_fleet.engine.render", "to_options"),
    "with_hooks": ("agent_fleet.engine.render", "with_hooks"),
    "with_subagents": ("agent_fleet.engine.render", "with_subagents"),
    "run_agent": ("agent_fleet.engine.run", "run_agent"),
    "SelectedCapabilities": ("agent_fleet.engine.select", "SelectedCapabilities"),
    "select": ("agent_fleet.engine.select", "select"),
    "Candidate": ("agent_fleet.engine.source", "Candidate"),
    "CatalogSource": ("agent_fleet.engine.source", "CatalogSource"),
    "InMemoryCatalogSource": ("agent_fleet.engine.source", "InMemoryCatalogSource"),
    "RecallQuery": ("agent_fleet.engine.source", "RecallQuery"),
    "AgentKey": ("agent_fleet.models.agent", "AgentKey"),
    "AgentRunRecord": ("agent_fleet.models.agent", "AgentRunRecord"),
    "AgentSpec": ("agent_fleet.models.agent", "AgentSpec"),
    "Finding": ("agent_fleet.models.agent", "Finding"),
    "FindingContent": ("agent_fleet.models.agent", "FindingContent"),
    "PoolEntry": ("agent_fleet.models.agent", "PoolEntry"),
    "ProblemRequest": ("agent_fleet.models.agent", "ProblemRequest"),
    "RunId": ("agent_fleet.models.agent", "RunId"),
    "RunOutcome": ("agent_fleet.models.agent", "RunOutcome"),
    "RunRecord": ("agent_fleet.models.agent", "RunRecord"),
    "SessionId": ("agent_fleet.models.agent", "SessionId"),
    "CapabilityRouter": ("agent_fleet.router.capability", "CapabilityRouter"),
    "McpCard": ("agent_fleet.router.capability", "McpCard"),
    "PluginCard": ("agent_fleet.router.capability", "PluginCard"),
    "SkillCard": ("agent_fleet.router.capability", "SkillCard"),
    "ToolCard": ("agent_fleet.router.capability", "ToolCard"),
}

__all__ = [
    "RuntimeAgentPool",
    "RuntimeAgentSpec",
    "RuntimeRunOutcome",
    "run_runtime_with_capture",
    *_EXPORTS,
]


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    try:
        value = getattr(import_module(module_name), attribute)
    except ModuleNotFoundError as exc:
        if exc.name == "capdisc" or (exc.name and exc.name.startswith("capdisc.")):
            raise ModuleNotFoundError(
                "Fleet discovery/assembly requires the 'discovery' extra: "
                "install claude-sdk-agent-fleet[discovery]"
            ) from exc
        raise
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
