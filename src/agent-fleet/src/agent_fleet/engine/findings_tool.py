from __future__ import annotations

import dataclasses
from typing import Any

from claude_agent_sdk import (
    ClaudeAgentOptions,
    McpSdkServerConfig,
    create_sdk_mcp_server,
    tool,
)
from claude_agent_sdk.types import AgentDefinition

from capdisc.base import InputModel

from ..models.agent import (
    AgentKey,
    AgentName,
    FindingContent,
    RunId,
    SessionId,
)
from .pool import AgentPool

FINDINGS_SERVER = "findings"
WRITE_FINDING_TOOL = "mcp__findings__write_finding"

_WRITE_FINDING_DESCRIPTION = (
    "Record a finding for this agent. Call this whenever you discover something worth "
    "preserving — every call is saved permanently and visible to the supervisor and every other "
    "lens; nothing is lost or overwritten."
)
_WRITE_FINDING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "content": {"type": "string", "description": "The finding to record."},
    },
    "required": ["content"],
}


class _WriteFindingArgs(InputModel):
    content: FindingContent


def build_findings_server(
    pool: AgentPool,
    agent_key: AgentKey,
    run_id: RunId,
    session_id: SessionId,
    agent_name: AgentName | None,
) -> McpSdkServerConfig:
    """Build an in-process MCP server exposing one `write_finding` tool bound to one fixed identity.

    Every write through the returned server is attributed to the `agent_name`/`session_id` fixed
    here — an in-process MCP tool handler receives only the raw args dict, never the identity of the
    caller, so a single server instance shared across several lenses cannot tell which lens called
    it and would record all their findings under this one identity. For per-lens attribution the
    caller therefore builds ONE server per lens, each with that lens's fixed `agent_name`. The
    `session_id` baked in is the wiring-time session (the run's main session), not the lens's own:
    a dispatched lens is handed its live session id only once it runs and `AgentDefinition` has no
    field to pre-assign one, so it isn't knowable at wiring time — the identity chosen here is the
    most specific one available when the server is built.

    Args:
        pool: The pool every finding is inserted into.
        agent_key: The pooled agent the findings belong to.
        run_id: The run the findings are produced within.
        session_id: The writing agent's session id, fixed for every write through this server.
        agent_name: The lens this server writes as, or None for the main/supervisor agent.

    Returns:
        The in-process MCP server config to mount under the `findings` name.
    """

    @tool("write_finding", _WRITE_FINDING_DESCRIPTION, _WRITE_FINDING_SCHEMA)
    async def _write_finding(args: dict[str, Any]) -> dict[str, Any]:
        parsed = _WriteFindingArgs.model_validate(args)
        pool.record_finding(agent_key, run_id, session_id, parsed.content, agent_name=agent_name)
        return {"content": [{"type": "text", "text": "recorded"}]}

    return create_sdk_mcp_server(FINDINGS_SERVER, tools=[_write_finding])


def with_findings_tool(
    options: ClaudeAgentOptions,
    pool: AgentPool,
    agent_key: AgentKey,
    run_id: RunId,
    session_id: SessionId,
    *,
    agent_name: AgentName | None = None,
) -> ClaudeAgentOptions:
    """Augment already-built options with the shared findings server and its tool grant.

    Mirrors `render.with_subagents`'s append-only `dataclasses.replace` discipline: the `findings`
    server is merged into `options.mcp_servers` (existing servers preserved) and
    `WRITE_FINDING_TOOL` is granted once (existing grants preserved, no duplicate). Every write
    through these options is attributed to the fixed `agent_name`/`session_id` — see
    `build_findings_server` for why a shared server records one identity.

    Args:
        options: The already-built options to augment; not mutated.
        pool: The pool findings are inserted into.
        agent_key: The pooled agent the findings belong to.
        run_id: The run the findings are produced within.
        session_id: The writing agent's session id, fixed for every write through these options.
        agent_name: The identity to write as, or None for the main/supervisor agent.

    Returns:
        A copy of `options` with the findings server mounted and its tool granted.
    """
    server = build_findings_server(pool, agent_key, run_id, session_id, agent_name)
    # options.mcp_servers is a dict of named servers or a path/str to a config file; findings is an
    # in-process server that can only merge into the dict form, which is what this package builds.
    existing = options.mcp_servers if isinstance(options.mcp_servers, dict) else {}
    mcp_servers = {**existing, FINDINGS_SERVER: server}
    allowed_tools = list(options.allowed_tools)
    if WRITE_FINDING_TOOL not in allowed_tools:
        allowed_tools.append(WRITE_FINDING_TOOL)
    return dataclasses.replace(options, mcp_servers=mcp_servers, allowed_tools=allowed_tools)


def grant_findings_to_subagent(
    definition: AgentDefinition, *, session_id: SessionId | None = None
) -> AgentDefinition:
    """Grant a subagent access to the parent-level shared findings server, append-only and deduped.

    The SDK inherits a parent's in-process MCP server into a subagent only when the subagent's own
    `AgentDefinition.mcpServers` names it, so a lens can reach the `findings` server only if this is
    applied: `FINDINGS_SERVER` is appended to `mcpServers` and `WRITE_FINDING_TOOL` to `tools`, each
    once. When `tools` is None ("inherit everything") it is left None — the harness already grants
    the findings tool in that case — so an inherit-all subagent is never narrowed to an explicit
    list.

    `session_id` is accepted but not wired into the definition: `AgentDefinition` has no field to
    pre-assign a lens's session, and the shared server records whichever identity was fixed when it
    was built (see `build_findings_server`). Per-lens attribution needs one findings server per lens
    with that lens's fixed `agent_name`, not a value threaded through the definition here.

    Args:
        definition: The subagent definition to grant findings access to; not mutated.
        session_id: Accepted for call-site symmetry; intentionally not applied — see above.

    Returns:
        A copy of `definition` with the findings server and tool grant added.
    """
    del session_id
    mcp_servers = list(definition.mcpServers) if definition.mcpServers is not None else []
    if FINDINGS_SERVER not in mcp_servers:
        mcp_servers.append(FINDINGS_SERVER)
    if definition.tools is None:
        tools = None
    else:
        tools = list(definition.tools)
        if WRITE_FINDING_TOOL not in tools:
            tools.append(WRITE_FINDING_TOOL)
    return dataclasses.replace(definition, mcpServers=mcp_servers, tools=tools)
