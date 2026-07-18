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
from pydantic import TypeAdapter

from capdisc.base import InputModel

from ..models.agent import AgentKey, AgentSpec, TaskBrief, TemplatedPrompt
from ..router.capability import CapabilityRouter
from .dispatch import run_with_capture
from .naming import slugify_name
from .pool import AgentPool

ACQUIRE_SERVER = "acquire"
ACQUIRE_TOOL = "mcp__acquire__acquire_capability"

_ACQUIRE_DESCRIPTION = (
    "Dynamically find and run a properly-equipped agent for a capability you don't currently "
    "have — describe what you need and what you want done with it; a fresh agent is assembled "
    "around the best-matching tools/MCP servers/skills found in the environment, run, and its "
    "output returned to you."
)
_ACQUIRE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "need": {
            "type": "string",
            "description": "What capability is missing — the tool, MCP server, or skill you need.",
        },
        "task": {
            "type": "string",
            "description": "What you want the newly-equipped agent to actually do.",
        },
    },
    "required": ["need", "task"],
}

_TOOL_SLATE = 5
_MCP_SLATE = 2
_SKILL_SLATE = 3

_ACQUIRE_INFIX = "-acquire-"
_MAX_AGENT_KEY_LEN = 128  # AgentKey's max_length

_AGENT_KEY_ADAPTER: TypeAdapter[AgentKey] = TypeAdapter(AgentKey)


class _AcquireArgs(InputModel):
    need: TaskBrief
    task: TaskBrief


def acquired_agent_key(agent_key: AgentKey, need: TaskBrief) -> AgentKey:
    """Derive the acquired agent's stable pool id from the caller's `agent_key` and its `need`.

    Deterministic by construction: the same `agent_key` and `need` always yield the same id, so a
    lens re-acquiring the same recurring capability resumes the same acquired agent's session (via
    the pool's reuse-session-on-overwrite behavior) instead of spawning a disconnected fresh one.
    The composed `"{agent_key}-acquire-{slug}"` is capped at `AgentKey`'s 128-char maximum by
    truncating the `agent_key` BASE, never the need-derived slug: the infix and full slug are always
    reserved and appended, and only the caller's `agent_key` prefix is trimmed to fit. That keeps
    the slug's discriminating power intact — distinct needs always yield distinct ids — and, because
    the infix-plus-slug tail is always present, the result never collapses to the caller's own
    `agent_key`. Any trailing separator on the trimmed base is stripped before the string is
    validated back into a real `AgentKey`.

    Args:
        agent_key: The caller/supervisor pooled agent the acquisition was requested from.
        need: The described missing capability the acquired agent is equipped for.

    Returns:
        The derived, validated `AgentKey` keying the acquired agent's pool entry.
    """
    suffix = f"{_ACQUIRE_INFIX}{slugify_name(need)}"
    base = agent_key[: _MAX_AGENT_KEY_LEN - len(suffix)].rstrip("-._")
    return _AGENT_KEY_ADAPTER.validate_python(f"{base}{suffix}")


def build_acquire_server(
    router: CapabilityRouter, pool: AgentPool, agent_key: AgentKey
) -> McpSdkServerConfig:
    """Build an in-process MCP server exposing one `acquire_capability` tool, purely by composition.

    Unlike `findings_tool.build_findings_server`, which must be built once per lens because a
    shared findings server cannot tell which lens called it and would misattribute every write,
    this server carries no per-caller identity: the `router`, `pool`, and `agent_key` fixed here
    are safely shareable across every lens of a run, so ONE instance mounted on the
    supervisor and inherited by its subagents is correct — there is nothing to over-restrict to a
    one-server-per-lens pattern.

    The handler is genuinely just composition of pieces that already exist — no new recall or
    execution machinery is introduced:

    - recall reuses the same BM25 slates every other caller uses (`CapabilityRouter.find_tools`/
      `find_mcp`/`find_skills`);
    - the fresh agent is an ordinary `AgentSpec` named by `naming.slugify_name` and translated by
      `render.to_options`, exactly as any other spec;
    - the acquired agent gets its OWN resumable pool entry, keyed by an `AgentKey` derived
      deterministically from the caller's `agent_key` and the need (`acquired_agent_key`), saved
      via `pool.save` so it is a real, listable, resumable entry (it shows up in
      `list_agents`/`find_agents` too — an intentional side effect);
    - execution reuses `dispatch.run_with_capture` under that derived `agent_key`, mirroring
      `run_agent`'s own resume-vs-new decision (resume when the derived entry already has
      prior runs, else start fresh), so the acquired run and its captured sessions live under the
      acquired agent's own id.

    Because the derivation is deterministic, a later `acquire_capability` call for the same
    recurring need — or a direct `run_agent`/`get_agent` on the derived id —
    continues the SAME conversation rather than starting fresh each time. This distinguishes
    acquisition from `render.with_subagents`' dispatch: this calls `query()` ourselves in Python
    rather than routing through the harness `Agent`/`Task` tool, so the acquired agent is a
    genuinely separate top-level run keyed by its own `agent_key`, not a harness-level subagent of
    the calling run.

    Args:
        router: The capability router whose BM25 slates equip the acquired agent.
        agent_key: The caller/supervisor pooled agent each acquired agent's own `agent_key` is
            derived from.
        pool: The pool the acquired agent's entry, run, and captured agent sessions are recorded in.

    Returns:
        The in-process MCP server config to mount under the `acquire` name.
    """

    @tool("acquire_capability", _ACQUIRE_DESCRIPTION, _ACQUIRE_SCHEMA)
    async def _acquire_capability(args: dict[str, Any]) -> dict[str, Any]:
        parsed = _AcquireArgs.model_validate(args)
        tool_cards = router.find_tools(parsed.need, limit=_TOOL_SLATE)
        mcp_cards = router.find_mcp(parsed.need, limit=_MCP_SLATE)
        skill_cards = router.find_skills(parsed.need, limit=_SKILL_SLATE)
        tools = [card.ref for card in tool_cards]
        skills = [card.ref for card in skill_cards]
        name = slugify_name(parsed.need)
        spec = AgentSpec(
            name=name,
            description=parsed.need,
            system_prompt=TemplatedPrompt(
                name=name, task=parsed.task, tools=tools, skills=skills
            ).body,
            tools=tools,
            skills=skills,
            mcp_servers=[card.ref for card in mcp_cards],
        )
        derived_agent_key = acquired_agent_key(agent_key, parsed.need)
        entry = pool.save(derived_agent_key, spec)
        prior_runs = pool.list_runs(derived_agent_key)
        options = pool.to_resume_options(entry) if prior_runs else pool.to_new_run_options(entry)
        outcome = await run_with_capture(pool, derived_agent_key, parsed.task, options)
        return {"content": [{"type": "text", "text": outcome.output}]}

    return create_sdk_mcp_server(ACQUIRE_SERVER, tools=[_acquire_capability])


def with_acquire_tool(
    options: ClaudeAgentOptions, router: CapabilityRouter, pool: AgentPool, agent_key: AgentKey
) -> ClaudeAgentOptions:
    """Augment already-built options with the shared acquire server and its tool grant.

    Mirrors `findings_tool.with_findings_tool`'s append-only `dataclasses.replace` discipline: the
    `acquire` server is merged into `options.mcp_servers` (existing servers preserved) and
    `ACQUIRE_TOOL` is granted once (existing grants preserved, no duplicate). The server is
    identity-free, so the single instance built here is safe for the main agent and every lens.

    Args:
        options: The already-built options to augment; not mutated.
        router: The capability router the acquire server equips agents from.
        pool: The pool acquired runs are recorded in.
        agent_key: The pooled agent acquired runs are tied into.

    Returns:
        A copy of `options` with the acquire server mounted and its tool granted.
    """
    server = build_acquire_server(router, pool, agent_key)
    existing = options.mcp_servers if isinstance(options.mcp_servers, dict) else {}
    mcp_servers = {**existing, ACQUIRE_SERVER: server}
    allowed_tools = list(options.allowed_tools)
    if ACQUIRE_TOOL not in allowed_tools:
        allowed_tools.append(ACQUIRE_TOOL)
    return dataclasses.replace(options, mcp_servers=mcp_servers, allowed_tools=allowed_tools)


def grant_acquire_to_subagent(definition: AgentDefinition) -> AgentDefinition:
    """Grant a subagent access to the parent-level shared acquire server, append-only and deduped.

    The SDK inherits a parent's in-process MCP server into a subagent only when the subagent's own
    `AgentDefinition.mcpServers` names it, so a lens can reach the `acquire` server only if this is
    applied: `ACQUIRE_SERVER` is appended to `mcpServers` and `ACQUIRE_TOOL` to `tools`, each once.
    When `tools` is None ("inherit everything") it is left None — the harness already grants the
    acquire tool in that case — so an inherit-all subagent is never narrowed to an explicit list.

    Unlike `findings_tool.grant_findings_to_subagent`, this takes no `session_id`: acquisition is
    not identity-sensitive (the shared server carries no per-caller identity), so there is nothing
    to thread through for attribution — a simpler signature.

    Args:
        definition: The subagent definition to grant acquire access to; not mutated.

    Returns:
        A copy of `definition` with the acquire server and tool grant added.
    """
    mcp_servers = list(definition.mcpServers) if definition.mcpServers is not None else []
    if ACQUIRE_SERVER not in mcp_servers:
        mcp_servers.append(ACQUIRE_SERVER)
    if definition.tools is None:
        tools = None
    else:
        tools = list(definition.tools)
        if ACQUIRE_TOOL not in tools:
            tools.append(ACQUIRE_TOOL)
    return dataclasses.replace(definition, mcpServers=mcp_servers, tools=tools)
