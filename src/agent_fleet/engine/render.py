from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import cast

from claude_agent_sdk import ClaudeAgentOptions
from claude_agent_sdk.types import AgentDefinition, ThinkingConfig

from capdisc.base import FrozenWireModel
from capdisc.catalog import ToolRef
from capdisc.hooks import HookConfig, HookEvent, MatcherGroup

from ..models.agent import AgentName, AgentSpec, ModelId

# The Claude Code harness renamed the subagent-delegation tool from "Task" to "Agent" in
# v2.1.63; current SDK releases require "Agent" in allowed_tools to grant it — do not revert.
SUBAGENT_TOOL = "Agent"

# The harness's native tool for continuing a specific previously-dispatched subagent by its id.
# Confirmed live: a resumed session calls `SendMessage(to=<agent_id>, summary=..., message=...)`
# to reach a backgrounded subagent from an earlier run, continuing that subagent's own
# conversation. `with_agent_resume` grants it only when actually resuming one such subagent.
SEND_MESSAGE_TOOL = "SendMessage"


def tool_grant(spec: AgentSpec) -> list[ToolRef]:
    """The Claude Agent SDK tool-grant for a spec — its named tools plus a `mcp__<server>__*`
    wildcard per selected MCP server.

    The single source of truth for `ClaudeAgentOptions.allowed_tools`. It lives here, not on
    `AgentSpec`, because the `mcp__<server>__*` wildcard is an SDK wire convention, not a domain
    fact — the spec keeps `tools` and `mcp_servers` separate.

    Args:
        spec: The agent spec whose tools and MCP servers form the grant.

    Returns:
        The merged grant list; empty when the spec names no tools and no MCP servers.
    """
    return [*spec.tools, *(f"mcp__{server}__*" for server in spec.mcp_servers)]


def to_options(spec: AgentSpec) -> ClaudeAgentOptions:
    """Build `ClaudeAgentOptions` from a spec — the single spec→SDK mapping.

    Used live by `engine.run.run_agent` and serialized to source by `render_claude_sdk`, so a
    run and an emitted program behave identically. Skills and the selected MCP servers load from
    the user's settings via `setting_sources`; an MCP server's command/url/credentials never enter
    from the spec.

    Args:
        spec: The agent spec to translate; `ModelId.inherit` and absent effort/permission/max-turns
            map to None so the SDK falls back to its own defaults.

    Returns:
        The options, with `setting_sources` set only when the spec uses skills or MCP servers.
    """
    uses_environment = bool(spec.skills or spec.mcp_servers)
    return ClaudeAgentOptions(
        system_prompt=spec.system_prompt,
        model=None if spec.model is ModelId.inherit else spec.model.value,
        effort=spec.effort.value if spec.effort is not None else None,
        # model_dump emits the SDK's ThinkingConfig TypedDict shape; cast translates it at the
        # SDK boundary (a Pydantic variant in, the SDK's TypedDict union out)
        thinking=cast(ThinkingConfig, spec.thinking.model_dump(mode="json", exclude_none=True))
        if spec.thinking is not None
        else None,
        allowed_tools=tool_grant(spec),
        disallowed_tools=list(spec.disallowed_tools),
        max_turns=spec.max_turns,
        permission_mode=spec.permission_mode.value if spec.permission_mode is not None else None,
        skills=list(spec.skills) or None,
        setting_sources=["user", "project"] if uses_environment else None,
    )


def to_agent_definition(spec: AgentSpec) -> AgentDefinition:
    """Build an `AgentDefinition` from a spec — the same spec→SDK mapping as `to_options`, targeting
    a subagent definition instead of top-level `ClaudeAgentOptions`.

    A spec run standalone (via `to_options`) and the same spec run as a subagent (via this) get
    equivalent grants and config: the tool-grant is `tool_grant(spec)` in both cases (its named
    tools plus the `mcp__<server>__*` wildcards), and `ModelId.inherit`/absent
    effort/memory/permission/max-turns map to None so the SDK falls back to its own defaults. The
    grant is passed through as-is — an empty list is a legitimate "no tools" grant, distinct from
    None ("inherit everything").

    Args:
        spec: The agent spec to translate into a subagent definition.

    Returns:
        The subagent definition the harness dispatches to when this spec is wired in as a subagent.
    """
    return AgentDefinition(
        description=spec.description,
        prompt=spec.system_prompt,
        tools=tool_grant(spec),
        disallowedTools=list(spec.disallowed_tools) or None,
        model=None if spec.model is ModelId.inherit else spec.model.value,
        skills=list(spec.skills) or None,
        memory=spec.memory.value if spec.memory is not None else None,
        mcpServers=list(spec.mcp_servers) or None,
        initialPrompt=spec.initial_prompt,
        maxTurns=spec.max_turns,
        background=spec.background,
        effort=spec.effort.value if spec.effort is not None else None,
        permissionMode=spec.permission_mode.value if spec.permission_mode is not None else None,
    )


def with_subagents(
    options: ClaudeAgentOptions, subagents: Mapping[AgentName, AgentSpec]
) -> ClaudeAgentOptions:
    """Augment already-built options so their main agent can dispatch to the given subagents.

    Wires the native Claude Code subagent capability onto options from `to_options` or the pool's
    `to_new_run_options`/`to_resume_options`, following the same "single mapping, no drift"
    `dataclasses.replace` discipline those pool builders use: each subagent spec is translated by
    `to_agent_definition`, and `SUBAGENT_TOOL` is granted (appended once, existing order preserved).

    The CONCURRENCY of dispatched subagents is the harness's own behavior once `agents` and the
    `SUBAGENT_TOOL` grant are present — this function only wires the capability in; it orchestrates
    nothing itself.

    Typical pool-driven composition::

        main = pool.to_resume_options(main_entry)
        subagents = {entry.name: entry.spec for entry in [sub_entry_1, sub_entry_2]}
        options = with_subagents(main, subagents)
        async for msg in query(prompt=task, options=options):
            ...

    Args:
        options: The already-built main-agent options to augment; not mutated.
        subagents: The subagent name → spec mapping to expose for dispatch; an empty mapping is a
            legitimate "no subagents configured yet" state and still grants `SUBAGENT_TOOL`.

    Returns:
        A copy of `options` with `agents` from the mapping and `SUBAGENT_TOOL` in `allowed_tools`.
    """
    agents = {name: to_agent_definition(spec) for name, spec in subagents.items()}
    allowed_tools = list(options.allowed_tools)
    if SUBAGENT_TOOL not in allowed_tools:
        allowed_tools.append(SUBAGENT_TOOL)
    return dataclasses.replace(options, agents=agents, allowed_tools=allowed_tools)


def with_agent_resume(options: ClaudeAgentOptions) -> ClaudeAgentOptions:
    """Grant `SEND_MESSAGE_TOOL` on already-built options so a run can resume a dispatched subagent.

    Least-privilege: `SendMessage` is the harness tool that continues one specific
    previously-dispatched subagent by its `AgentId`, so it is granted only for a run that actually
    resumes such a subagent — not on every subagent-capable run. Follows `with_subagents`' append-
    once discipline: `SEND_MESSAGE_TOOL` is added to `allowed_tools` only if absent, existing order
    preserved.

    Args:
        options: The already-built options to augment; not mutated.

    Returns:
        A copy of `options` with `SEND_MESSAGE_TOOL` in `allowed_tools`.
    """
    allowed_tools = list(options.allowed_tools)
    if SEND_MESSAGE_TOOL not in allowed_tools:
        allowed_tools.append(SEND_MESSAGE_TOOL)
    return dataclasses.replace(options, allowed_tools=allowed_tools)


class _HookSettingsFile(FrozenWireModel):
    """The `{"hooks": ...}` shape of a Claude Code settings JSON file — the one key the SDK's
    native `settings` loader reads for declarative hooks."""

    hooks: HookConfig


def _merge_hook_configs(configs: Iterable[HookConfig]) -> HookConfig | None:
    """Concatenate several hook configs into one, or None when none of them declare any hooks.

    `HookConfig` is `dict[HookEvent, list[MatcherGroup]]`, so the merge is per-event list
    concatenation: every matcher group fires regardless of which spec declared it — there is no
    override or conflict resolution to do.
    """
    merged: dict[HookEvent, list[MatcherGroup]] = {}
    for config in configs:
        for event, groups in config.root.items():
            merged.setdefault(event, []).extend(groups)
    return HookConfig(merged) if merged else None


def with_hooks(
    options: ClaudeAgentOptions,
    spec: AgentSpec,
    directory: Path,
    *,
    subagents: Mapping[AgentName, AgentSpec] | None = None,
) -> ClaudeAgentOptions:
    """Augment already-built options to load the main spec's (and any subagents') declarative hooks.

    Claude Code hooks are session-wide — a matching tool event fires the hook whichever agent (main
    or a dispatched subagent) triggered it — so one settings file loaded by the top-level options
    covers every agent in the run. The main spec's `hooks` and each subagent spec's `hooks` are
    therefore merged into ONE `<spec.name>.hooks.json` written under `directory`, and
    `options.settings` is pointed at it; the SDK's native `--settings` loader already understands
    the exact declarative shape `HookConfig` models, so no callback translation is needed. Follows
    the `with_subagents` append-only `dataclasses.replace` discipline.

    A no-op returning `options` unchanged — and writing no file — when neither the main spec nor any
    subagent declares hooks.

    Args:
        options: The already-built options to augment; not mutated.
        spec: The main agent spec whose hooks (when set) are loaded.
        directory: The directory the merged settings file is written to; created when missing.
        subagents: The dispatchable subagents whose own hooks fold into the same file; None or empty
            means the main spec's hooks alone.

    Returns:
        A copy of `options` with `settings` pointed at the written file, or `options` unchanged when
        there are no hooks to load.
    """
    specs = [spec, *(subagents.values() if subagents is not None else ())]
    merged = _merge_hook_configs(s.hooks for s in specs if s.hooks is not None)
    if merged is None:
        return options
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{spec.name}.hooks.json"
    path.write_text(
        _HookSettingsFile(hooks=merged).model_dump_json(by_alias=True, exclude_none=True),
        encoding="utf-8",
    )
    return dataclasses.replace(options, settings=str(path))


def _field_default(field: dataclasses.Field[object]) -> object:
    """The default value of a dataclass field (calling its factory), or None for a required one."""
    if field.default is not dataclasses.MISSING:
        return field.default
    if field.default_factory is not dataclasses.MISSING:
        return field.default_factory()
    return None


def render_claude_sdk(spec: AgentSpec, options: ClaudeAgentOptions | None = None) -> str:
    """Emit a runnable Claude Agent SDK program from the spec.

    Serializes the `ClaudeAgentOptions` that `to_options` builds — emitting only the options set to
    a meaningful, non-default value — so the emitted program and the live run can never drift.

    Args:
        spec: The agent spec to render; supplies the emitted module's name/description text.
        options: The options to serialize; `to_options(spec)` when None. `generate` passes options
            already augmented (e.g. `with_hooks`) so the emitted program carries them too.

    Returns:
        The Python source of a standalone, runnable agent module.
    """
    options = options if options is not None else to_options(spec)
    lines: list[str] = []
    # reflective serialization of the SDK options dataclass — the deliberate exception to the
    # no-dynamic-access rule, so the option list has exactly one author (to_options)
    for field in dataclasses.fields(options):
        value = getattr(options, field.name)
        if value not in (None, [], {}) and value != _field_default(field):
            lines.append(f"    {field.name}={value!r},")
    body = "\n".join(lines)
    return (
        "from __future__ import annotations\n\n"
        "import asyncio\n\n"
        "from claude_agent_sdk import ClaudeAgentOptions, query\n\n\n"
        f"# generated agent: {spec.name}\n"
        f"OPTIONS = ClaudeAgentOptions(\n{body}\n)\n\n\n"
        "async def run(task: str) -> None:\n"
        "    async for message in query(prompt=task, options=OPTIONS):\n"
        "        print(message)\n\n\n"
        'if __name__ == "__main__":\n'
        "    import sys\n\n"
        f"    asyncio.run(run(sys.argv[1] if len(sys.argv) > 1 else {spec.description!r}))\n"
    )
