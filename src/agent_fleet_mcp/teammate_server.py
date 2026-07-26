"""The teammate MCP server: the one way to run a named specialist.

Four tools — read the roster, check one teammate, run one, dismiss one. Everything that used to
be a separate verb is a flag on `run_teammate`.
"""

from __future__ import annotations

import asyncio

from fastmcp import FastMCP

from agent_fleet.engine.dispatch import prepare_provider_run, run_provider_capture
from agent_fleet.engine.teammates import resolve_template
from agent_fleet.models.agent import (
    AgentId,
    AwaitRun,
    CodexRunConfig,
    Provider,
    ResumeSession,
    RosterEntry,
    TeammateName,
    TeammateStatus,
    TeammateTurn,
    teammate_key,
)

from . import context

mcp = FastMCP("teammates")


@mcp.tool
def roster() -> list[RosterEntry]:
    """Every teammate template plus its live pool state — the teammate directory.

    Returns:
        One row per roster template, in roster order: the template, the derived status of its
        latest run (`unspawned` before first spawn), and its session id once it exists.
    """
    entries: list[RosterEntry] = []
    for template in context.roster_in_force().teammates:
        status = context.teammate_status(template.name)
        entries.append(
            RosterEntry(template=template, status=status.status, session_id=status.session_id)
        )
    return entries


@mcp.tool
def check_teammate(name: TeammateName) -> TeammateStatus:
    """The teammate's latest-run status; persisted output/structured output/cost once finished,
    or its captured error once failed.

    `stale` means an unfinished run this server process doesn't own — this process died mid-run,
    or another process is running it (`pool.db` can be shared) — the session itself is intact, so
    `run_teammate` resumes the conversation. `failed` means the run raised; its text is on
    `TeammateStatus.error`.

    Args:
        name: The teammate to check; valid names come from `roster()`.

    Raises:
        ValueError: When `name` is not on the roster.
    """
    resolve_template(
        name, context.roster_in_force()
    )  # roster gate: an off-roster name raises before any pool read
    return context.teammate_status(name)


@mcp.tool
async def run_teammate(
    name: TeammateName,
    task: TeammateTurn,
    resume: ResumeSession = True,
    wait: AwaitRun = False,
    resume_agent_id: AgentId | None = None,
    provider: Provider | None = None,
    codex: CodexRunConfig | None = None,
) -> TeammateStatus:
    """Run `task` on the teammate — the one way to invoke one.

    There is deliberately no separate spawn-versus-message pair: standing a teammate up and
    sending it another turn are the same operation, differing only in whether the existing session
    continues. The entry is created from its roster template when absent, so the first call and
    the hundredth look identical to the caller.

    When the teammate's latest run is still live, `task` is NOT queued: the returned status
    describes that already-running run rather than opening a second one against the same session
    — re-send after it finishes. That check runs before `resume` takes effect, so a live run's
    session cannot be reset out from under it. With `wait=True` against a busy teammate, this
    waits for the LIVE run (not your turn — it was never sent) and reports its terminal status.

    Deliberately `async def` with no `await` before the spawn: `TeammateRunner.spawn` needs the
    currently-running event loop, which exists only because this coroutine executes on it — a
    threadpool execution (e.g. an `asyncio.to_thread` wrapper) would break it.

    Args:
        name: The teammate to run; valid names come from `roster()`.
        task: The turn to send. On a resumed session this is conversational — it continues an
            existing thread rather than restating the teammate's job.
        resume: Continue the teammate's standing session, keeping everything it already knows.
            False mints a new session UUID first, discarding that history — use it when the
            accumulated context has gone stale rather than helpful. Ignored while a run is live.
        wait: Block until the run finishes and return its outcome on the status, re-raising
            whatever the run raised (the row is stamped `failed` with the error text first).
            False backgrounds the run and returns immediately.
        resume_agent_id: Continue one specific previously-dispatched subagent of this teammate.
            Claude only — raises if given together with `provider=codex`.
        provider: Which backend runs this turn; omitted defers to the roster template's
            own `provider`, so a teammate declared as codex runs on codex without
            restating it here. `resume` is accepted but inert for `codex`: every
            Codex call is an independent turn, since Codex's `exec` mode takes no thread-id input.
        codex: Codex run settings (cwd, scope, model, timeout); required when `provider` is
            `codex`, ignored otherwise.

    Raises:
        ValueError: When `name` is not on the roster, or `provider=codex` is combined with
            `resume_agent_id`, a teammate whose toolkits wire subagents, or a missing `codex`.
    """
    template = resolve_template(name, context.roster_in_force())
    key = teammate_key(name)
    live = context.live_run(key)
    if live is not None:
        if wait:
            live_task = context.runner().task_for(live.run_id)
            if live_task is not None:
                # exceptions are already stamped+logged by record_failure/_deregister; this
                # caller is a bystander to someone else's run and must not receive its failure
                await asyncio.gather(live_task, return_exceptions=True)
        return context.teammate_status(name)
    entry = context.ensure_teammate(name, fresh_session=not resume)
    # An omitted provider defers to the roster: a template that declares codex would otherwise run
    # on Claude without a word, which makes the declaration a lie rather than a default.
    provider = provider if provider is not None else template.provider
    run, request, _prompt = prepare_provider_run(
        context.pool(),
        context.capability_router(),
        entry.agent_key,
        task,
        subagent_agent_keys=context.subagent_keys(name),
        resume_agent_id=resume_agent_id,
        extra_hooks=context.notify_hooks(),
        provider=provider,
        codex=codex,
    )
    coro = context.record_failure(
        run.run_id,
        run_provider_capture(context.pool(), entry.agent_key, task, request, run=run),
    )
    # registered via the runner (not a bare `await coro`) so a concurrent `check_teammate` sees
    # `running`, not `stale`, for the whole duration of the wait
    task_handle = context.runner().spawn(run.run_id, coro)
    if wait:
        await task_handle
    return context.teammate_status(name)


@mcp.tool
def dismiss_teammate(name: TeammateName) -> bool:
    """Forget the teammate's standing session, discarding its conversation and run history.

    The roster entry is untouched — the next `run_teammate` rebuilds the teammate from its
    template with a clean session. This is the deliberate way to clear context that has gone
    stale; it is not a way to remove a teammate from the roster, which is the file's job.

    Args:
        name: The teammate to dismiss; valid names come from `roster()`.

    Returns:
        True when a standing session existed and was discarded, False when there was none.

    Raises:
        ValueError: When `name` is not on the roster.
    """
    resolve_template(name, context.roster_in_force())
    return context.pool().delete(teammate_key(name))


def main() -> None:
    """Run the teammate MCP server over stdio (the `teammate-mcp` console entry point)."""
    mcp.run()


if __name__ == "__main__":
    main()
