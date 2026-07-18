from __future__ import annotations

import dataclasses

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    Message,
    ResultMessage,
    TaskStartedMessage,
    TextBlock,
    ToolUseBlock,
    query,
)
from pydantic import JsonValue, TypeAdapter

from ..models.agent import (
    AgentId,
    AgentKey,
    AgentName,
    AgentSpec,
    RunOutcome,
    RunRecord,
    SessionId,
    TaskBrief,
)
from ..router.capability import CapabilityRouter
from .findings_tool import grant_findings_to_subagent, with_findings_tool
from .pool import AgentPool
from .render import SUBAGENT_TOOL, with_agent_resume, with_subagents

_TASK_BRIEF_ADAPTER: TypeAdapter[TaskBrief] = TypeAdapter(TaskBrief)


def build_resume_prompt(agent_id: AgentId, task: TaskBrief) -> TaskBrief:
    """Build the literal turn that resumes one dispatched subagent, validated as a `TaskBrief`.

    The harness continues a specific backgrounded subagent when a resumed session's turn names its
    `AgentId` and calls `SendMessage`; this composes that turn (`"Resume agent {agent_id} and now:
    {task}"`) and validates it back through `TaskBrief` rather than constructing it unchecked.
    Shared by every caller that offers subagent-resume (the MCP pool server, the HTTP API) so the
    wrapper-turn format stays in one place.

    Args:
        agent_id: The dispatched subagent to continue (from an earlier `AgentRunRecord.agent_id`).
        task: The new work to hand that subagent.

    Returns:
        The validated literal prompt to send this turn.
    """
    return _TASK_BRIEF_ADAPTER.validate_python(f"Resume agent {agent_id} and now: {task}")


# The Claude Code harness renamed the subagent-delegation tool from "Task" to "Agent" in v2.1.63.
# `render.SUBAGENT_TOOL` is the single current grant name; detection tolerates both so a stream
# emitted by either harness version is captured, mirroring the SDK docs' own `block.name in
# ("Task", "Agent")` example.
SUBAGENT_TOOL_NAMES = frozenset({"Task", SUBAGENT_TOOL})

_SUBAGENT_TYPE_KEY = "subagent_type"


def _extract_text(msg: Message) -> list[str]:
    """Pull text strings out of one AssistantMessage; other message types return empty.

    Duplicated from `engine.orchestrate` rather than imported: `orchestrate` imports from `.run`
    and this module imports from `.pool`/`.render`, and keeping the tiny helper local avoids
    coupling the capture path to the orchestrator's import graph.
    """
    if isinstance(msg, AssistantMessage):
        return [block.text for block in msg.content if isinstance(block, TextBlock)]
    return []


def _message_session(msg: Message) -> tuple[SessionId | None, str | None]:
    """The (session_id, parent_tool_use_id) a message carries, or None where it carries neither.

    Only `AssistantMessage` and `ResultMessage` carry a `session_id`; `parent_tool_use_id` rides on
    `AssistantMessage` and is absent on `ResultMessage` (whose session is always the top-level one).
    Narrowing by type rather than reflective attribute access keeps every field access explicit.
    """
    if isinstance(msg, AssistantMessage):
        return msg.session_id, msg.parent_tool_use_id
    if isinstance(msg, ResultMessage):
        return msg.session_id, None
    return None, None


def _track_dispatches(msg: Message, open_dispatches: dict[str, AgentName | None]) -> None:
    """Record any subagent-dispatch tool-use blocks in `msg` into `open_dispatches` by block id.

    The dispatched agent's session id later threads back through `parent_tool_use_id` matching one
    of these ids; the value is the block's `subagent_type` (None when the harness omits it).
    """
    if not isinstance(msg, AssistantMessage):
        return
    for block in msg.content:
        if isinstance(block, ToolUseBlock) and block.name in SUBAGENT_TOOL_NAMES:
            subagent_type = block.input.get(_SUBAGENT_TYPE_KEY)
            open_dispatches[block.id] = subagent_type


async def run_with_capture(
    pool: AgentPool,
    agent_key: AgentKey,
    task: TaskBrief,
    options: ClaudeAgentOptions,
    *,
    run: RunRecord | None = None,
    prompt: TaskBrief | None = None,
) -> RunOutcome:
    """Run an agent live and record every agent that participates, main and dispatched.

    A dispatched subagent does NOT get its own independently-resumable session: every agent in a
    run — the main/top-level agent and each subagent it dispatches via the `Agent`/`Task` tool —
    shares the SAME `session_id`, which identifies the pool entry's conversation, not an individual
    subagent. What uniquely identifies a dispatched subagent is the harness's own `task_id`, emitted
    once per dispatch on a `TaskStartedMessage` (tagged with the dispatching tool call via
    `TaskStartedMessage.tool_use_id`). This captures that `task_id` as the row's `AgentId` — the
    durable handle to continue that one subagent later via the harness's `SendMessage` tool. The
    pairing for acting on it is `render.with_agent_resume` (grant `SendMessage`) plus a resumed
    parent session that names the `AgentId` in its prompt text.

    One `agent_runs` row is persisted per agent: the main agent (an `AssistantMessage` with
    `parent_tool_use_id is None`, carrying the run's session id) plus each dispatched subagent (its
    `TaskStartedMessage` whose `tool_use_id` matches a tracked `Agent`/`Task` dispatch). A
    tracked-dispatch dict maps each subagent tool-use block id to its `subagent_type`; a set of
    already-seen `task_id`s dedupes subagent rows (mostly a safety net — `TaskStartedMessage` fires
    once per dispatch), and the main session id is deduped so it records a single main row. If the
    first observed main session id differs from what `options` asked to resume/pin, the pool is
    reconciled to the observed id — the self-healing step for a resume that returned a fresh session
    (the cwd-mismatch failure mode).

    The run's terminal `ResultMessage` (there is exactly one per top-level `query()` stream — a
    dispatched subagent's own completion surfaces as an `AssistantMessage`/`ToolResultBlock` to its
    parent, never a second top-level `ResultMessage`) also carries `structured_output` and
    `total_cost_usd` when `options.output_format` requested forced structured output; both are
    captured onto the returned `RunOutcome` so a caller using `output_format` has a supported way
    to get the result out, instead of re-walking the message stream itself.

    Args:
        pool: The sync pool the run and its captured agents are recorded in.
        agent_key: The pooled agent this run belongs to.
        task: The run's recorded/logical task — what `pool.start_run` logs — and the literal first
            user turn when `prompt` is not given.
        options: The already-built live options (from the pool's option builders, optionally wrapped
            by `with_subagents`/`with_agent_resume`).
        run: An already-started run record, or None to start one here. Left None, this mints a
            fresh run via `pool.start_run(agent_key, task)` — the path every plain caller takes. A
            caller passes an already-started run only when it needed the run id before the stream
            began, to build `options` around it: `with_findings_tool` requires a `run_id` at wiring
            time, so wiring the run-scoped findings tool means starting the run first, then handing
            the record in here so no second, duplicate run is opened.
        prompt: An optional literal first turn to send this run instead of `task`. Resuming a
            specific dispatched subagent needs a wrapper turn (e.g. `"Resume agent {agent_id} and
            now: {task}"`) sent verbatim, while the run record must still log the caller's real
            `task`, not the wrapper — so the recorded task and the sent text are decoupled here.

    Returns:
        The run's collected assistant text, the finished run record, every captured agent run, and
        — when produced — the terminal result's structured output and total cost.
    """
    run = run if run is not None else pool.start_run(agent_key, task)
    literal_prompt = prompt if prompt is not None else task
    expected_session = options.resume or options.session_id
    open_dispatches: dict[str, AgentName | None] = {}
    recorded_main: set[SessionId] = set()
    seen_tasks: set[str] = set()
    main_session_id: SessionId | None = None
    main_recorded = False
    parts: list[str] = []
    structured_output: JsonValue | None = None
    total_cost_usd: float | None = None
    async for message in query(prompt=literal_prompt, options=options):
        parts.extend(_extract_text(message))
        _track_dispatches(message, open_dispatches)
        if isinstance(message, ResultMessage):
            structured_output = message.structured_output
            total_cost_usd = message.total_cost_usd
        if (
            isinstance(message, TaskStartedMessage)
            and message.tool_use_id is not None
            and message.tool_use_id in open_dispatches
            and message.task_id not in seen_tasks
            and main_session_id is not None
        ):
            seen_tasks.add(message.task_id)
            pool.record_agent_run(
                run.run_id,
                main_session_id,
                tool_use_id=message.tool_use_id,
                agent_name=open_dispatches[message.tool_use_id],
                agent_id=message.task_id,
            )
        session_id, parent_tool_use_id = _message_session(message)
        if session_id is None:
            continue
        if parent_tool_use_id is None:
            main_session_id = session_id
            if session_id not in recorded_main:
                recorded_main.add(session_id)
                pool.record_agent_run(run.run_id, session_id)
            if not main_recorded:
                main_recorded = True
                if expected_session is not None and session_id != expected_session:
                    pool.reconcile_session(agent_key, session_id)
    finished = pool.finish_run(run.run_id)
    return RunOutcome(
        output="\n".join(parts),
        run=finished,
        agent_runs=pool.list_agent_runs(run.run_id),
        structured_output=structured_output,
        total_cost_usd=total_cost_usd,
    )


def prepare_run(
    pool: AgentPool,
    capability_router: CapabilityRouter,
    agent_key: AgentKey,
    task: TaskBrief,
    *,
    subagent_agent_keys: dict[AgentName, AgentKey] | None = None,
    resume_agent_id: AgentId | None = None,
) -> tuple[RunRecord, ClaudeAgentOptions, TaskBrief | None]:
    """Start a run and assemble its fully-wired live options, ready to hand to `run_with_capture`.

    The single shared preparation sequence behind both live-run entry points (the MCP `run_agent`
    tool and the HTTP `run_pool_entry` route): look up the pooled entry, decide resume-vs-fresh from
    its run history, start the run (so `with_findings_tool` has a run id at wiring time), resolve
    and wire any named subagents, grant the run-scoped findings-writer and the acquire tool to main
    agent and — when subagents are present — to every subagent definition, and, when resuming one
    dispatched subagent, grant `SendMessage` and build the wrapper turn. The run is started here
    rather than inside `run_with_capture`, and handed back so the caller passes the same record on.

    Missing entries surface as `KeyError` carrying the absent `AgentKey`: the main entry raises
    `KeyError(agent_key)`, a missing subagent raises `KeyError(sub_agent_key)`. The two are
    distinguishable by comparing the raised key against `agent_key` — a subagent lookup can only
    fail after the main entry resolved, so a subagent key equal to `agent_key` would itself resolve
    and never raise, meaning a subagent `KeyError` never carries `agent_key`.

    Args:
        pool: The sync pool the run is recorded in and every entry is resolved from.
        capability_router: The router the acquire tool equips acquired agents from.
        agent_key: The pooled agent to run; its entry must exist.
        task: The first user turn and the run's recorded task.
        subagent_agent_keys: A subagent name → pooled `agent_key` mapping to wire in as dispatchable
            subagents; None or empty runs the agent solo.
        resume_agent_id: When set, resume this specific previously-dispatched subagent — grant
            `with_agent_resume` and wrap `task` as the resume turn while still recording `task`.

    Returns:
        The started run record, the fully-wired live options, and the literal prompt to send (None
        to send `task` verbatim).

    Raises:
        KeyError: Carrying `agent_key` when the main entry is absent, or the missing subagent's
            `AgentKey` when a named subagent entry is absent.
    """
    # Deferred to break the acquire_tool -> dispatch (run_with_capture) import cycle.
    from .acquire_tool import grant_acquire_to_subagent, with_acquire_tool  # noqa: PLC0415

    entry = pool.get_by_key(agent_key)
    if entry is None:
        raise KeyError(agent_key)
    run, options = pool.begin_run(entry, task)
    if subagent_agent_keys:
        subagents: dict[AgentName, AgentSpec] = {}
        for name, sub_agent_key in subagent_agent_keys.items():
            sub_entry = pool.get_by_key(sub_agent_key)
            if sub_entry is None:
                raise KeyError(sub_agent_key)
            subagents[name] = sub_entry.spec
        options = with_subagents(options, subagents)
    options = with_findings_tool(
        options, pool, agent_key, run.run_id, entry.session_id, agent_name=None
    )
    options = with_acquire_tool(options, capability_router, pool, agent_key)
    if options.agents:
        agents = {
            name: grant_acquire_to_subagent(grant_findings_to_subagent(definition))
            for name, definition in options.agents.items()
        }
        options = dataclasses.replace(options, agents=agents)
    prompt: TaskBrief | None = None
    if resume_agent_id is not None:
        options = with_agent_resume(options)
        prompt = build_resume_prompt(resume_agent_id, task)
    return run, options, prompt
