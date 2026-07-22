"""Claude Agent SDK execution with durable Fleet runtime capture."""

from __future__ import annotations

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
from pydantic import JsonValue

from .models import RuntimeRunOutcome, RuntimeRunRecord
from .pool import RuntimeAgentPool


def _message_session(message: Message) -> tuple[str | None, str | None]:
    if isinstance(message, AssistantMessage):
        return message.session_id, message.parent_tool_use_id
    if isinstance(message, ResultMessage):
        return message.session_id, None
    return None, None


async def run_runtime_with_capture(
    pool: RuntimeAgentPool,
    agent_key: str,
    task: str,
    options: ClaudeAgentOptions,
    *,
    run: RuntimeRunRecord | None = None,
) -> RuntimeRunOutcome:
    """Execute one fixed runtime role and persist session/run metadata."""
    active_run = run if run is not None else pool.start_run(agent_key, task)
    expected_session = options.resume or options.session_id
    open_dispatches: dict[str, str | None] = {}
    seen_agent_ids: set[str] = set()
    seen_main_sessions: set[str] = set()
    main_session_id: str | None = None
    parts: list[str] = []
    structured_output: JsonValue | None = None
    total_cost_usd: float | None = None
    try:
        async for message in query(prompt=task, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        parts.append(block.text)
                    elif isinstance(block, ToolUseBlock) and block.name in {"Agent", "Task"}:
                        raw_name = block.input.get("subagent_type")
                        open_dispatches[block.id] = (
                            raw_name if isinstance(raw_name, str) else None
                        )
            if isinstance(message, ResultMessage):
                structured_output = message.structured_output
                total_cost_usd = message.total_cost_usd
            if (
                isinstance(message, TaskStartedMessage)
                and message.tool_use_id is not None
                and message.tool_use_id in open_dispatches
                and message.task_id not in seen_agent_ids
                and main_session_id is not None
            ):
                seen_agent_ids.add(message.task_id)
                pool.record_agent_run(
                    active_run.run_id,
                    main_session_id,
                    tool_use_id=message.tool_use_id,
                    agent_name=open_dispatches[message.tool_use_id],
                    agent_id=message.task_id,
                )
            session_id, parent_tool_use_id = _message_session(message)
            if session_id is None or parent_tool_use_id is not None:
                continue
            main_session_id = session_id
            if session_id not in seen_main_sessions:
                seen_main_sessions.add(session_id)
                pool.record_agent_run(active_run.run_id, session_id)
            if expected_session is not None and session_id != expected_session:
                pool.reconcile_session(agent_key, session_id)
                expected_session = session_id
    finally:
        finished = pool.finish_run(active_run.run_id)
    return RuntimeRunOutcome(
        output="\n".join(parts),
        run=finished,
        agent_runs=pool.list_agent_runs(active_run.run_id),
        structured_output=structured_output,
        total_cost_usd=total_cost_usd,
    )
