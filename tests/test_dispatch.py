from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    Message,
    ResultMessage,
    TaskStartedMessage,
    TextBlock,
    ToolUseBlock,
)

from agent_fleet import AgentPool, AgentSpec, RunOutcome
from agent_fleet.engine.dispatch import run_with_capture

_PROMPT = "You are auditor. Audit the code for vulnerabilities and stop."
_AGENT_KEY = "PROJ-4821"
_TASK = "audit the codebase for security vulnerabilities now"


def _spec() -> AgentSpec:
    return AgentSpec(
        name="auditor",
        description="Audits code for vulnerabilities.",
        system_prompt=_PROMPT,
        tools=("Read", "Grep"),
    )


def _pool(tmp_path: Path) -> AgentPool:
    return AgentPool(tmp_path / "pool.db")


def _assistant(
    text: str, *, session_id: str, parent_tool_use_id: str | None = None
) -> AssistantMessage:
    return AssistantMessage(
        content=[TextBlock(text=text)],
        model="test-model",
        session_id=session_id,
        parent_tool_use_id=parent_tool_use_id,
    )


def _dispatch(session_id: str, tool_use_id: str, subagent_type: str) -> AssistantMessage:
    return AssistantMessage(
        content=[
            ToolUseBlock(id=tool_use_id, name="Agent", input={"subagent_type": subagent_type})
        ],
        model="test-model",
        session_id=session_id,
        parent_tool_use_id=None,
    )


def _task_started(tool_use_id: str, task_id: str, session_id: str) -> TaskStartedMessage:
    return TaskStartedMessage(
        subtype="task_started",
        data={},
        task_id=task_id,
        description="dispatched subagent",
        uuid=str(uuid.uuid4()),
        session_id=session_id,
        tool_use_id=tool_use_id,
        task_type="local_agent",
    )


def _fake_query(messages: list[Message]) -> object:
    async def _query(**kwargs: object) -> AsyncIterator[Message]:
        for message in messages:
            yield message

    return _query


def _fake_query_capturing(messages: list[Message], captured: list[dict[str, object]]) -> object:
    async def _query(**kwargs: object) -> AsyncIterator[Message]:
        captured.append(dict(kwargs))
        for message in messages:
            yield message

    return _query


def test_run_with_no_dispatched_agents_records_one_main_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool = _pool(tmp_path)
    entry = pool.save(_AGENT_KEY, _spec())
    main_session = entry.session_id
    messages: list[Message] = [
        _assistant("hello", session_id=main_session),
        _assistant("world", session_id=main_session),
    ]
    monkeypatch.setattr("agent_fleet.engine.dispatch.query", _fake_query(messages))
    outcome = asyncio.run(run_with_capture(pool, _AGENT_KEY, _TASK, pool.to_new_run_options(entry)))

    assert isinstance(outcome, RunOutcome)
    assert "hello" in outcome.output
    assert "world" in outcome.output
    assert len(outcome.agent_runs) == 1
    assert outcome.agent_runs[0].tool_use_id is None
    assert outcome.agent_runs[0].session_id == main_session
    assert outcome.run.finished_at is not None


def test_run_with_dispatched_agent_records_two_linked_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool = _pool(tmp_path)
    entry = pool.save(_AGENT_KEY, _spec())
    main_session = entry.session_id
    messages: list[Message] = [
        _assistant("planning", session_id=main_session),
        _dispatch(main_session, tool_use_id="toolu_1", subagent_type="reviewer"),
        _task_started(tool_use_id="toolu_1", task_id="abc123def456", session_id=main_session),
        # the subagent's own reply shares the parent's session — no separate resumable session
        _assistant("sub result", session_id=main_session, parent_tool_use_id="toolu_1"),
    ]
    monkeypatch.setattr("agent_fleet.engine.dispatch.query", _fake_query(messages))
    outcome = asyncio.run(run_with_capture(pool, _AGENT_KEY, _TASK, pool.to_new_run_options(entry)))

    rows = outcome.agent_runs
    assert len(rows) == 2
    assert rows[0].tool_use_id is None and rows[0].session_id == main_session
    assert rows[0].agent_id is None
    assert rows[1].tool_use_id == "toolu_1"
    assert rows[1].agent_name == "reviewer"
    assert rows[1].agent_id == "abc123def456"
    assert rows[1].session_id == main_session  # same session as the main agent, not a distinct one


def test_prompt_override_is_sent_while_task_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool = _pool(tmp_path)
    entry = pool.save(_AGENT_KEY, _spec())
    override = "Resume agent abc123def456 and now: keep auditing the codebase"
    captured: list[dict[str, object]] = []
    messages: list[Message] = [_assistant("resumed", session_id=entry.session_id)]
    monkeypatch.setattr(
        "agent_fleet.engine.dispatch.query", _fake_query_capturing(messages, captured)
    )

    outcome = asyncio.run(
        run_with_capture(pool, _AGENT_KEY, _TASK, pool.to_new_run_options(entry), prompt=override)
    )

    assert len(captured) == 1
    assert captured[0]["prompt"] == override  # the literal override text was sent to query()
    assert outcome.run.task == _TASK  # the recorded run task stays the caller's original task
    assert pool.list_runs(_AGENT_KEY)[0].task == _TASK


def test_repeated_session_ids_are_deduped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pool = _pool(tmp_path)
    entry = pool.save(_AGENT_KEY, _spec())
    main_session = entry.session_id
    messages: list[Message] = [
        _assistant("a", session_id=main_session),
        _assistant("b", session_id=main_session),
        _assistant("c", session_id=main_session),
    ]
    monkeypatch.setattr("agent_fleet.engine.dispatch.query", _fake_query(messages))
    outcome = asyncio.run(run_with_capture(pool, _AGENT_KEY, _TASK, pool.to_new_run_options(entry)))
    assert len(outcome.agent_runs) == 1  # one row despite three messages


def test_given_run_reuses_the_started_run_not_a_second(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool = _pool(tmp_path)
    entry = pool.save(_AGENT_KEY, _spec())
    run = pool.start_run(_AGENT_KEY, _TASK)  # caller starts the run itself
    messages: list[Message] = [_assistant("hi", session_id=entry.session_id)]
    monkeypatch.setattr("agent_fleet.engine.dispatch.query", _fake_query(messages))
    outcome = asyncio.run(
        run_with_capture(pool, _AGENT_KEY, _TASK, pool.to_new_run_options(entry), run=run)
    )

    assert outcome.run.run_id == run.run_id
    assert len(pool.list_runs(_AGENT_KEY)) == 1  # no second, duplicate run was minted


def test_captures_structured_output_and_cost_from_the_terminal_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool = _pool(tmp_path)
    entry = pool.save(_AGENT_KEY, _spec())
    messages: list[Message] = [
        _assistant("done", session_id=entry.session_id),
        ResultMessage(
            subtype="success",
            duration_ms=100,
            duration_api_ms=80,
            is_error=False,
            num_turns=1,
            session_id=entry.session_id,
            structured_output={"verdict": "clean"},
            total_cost_usd=0.0042,
        ),
    ]
    monkeypatch.setattr("agent_fleet.engine.dispatch.query", _fake_query(messages))
    outcome = asyncio.run(run_with_capture(pool, _AGENT_KEY, _TASK, pool.to_new_run_options(entry)))

    assert outcome.structured_output == {"verdict": "clean"}
    assert outcome.total_cost_usd == 0.0042


def test_structured_output_and_cost_are_none_without_a_result_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool = _pool(tmp_path)
    entry = pool.save(_AGENT_KEY, _spec())
    messages: list[Message] = [_assistant("done", session_id=entry.session_id)]
    monkeypatch.setattr("agent_fleet.engine.dispatch.query", _fake_query(messages))
    outcome = asyncio.run(run_with_capture(pool, _AGENT_KEY, _TASK, pool.to_new_run_options(entry)))

    assert outcome.structured_output is None
    assert outcome.total_cost_usd is None


def test_main_session_drift_reconciles_the_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool = _pool(tmp_path)
    entry = pool.save(_AGENT_KEY, _spec())
    expected = entry.session_id
    observed = str(uuid.uuid4())  # the live run reports a different main session
    messages: list[Message] = [_assistant("drifted", session_id=observed)]
    monkeypatch.setattr("agent_fleet.engine.dispatch.query", _fake_query(messages))

    # options pin the entry's expected session; the observed one differs -> reconcile
    asyncio.run(run_with_capture(pool, _AGENT_KEY, _TASK, pool.to_new_run_options(entry)))

    reconciled = pool.get_by_key(_AGENT_KEY)
    assert reconciled is not None
    assert reconciled.session_id == observed
    assert reconciled.session_id != expected
