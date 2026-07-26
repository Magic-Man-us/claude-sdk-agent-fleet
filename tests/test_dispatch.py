from __future__ import annotations

import asyncio
import sqlite3
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
from agent_fleet.engine.dispatch import prepare_provider_run, run_provider_capture, run_with_capture
from agent_fleet.models.agent import (
    RUN_OUTPUT_MAX,
    ClaudeRunRequest,
    CodexModelId,
    CodexRunConfig,
    CodexRunRequest,
    Provider,
    RunMode,
    RunScope,
)
from agent_fleet.router.capability import CapabilityRouter

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


def test_run_with_overlong_output_stores_bounded_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool = _pool(tmp_path)
    entry = pool.save(_AGENT_KEY, _spec())
    overlong = "x" * (RUN_OUTPUT_MAX + 500)
    messages: list[Message] = [_assistant(overlong, session_id=entry.session_id)]
    monkeypatch.setattr("agent_fleet.engine.dispatch.query", _fake_query(messages))

    outcome = asyncio.run(run_with_capture(pool, _AGENT_KEY, _TASK, pool.to_new_run_options(entry)))

    assert len(outcome.output) <= RUN_OUTPUT_MAX
    reread = pool.get_run(outcome.run.run_id)
    assert reread is not None
    assert reread.output == outcome.output

    # the row itself must already be bounded on write -- proves the write path validated rather
    # than relying solely on RunOutput's read-path truncation to mask an over-long stored value
    conn = sqlite3.connect(pool.db_path)
    stored = conn.execute(
        "SELECT output FROM runs WHERE run_id = ?", (outcome.run.run_id,)
    ).fetchone()[0]
    conn.close()
    assert len(stored) <= RUN_OUTPUT_MAX


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


def test_finished_run_row_carries_the_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool = _pool(tmp_path)
    entry = pool.save("PROJ-PERSIST", _spec())
    messages = [_assistant("captured text", session_id=entry.session_id)]
    monkeypatch.setattr("agent_fleet.engine.dispatch.query", _fake_query(messages))

    outcome = asyncio.run(
        run_with_capture(pool, entry.agent_key, _TASK, pool.to_new_run_options(entry))
    )
    stored = pool.get_run(outcome.run.run_id)
    assert stored is not None
    assert stored.output == "captured text"


def test_finished_run_row_carries_structured_output_and_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool = _pool(tmp_path)
    entry = pool.save("PROJ-PERSIST-FULL", _spec())
    messages: list[Message] = [
        _assistant("captured text", session_id=entry.session_id),
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

    outcome = asyncio.run(
        run_with_capture(pool, entry.agent_key, _TASK, pool.to_new_run_options(entry))
    )
    stored = pool.get_run(outcome.run.run_id)
    assert stored is not None
    assert stored.output == "captured text"
    assert stored.structured_output == {"verdict": "clean"}
    assert stored.total_cost_usd == 0.0042


def _router() -> CapabilityRouter:
    return CapabilityRouter([], [], [], [], {})


def _codex_config(tmp_path: Path) -> CodexRunConfig:
    return CodexRunConfig(
        cwd=tmp_path,
        scope=RunScope(mode=RunMode.read),
        model=CodexModelId.gpt_5_6_sol,
        timeout_s=60,
    )


def test_prepare_provider_run_claude_wraps_prepare_run_options(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    entry = pool.save(_AGENT_KEY, _spec())
    run, request, prompt = prepare_provider_run(pool, _router(), entry.agent_key, _TASK)

    assert isinstance(request, ClaudeRunRequest)
    assert request.options.resume is None  # first run: fresh session, not a resume
    assert prompt is None
    assert run.task == _TASK


def test_prepare_provider_run_codex_derives_developer_instructions_from_the_spec(
    tmp_path: Path,
) -> None:
    pool = _pool(tmp_path)
    entry = pool.save(_AGENT_KEY, _spec())
    run, request, prompt = prepare_provider_run(
        pool,
        _router(),
        entry.agent_key,
        _TASK,
        provider=Provider.codex,
        codex=_codex_config(tmp_path),
    )

    assert isinstance(request, CodexRunRequest)
    assert request.developer_instructions == _PROMPT
    assert prompt is None
    assert run.task == _TASK
    assert len(pool.list_runs(_AGENT_KEY)) == 1  # start_run was called exactly once


def test_prepare_provider_run_codex_requires_codex_settings(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    entry = pool.save(_AGENT_KEY, _spec())
    with pytest.raises(ValueError, match="codex run settings are required"):
        prepare_provider_run(pool, _router(), entry.agent_key, _TASK, provider=Provider.codex)


def test_prepare_provider_run_codex_rejects_resume_agent_id(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    entry = pool.save(_AGENT_KEY, _spec())
    with pytest.raises(ValueError, match="neither subagent dispatch nor resume_agent_id"):
        prepare_provider_run(
            pool,
            _router(),
            entry.agent_key,
            _TASK,
            provider=Provider.codex,
            codex=_codex_config(tmp_path),
            resume_agent_id="abc123def456",
        )


def test_prepare_provider_run_codex_rejects_subagent_keys(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    entry = pool.save(_AGENT_KEY, _spec())
    with pytest.raises(ValueError, match="neither subagent dispatch nor resume_agent_id"):
        prepare_provider_run(
            pool,
            _router(),
            entry.agent_key,
            _TASK,
            provider=Provider.codex,
            codex=_codex_config(tmp_path),
            subagent_agent_keys={"reviewer": "teammate.reviewer"},
        )


def test_run_provider_capture_dispatches_claude_requests_through_run_with_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool = _pool(tmp_path)
    entry = pool.save(_AGENT_KEY, _spec())
    messages: list[Message] = [
        _assistant("hello via provider dispatch", session_id=entry.session_id)
    ]
    monkeypatch.setattr("agent_fleet.engine.dispatch.query", _fake_query(messages))

    request = ClaudeRunRequest(options=pool.to_new_run_options(entry))
    outcome = asyncio.run(run_provider_capture(pool, _AGENT_KEY, _TASK, request))

    assert outcome.output == "hello via provider dispatch"


def test_run_provider_capture_dispatches_codex_requests_through_run_codex_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool = _pool(tmp_path)
    entry = pool.save(_AGENT_KEY, _spec())
    captured: dict[str, object] = {}

    async def _fake_codex(pool_arg, agent_key_arg, task_arg, request_arg, *, run=None):
        captured["agent_key"] = agent_key_arg
        captured["request"] = request_arg
        started = pool_arg.start_run(agent_key_arg, task_arg)
        return RunOutcome(
            output="codex result",
            run=pool_arg.finish_run(started.run_id, output="codex result"),
            agent_runs=[],
        )

    monkeypatch.setattr("agent_fleet.engine.dispatch.run_codex_capture", _fake_codex)
    request = CodexRunRequest(
        cwd=tmp_path,
        scope=RunScope(mode=RunMode.read),
        model=CodexModelId.gpt_5_6_sol,
        timeout_s=60,
        developer_instructions=_PROMPT,
    )
    outcome = asyncio.run(run_provider_capture(pool, entry.agent_key, _TASK, request))

    assert outcome.output == "codex result"
    assert captured["agent_key"] == entry.agent_key
    assert captured["request"] is request
