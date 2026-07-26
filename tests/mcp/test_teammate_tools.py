from __future__ import annotations

import asyncio
import logging
import sqlite3
from pathlib import Path

import pytest
from claude_agent_sdk import ClaudeAgentOptions, Message

from agent_fleet import AgentPool
from agent_fleet.engine.render import SEND_MESSAGE_TOOL
from agent_fleet.engine.source import InMemoryCatalogSource
from agent_fleet.engine.teammates import DEFAULT_ROSTER
from agent_fleet.models.agent import (
    RUN_ERROR_MAX,
    CodexModelId,
    CodexRunConfig,
    CodexRunRequest,
    Provider,
    RunMode,
    RunOutcome,
    RunScope,
    TeammateRunStatus,
    teammate_key,
)
from agent_fleet.router.capability import CapabilityRouter
from agent_fleet_mcp import context, teammate_server
from agent_fleet_mcp.runner import TeammateRunner
from capdisc.catalog import Catalog
from test_dispatch import _assistant

_NAME = DEFAULT_ROSTER.teammates[0].name
_TASK = "run the roster teammate against a synthetic task"


@pytest.fixture
def pool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AgentPool:
    built = AgentPool(tmp_path / "pool.db")
    runner = TeammateRunner()
    monkeypatch.setattr(context, "pool", lambda: built)
    monkeypatch.setattr(context, "source", lambda: InMemoryCatalogSource(Catalog(entries=[])))
    monkeypatch.setattr(context, "capability_router", lambda: CapabilityRouter([], [], [], [], {}))
    monkeypatch.setattr(context, "runner", lambda: runner)
    # pin the roster: without this the suite reads whatever roster file the developer has
    monkeypatch.setattr(context, "roster_in_force", lambda: DEFAULT_ROSTER)
    return built


def _fake_query(messages: list[Message]) -> object:
    async def _query(**kwargs: object) -> object:
        for message in messages:
            yield message

    return _query


def _fake_query_then_raise(messages: list[Message]) -> object:
    async def _query(**kwargs: object) -> object:
        for message in messages:
            yield message
        raise RuntimeError("boom mid-stream")

    return _query


def test_schema_teammate_param_descriptions_reach_the_tools() -> None:
    tools = asyncio.run(teammate_server.mcp.list_tools())
    by_name = {tool.name: tool for tool in tools}

    # every teammate tool points the caller at roster() for valid names
    for tool in ("check_teammate", "run_teammate", "dismiss_teammate"):
        schema = by_name[tool].parameters
        assert "roster" in schema["properties"]["name"]["description"].lower(), tool

    run_schema = by_name["run_teammate"].parameters
    assert "standing session" in run_schema["properties"]["resume"]["description"].lower()
    # the one invocation carries the variants as flags rather than spawning sibling tools
    assert {"resume", "wait", "resume_agent_id"} <= set(run_schema["properties"])
    assert "spawn_teammate" not in by_name
    assert "message_teammate" not in by_name


def test_roster_lists_templates_unspawned(pool: AgentPool) -> None:
    entries = teammate_server.roster()
    assert [e.template.name for e in entries] == [t.name for t in DEFAULT_ROSTER.teammates]
    assert all(e.status is TeammateRunStatus.unspawned for e in entries)


def test_check_unspawned(pool: AgentPool) -> None:
    status = teammate_server.check_teammate(_NAME)
    assert status.status is TeammateRunStatus.unspawned
    assert status.agent_key == teammate_key(_NAME)


def test_check_unknown_name_raises_with_roster(pool: AgentPool) -> None:
    with pytest.raises(ValueError, match=_NAME):
        teammate_server.check_teammate("nobody-here")


def test_spawn_runs_in_background_and_persists(
    pool: AgentPool, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        release = asyncio.Event()

        async def gated(**kwargs: object) -> object:
            await release.wait()
            entry = pool.get_by_key(teammate_key(_NAME))
            assert entry is not None
            yield _assistant("background done", session_id=entry.session_id)

        monkeypatch.setattr("agent_fleet.engine.dispatch.query", gated)

        status = await teammate_server.run_teammate(_NAME, _TASK)
        assert status.status is TeammateRunStatus.running
        assert status.run_id is not None

        # spawning again while running returns the live run, not a second one
        again = await teammate_server.run_teammate(_NAME, _TASK)
        assert again.run_id == status.run_id

        release.set()
        await context.runner().wait_all()

        finished = teammate_server.check_teammate(_NAME)
        assert finished.status is TeammateRunStatus.finished
        assert finished.output == "background done"

    asyncio.run(scenario())


def test_stale_after_registry_loss(pool: AgentPool) -> None:
    # create the entry through the template path, then open a run no registry knows about
    context.ensure_teammate(_NAME)
    pool.start_run(teammate_key(_NAME), _TASK)
    status = teammate_server.check_teammate(_NAME)
    assert status.status is TeammateRunStatus.stale


def test_message_wait_returns_finished_status(
    pool: AgentPool, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        context.ensure_teammate(_NAME)
        entry = pool.get_by_key(teammate_key(_NAME))
        assert entry is not None
        monkeypatch.setattr(
            "agent_fleet.engine.dispatch.query",
            _fake_query([_assistant("sync reply", session_id=entry.session_id)]),
        )
        status = await teammate_server.run_teammate(_NAME, _TASK, wait=True)
        assert status.status is TeammateRunStatus.finished
        assert status.output == "sync reply"
        # re-messaging revived the SAME standing session, not a fresh one
        revived = pool.get_by_key(teammate_key(_NAME))
        assert revived is not None and revived.session_id == entry.session_id

    asyncio.run(scenario())


def test_notify_command_folds_a_stop_hook_into_settings(
    pool: AgentPool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENT_FLEET_NOTIFY_COMMAND", "notify-send teammate-done")

    async def scenario() -> None:
        context.ensure_teammate(_NAME)
        entry = pool.get_by_key(teammate_key(_NAME))
        assert entry is not None
        captured: list[object] = []

        async def capturing(**kwargs: object):
            captured.append(kwargs["options"])
            yield _assistant("ok then", session_id=entry.session_id)

        monkeypatch.setattr("agent_fleet.engine.dispatch.query", capturing)
        await teammate_server.run_teammate(_NAME, _TASK, wait=True)

        options = captured[0]
        assert options.settings is not None
        text = Path(options.settings).read_text(encoding="utf-8")
        assert '"Stop"' in text
        assert "notify-send teammate-done" in text

    asyncio.run(scenario())


def test_spawn_failure_marks_run_failed(pool: AgentPool, monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        entry = context.ensure_teammate(_NAME)
        monkeypatch.setattr(
            "agent_fleet.engine.dispatch.query",
            _fake_query_then_raise([_assistant("partial", session_id=entry.session_id)]),
        )
        status = await teammate_server.run_teammate(_NAME, _TASK)
        assert status.status is TeammateRunStatus.running

        await context.runner().wait_all()

        failed = teammate_server.check_teammate(_NAME)
        assert failed.status is TeammateRunStatus.failed
        assert failed.error is not None
        assert "boom mid-stream" in failed.error

    asyncio.run(scenario())


def test_spawn_failure_with_overlong_text_is_bounded_on_write(
    pool: AgentPool, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        entry = context.ensure_teammate(_NAME)
        overlong_message = "x" * (RUN_ERROR_MAX + 500)

        async def raising(**kwargs: object) -> object:
            yield _assistant("partial", session_id=entry.session_id)
            raise RuntimeError(overlong_message)

        monkeypatch.setattr("agent_fleet.engine.dispatch.query", raising)

        status = await teammate_server.run_teammate(_NAME, _TASK)
        assert status.status is TeammateRunStatus.running

        await context.runner().wait_all()

        failed = teammate_server.check_teammate(_NAME)
        assert failed.status is TeammateRunStatus.failed
        assert failed.error is not None
        assert len(failed.error) <= RUN_ERROR_MAX
        run_id = failed.run_id
        assert run_id is not None

        # the row itself must already be bounded on write -- proves the write path validated
        # rather than relying solely on RunError's read-path truncation to mask an over-long value
        conn = sqlite3.connect(pool.db_path)
        stored = conn.execute("SELECT error FROM runs WHERE run_id = ?", (run_id,)).fetchone()[0]
        conn.close()
        assert len(stored) <= RUN_ERROR_MAX

        assert pool.get_run(run_id) is not None
        assert pool.list_runs(teammate_key(_NAME))[0].run_id == run_id

    asyncio.run(scenario())


def test_message_wait_raises_and_marks_run_failed(
    pool: AgentPool, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        context.ensure_teammate(_NAME)
        entry = pool.get_by_key(teammate_key(_NAME))
        assert entry is not None
        monkeypatch.setattr(
            "agent_fleet.engine.dispatch.query",
            _fake_query_then_raise([_assistant("partial", session_id=entry.session_id)]),
        )

        with pytest.raises(RuntimeError, match="boom mid-stream"):
            await teammate_server.run_teammate(_NAME, _TASK, wait=True)

        failed = teammate_server.check_teammate(_NAME)
        assert failed.status is TeammateRunStatus.failed
        assert failed.error is not None
        assert "boom mid-stream" in failed.error

    asyncio.run(scenario())


def test_delete_mid_run_preserves_original_failure_and_logs_the_deletion(
    pool: AgentPool, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    async def scenario() -> None:
        context.ensure_teammate(_NAME)
        release = asyncio.Event()

        async def gated(**kwargs: object) -> object:
            await release.wait()
            raise RuntimeError("boom after deletion")
            yield  # pragma: no cover - unreachable; keeps this an async generator

        monkeypatch.setattr("agent_fleet.engine.dispatch.query", gated)

        status = await teammate_server.run_teammate(_NAME, _TASK)
        assert status.run_id is not None
        task_handle = context.runner().task_for(status.run_id)
        assert task_handle is not None

        assert pool.delete(teammate_key(_NAME))  # dismiss the teammate while its run is live

        with caplog.at_level(logging.ERROR):
            release.set()
            await asyncio.gather(task_handle, return_exceptions=True)

        # the run's real failure survives as the task's terminal exception, not masked by the
        # finish_run KeyError raised when the row it tries to stamp no longer exists
        exc = task_handle.exception()
        assert isinstance(exc, RuntimeError)
        assert "boom after deletion" in str(exc)
        assert "row deleted while live" in caplog.text

        final = teammate_server.check_teammate(_NAME)
        assert final.status is TeammateRunStatus.unspawned

    asyncio.run(scenario())


def test_message_wait_registers_before_awaiting_so_concurrent_check_sees_running(
    pool: AgentPool, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        context.ensure_teammate(_NAME)
        entry = pool.get_by_key(teammate_key(_NAME))
        assert entry is not None
        entered = asyncio.Event()
        release = asyncio.Event()

        async def gated(**kwargs: object) -> object:
            entered.set()
            await release.wait()
            yield _assistant("done waiting", session_id=entry.session_id)

        monkeypatch.setattr("agent_fleet.engine.dispatch.query", gated)

        wait_task = asyncio.create_task(teammate_server.run_teammate(_NAME, _TASK, wait=True))
        # the runner registers the task synchronously before message_teammate's first await, so
        # waiting for the stream to actually start guarantees registration already happened
        await entered.wait()

        mid_flight = teammate_server.check_teammate(_NAME)
        assert mid_flight.status is TeammateRunStatus.running

        release.set()
        status = await wait_task
        assert status.status is TeammateRunStatus.finished

    asyncio.run(scenario())


def test_message_while_running_returns_live_status_without_opening_second_run(
    pool: AgentPool, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        release = asyncio.Event()

        async def gated(**kwargs: object) -> object:
            await release.wait()
            entry = pool.get_by_key(teammate_key(_NAME))
            assert entry is not None
            yield _assistant("first run done", session_id=entry.session_id)

        monkeypatch.setattr("agent_fleet.engine.dispatch.query", gated)

        first = await teammate_server.run_teammate(_NAME, _TASK)
        assert first.status is TeammateRunStatus.running

        second = await teammate_server.run_teammate(_NAME, "a second turn sent while busy")
        assert second.run_id == first.run_id  # no new run was opened

        release.set()
        await context.runner().wait_all()
        assert len(pool.list_runs(teammate_key(_NAME))) == 1

    asyncio.run(scenario())


def test_message_wait_on_a_live_run_waits_for_it_and_reports_terminal(
    pool: AgentPool, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        release = asyncio.Event()

        async def gated(**kwargs: object) -> object:
            await release.wait()
            entry = pool.get_by_key(teammate_key(_NAME))
            assert entry is not None
            yield _assistant("live run done", session_id=entry.session_id)

        monkeypatch.setattr("agent_fleet.engine.dispatch.query", gated)

        first = await teammate_server.run_teammate(_NAME, _TASK)
        assert first.status is TeammateRunStatus.running

        wait_task = asyncio.create_task(
            teammate_server.run_teammate(_NAME, "a second turn sent while busy", wait=True)
        )
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        assert not wait_task.done()  # wait=True must actually block on the already-live run

        release.set()
        status = await wait_task

        assert status.status is TeammateRunStatus.finished
        assert status.run_id == first.run_id
        assert status.output == "live run done"
        assert len(pool.list_runs(teammate_key(_NAME))) == 1

    asyncio.run(scenario())


def test_spawn_fresh_session_ignored_while_run_is_live(
    pool: AgentPool, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        release = asyncio.Event()

        async def gated(**kwargs: object) -> object:
            await release.wait()
            entry = pool.get_by_key(teammate_key(_NAME))
            assert entry is not None
            yield _assistant("first run done", session_id=entry.session_id)

        monkeypatch.setattr("agent_fleet.engine.dispatch.query", gated)

        first = await teammate_server.run_teammate(_NAME, _TASK)
        assert first.status is TeammateRunStatus.running
        original_session = first.session_id

        again = await teammate_server.run_teammate(_NAME, _TASK, resume=False)
        assert again.run_id == first.run_id  # same live run, not reset/requeued
        assert again.session_id == original_session  # the live session was NOT reset

        release.set()
        await context.runner().wait_all()

    asyncio.run(scenario())


def test_spawn_fresh_session_changes_session_id_when_idle(
    pool: AgentPool, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        original = context.ensure_teammate(_NAME)
        monkeypatch.setattr("agent_fleet.engine.dispatch.query", _fake_query([]))

        await teammate_server.run_teammate(_NAME, _TASK, resume=False)
        await context.runner().wait_all()

        revived = pool.get_by_key(teammate_key(_NAME))
        assert revived is not None
        assert revived.session_id != original.session_id

    asyncio.run(scenario())


def test_message_default_backgrounds_then_finishes(
    pool: AgentPool, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        context.ensure_teammate(_NAME)
        entry = pool.get_by_key(teammate_key(_NAME))
        assert entry is not None
        release = asyncio.Event()

        async def gated(**kwargs: object) -> object:
            await release.wait()
            yield _assistant("background reply", session_id=entry.session_id)

        monkeypatch.setattr("agent_fleet.engine.dispatch.query", gated)

        status = await teammate_server.run_teammate(_NAME, _TASK)
        assert status.status is TeammateRunStatus.running

        release.set()
        await context.runner().wait_all()

        finished = teammate_server.check_teammate(_NAME)
        assert finished.status is TeammateRunStatus.finished
        assert finished.output == "background reply"

    asyncio.run(scenario())


def test_message_resume_agent_id_grants_send_message_and_wraps_prompt(
    pool: AgentPool, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        entry = context.ensure_teammate(_NAME)
        agent_id = "abc123def456"
        captured: list[dict[str, object]] = []

        async def capturing(**kwargs: object) -> object:
            captured.append(dict(kwargs))
            yield _assistant("resumed", session_id=entry.session_id)

        monkeypatch.setattr("agent_fleet.engine.dispatch.query", capturing)

        status = await teammate_server.run_teammate(
            _NAME, _TASK, wait=True, resume_agent_id=agent_id
        )
        assert status.status is TeammateRunStatus.finished

        assert len(captured) == 1
        options = captured[0]["options"]
        assert isinstance(options, ClaudeAgentOptions)
        assert SEND_MESSAGE_TOOL in options.allowed_tools
        prompt = captured[0]["prompt"]
        assert isinstance(prompt, str)
        assert f"Resume agent {agent_id}" in prompt
        assert _TASK in prompt

    asyncio.run(scenario())


def test_run_teammate_codex_provider_dispatches_run_codex_capture(
    pool: AgentPool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        captured: dict[str, object] = {}

        async def fake_codex_capture(pool_arg, agent_key_arg, task_arg, request_arg, *, run=None):
            captured["agent_key"] = agent_key_arg
            captured["request"] = request_arg
            started = run if run is not None else pool_arg.start_run(agent_key_arg, task_arg)
            return RunOutcome(
                output="codex reply",
                run=pool_arg.finish_run(started.run_id, output="codex reply"),
                agent_runs=[],
            )

        monkeypatch.setattr("agent_fleet.engine.dispatch.run_codex_capture", fake_codex_capture)
        codex_config = CodexRunConfig(
            cwd=tmp_path,
            scope=RunScope(mode=RunMode.read),
            model=CodexModelId.gpt_5_6_sol,
            timeout_s=60,
        )
        status = await teammate_server.run_teammate(
            _NAME, _TASK, wait=True, provider=Provider.codex, codex=codex_config
        )

        assert status.status is TeammateRunStatus.finished
        assert status.output == "codex reply"
        assert captured["agent_key"] == teammate_key(_NAME)
        assert isinstance(captured["request"], CodexRunRequest)
        assert captured["request"].cwd == tmp_path

    asyncio.run(scenario())


def test_run_teammate_codex_provider_without_config_raises(pool: AgentPool) -> None:
    async def scenario() -> None:
        with pytest.raises(ValueError, match="codex run settings are required"):
            await teammate_server.run_teammate(_NAME, _TASK, provider=Provider.codex)

    asyncio.run(scenario())


def test_roster_after_spawn_shows_populated_row(
    pool: AgentPool, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        monkeypatch.setattr("agent_fleet.engine.dispatch.query", _fake_query([]))

        await teammate_server.run_teammate(_NAME, _TASK)

        entries = teammate_server.roster()
        row = next(e for e in entries if e.template.name == _NAME)
        assert row.session_id is not None
        assert row.status is not TeammateRunStatus.unspawned

        await context.runner().wait_all()

    asyncio.run(scenario())
