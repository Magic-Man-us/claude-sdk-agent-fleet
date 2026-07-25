from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from claude_agent_sdk import ClaudeAgentOptions, Message

from agent_fleet import AgentPool
from agent_fleet.engine.render import SEND_MESSAGE_TOOL
from agent_fleet.engine.source import InMemoryCatalogSource
from agent_fleet.engine.teammates import ROSTER
from agent_fleet.models.agent import TeammateRunStatus, teammate_key
from agent_fleet.router.capability import CapabilityRouter
from agent_fleet_mcp import pool_server
from agent_fleet_mcp.runner import TeammateRunner
from capdisc.catalog import Catalog
from test_dispatch import _assistant

_NAME = ROSTER[0].name
_TASK = "run the roster teammate against a synthetic task"


@pytest.fixture
def pool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AgentPool:
    built = AgentPool(tmp_path / "pool.db")
    runner = TeammateRunner()
    monkeypatch.setattr(pool_server, "_pool", lambda: built)
    monkeypatch.setattr(pool_server, "_source", lambda: InMemoryCatalogSource(Catalog(entries=[])))
    monkeypatch.setattr(
        pool_server, "_capability_router", lambda: CapabilityRouter([], [], [], [], {})
    )
    monkeypatch.setattr(pool_server, "_runner", lambda: runner)
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


def test_roster_lists_templates_unspawned(pool: AgentPool) -> None:
    entries = pool_server.roster()
    assert [e.template.name for e in entries] == [t.name for t in ROSTER]
    assert all(e.status is TeammateRunStatus.unspawned for e in entries)


def test_check_unspawned(pool: AgentPool) -> None:
    status = pool_server.check_teammate(_NAME)
    assert status.status is TeammateRunStatus.unspawned
    assert status.agent_key == teammate_key(_NAME)


def test_check_unknown_name_raises_with_roster(pool: AgentPool) -> None:
    with pytest.raises(ValueError, match=_NAME):
        pool_server.check_teammate("nobody-here")


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

        status = await pool_server.spawn_teammate(_NAME, _TASK)
        assert status.status is TeammateRunStatus.running
        assert status.run_id is not None

        # spawning again while running returns the live run, not a second one
        again = await pool_server.spawn_teammate(_NAME, _TASK)
        assert again.run_id == status.run_id

        release.set()
        await pool_server._runner().wait_all()

        finished = pool_server.check_teammate(_NAME)
        assert finished.status is TeammateRunStatus.finished
        assert finished.output == "background done"

    asyncio.run(scenario())


def test_stale_after_registry_loss(pool: AgentPool) -> None:
    # create the entry through the template path, then open a run no registry knows about
    pool_server._ensure_teammate(_NAME)
    pool.start_run(teammate_key(_NAME), _TASK)
    status = pool_server.check_teammate(_NAME)
    assert status.status is TeammateRunStatus.stale


def test_message_wait_returns_finished_status(
    pool: AgentPool, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        pool_server._ensure_teammate(_NAME)
        entry = pool.get_by_key(teammate_key(_NAME))
        assert entry is not None
        monkeypatch.setattr(
            "agent_fleet.engine.dispatch.query",
            _fake_query([_assistant("sync reply", session_id=entry.session_id)]),
        )
        status = await pool_server.message_teammate(_NAME, _TASK, wait=True)
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
        pool_server._ensure_teammate(_NAME)
        entry = pool.get_by_key(teammate_key(_NAME))
        assert entry is not None
        captured: list[object] = []

        async def capturing(**kwargs: object):
            captured.append(kwargs["options"])
            yield _assistant("ok then", session_id=entry.session_id)

        monkeypatch.setattr("agent_fleet.engine.dispatch.query", capturing)
        await pool_server.message_teammate(_NAME, _TASK, wait=True)

        options = captured[0]
        assert options.settings is not None
        text = Path(options.settings).read_text(encoding="utf-8")
        assert '"Stop"' in text
        assert "notify-send teammate-done" in text

    asyncio.run(scenario())


def test_spawn_failure_marks_run_failed(pool: AgentPool, monkeypatch: pytest.MonkeyPatch) -> None:
    async def scenario() -> None:
        entry = pool_server._ensure_teammate(_NAME)
        monkeypatch.setattr(
            "agent_fleet.engine.dispatch.query",
            _fake_query_then_raise([_assistant("partial", session_id=entry.session_id)]),
        )
        status = await pool_server.spawn_teammate(_NAME, _TASK)
        assert status.status is TeammateRunStatus.running

        await pool_server._runner().wait_all()

        failed = pool_server.check_teammate(_NAME)
        assert failed.status is TeammateRunStatus.failed
        assert failed.error is not None
        assert "boom mid-stream" in failed.error

    asyncio.run(scenario())


def test_message_wait_raises_and_marks_run_failed(
    pool: AgentPool, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        pool_server._ensure_teammate(_NAME)
        entry = pool.get_by_key(teammate_key(_NAME))
        assert entry is not None
        monkeypatch.setattr(
            "agent_fleet.engine.dispatch.query",
            _fake_query_then_raise([_assistant("partial", session_id=entry.session_id)]),
        )

        with pytest.raises(RuntimeError, match="boom mid-stream"):
            await pool_server.message_teammate(_NAME, _TASK, wait=True)

        failed = pool_server.check_teammate(_NAME)
        assert failed.status is TeammateRunStatus.failed
        assert failed.error is not None
        assert "boom mid-stream" in failed.error

    asyncio.run(scenario())


def test_message_wait_registers_before_awaiting_so_concurrent_check_sees_running(
    pool: AgentPool, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        pool_server._ensure_teammate(_NAME)
        entry = pool.get_by_key(teammate_key(_NAME))
        assert entry is not None
        release = asyncio.Event()

        async def gated(**kwargs: object) -> object:
            await release.wait()
            yield _assistant("done waiting", session_id=entry.session_id)

        monkeypatch.setattr("agent_fleet.engine.dispatch.query", gated)

        wait_task = asyncio.create_task(pool_server.message_teammate(_NAME, _TASK, wait=True))
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        mid_flight = pool_server.check_teammate(_NAME)
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

        first = await pool_server.spawn_teammate(_NAME, _TASK)
        assert first.status is TeammateRunStatus.running

        second = await pool_server.message_teammate(_NAME, "a second turn sent while busy")
        assert second.run_id == first.run_id  # no new run was opened

        release.set()
        await pool_server._runner().wait_all()
        assert len(pool_server.list_runs(teammate_key(_NAME))) == 1

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

        first = await pool_server.spawn_teammate(_NAME, _TASK)
        assert first.status is TeammateRunStatus.running
        original_session = first.session_id

        again = await pool_server.spawn_teammate(_NAME, _TASK, fresh_session=True)
        assert again.run_id == first.run_id  # same live run, not reset/requeued
        assert again.session_id == original_session  # the live session was NOT reset

        release.set()
        await pool_server._runner().wait_all()

    asyncio.run(scenario())


def test_spawn_fresh_session_changes_session_id_when_idle(
    pool: AgentPool, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        original = pool_server._ensure_teammate(_NAME)
        monkeypatch.setattr("agent_fleet.engine.dispatch.query", _fake_query([]))

        await pool_server.spawn_teammate(_NAME, _TASK, fresh_session=True)
        await pool_server._runner().wait_all()

        revived = pool.get_by_key(teammate_key(_NAME))
        assert revived is not None
        assert revived.session_id != original.session_id

    asyncio.run(scenario())


def test_message_default_backgrounds_then_finishes(
    pool: AgentPool, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        pool_server._ensure_teammate(_NAME)
        entry = pool.get_by_key(teammate_key(_NAME))
        assert entry is not None
        release = asyncio.Event()

        async def gated(**kwargs: object) -> object:
            await release.wait()
            yield _assistant("background reply", session_id=entry.session_id)

        monkeypatch.setattr("agent_fleet.engine.dispatch.query", gated)

        status = await pool_server.message_teammate(_NAME, _TASK)
        assert status.status is TeammateRunStatus.running

        release.set()
        await pool_server._runner().wait_all()

        finished = pool_server.check_teammate(_NAME)
        assert finished.status is TeammateRunStatus.finished
        assert finished.output == "background reply"

    asyncio.run(scenario())


def test_message_resume_agent_id_grants_send_message_and_wraps_prompt(
    pool: AgentPool, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        entry = pool_server._ensure_teammate(_NAME)
        agent_id = "abc123def456"
        captured: list[dict[str, object]] = []

        async def capturing(**kwargs: object) -> object:
            captured.append(dict(kwargs))
            yield _assistant("resumed", session_id=entry.session_id)

        monkeypatch.setattr("agent_fleet.engine.dispatch.query", capturing)

        status = await pool_server.message_teammate(
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


def test_roster_after_spawn_shows_populated_row(
    pool: AgentPool, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        monkeypatch.setattr("agent_fleet.engine.dispatch.query", _fake_query([]))

        await pool_server.spawn_teammate(_NAME, _TASK)

        entries = pool_server.roster()
        row = next(e for e in entries if e.template.name == _NAME)
        assert row.session_id is not None
        assert row.status is not TeammateRunStatus.unspawned

        await pool_server._runner().wait_all()

    asyncio.run(scenario())
