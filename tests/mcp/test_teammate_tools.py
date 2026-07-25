from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from claude_agent_sdk import Message

from agent_fleet import AgentPool
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
