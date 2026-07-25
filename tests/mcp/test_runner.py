from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

import pytest

from agent_fleet.models.agent import RunRecord, TeammateRunStatus
from agent_fleet_mcp.runner import TeammateRunner, run_status

_RUN_ID = "7c9e6679-7425-40de-944b-e07fc1f90ae7"
_RUN_ID_CHILD = "8c9e6679-7425-40de-944b-e07fc1f90ae8"


def _open_run() -> RunRecord:
    return RunRecord(
        run_id=_RUN_ID,
        agent_key="teammate.reviewer",
        task="an unfinished run for status derivation tests",
        started_at=datetime.now(UTC),
    )


def test_status_of_no_run_is_idle() -> None:
    assert run_status(None, TeammateRunner()) is TeammateRunStatus.idle


def test_status_of_finished_run() -> None:
    run = _open_run().model_copy(update={"finished_at": datetime.now(UTC)})
    assert run_status(run, TeammateRunner()) is TeammateRunStatus.finished


def test_status_running_while_registered_then_stale_after() -> None:
    async def scenario() -> None:
        runner = TeammateRunner()
        release = asyncio.Event()

        async def job() -> str:
            await release.wait()
            return "done"

        runner.spawn(_RUN_ID, job())
        assert run_status(_open_run(), runner) is TeammateRunStatus.running
        release.set()
        await runner.wait_all()
        # task completed and deregistered; the run row was never stamped -> stale
        assert run_status(_open_run(), runner) is TeammateRunStatus.stale

    asyncio.run(scenario())


def test_failing_task_deregisters_without_raising(caplog: pytest.LogCaptureFixture) -> None:
    async def scenario() -> None:
        runner = TeammateRunner()

        async def job() -> str:
            raise RuntimeError("boom")

        with caplog.at_level(logging.ERROR):
            runner.spawn(_RUN_ID, job())
            await runner.wait_all()
        assert not runner.is_running(_RUN_ID)
        assert _RUN_ID in caplog.text

    asyncio.run(scenario())


def test_wait_all_waits_for_tasks_spawned_by_running_tasks() -> None:
    async def scenario() -> None:
        runner = TeammateRunner()
        child_done = asyncio.Event()

        async def child() -> str:
            await asyncio.sleep(0.05)
            child_done.set()
            return "child"

        async def parent() -> str:
            runner.spawn(_RUN_ID_CHILD, child())
            return "parent"

        runner.spawn(_RUN_ID, parent())
        await runner.wait_all()

        assert child_done.is_set()
        assert not runner.is_running(_RUN_ID)
        assert not runner.is_running(_RUN_ID_CHILD)

    asyncio.run(scenario())


def test_deregister_ignores_a_stale_task_under_a_reused_run_id() -> None:
    async def scenario() -> None:
        runner = TeammateRunner()
        release_first = asyncio.Event()
        finished_first = asyncio.Event()
        release_second = asyncio.Event()

        async def first() -> str:
            await release_first.wait()
            finished_first.set()
            return "first"

        async def second() -> str:
            await release_second.wait()
            return "second"

        runner.spawn(_RUN_ID, first())
        runner.spawn(_RUN_ID, second())  # reuses the id; registry now points at `second`

        release_first.set()
        await finished_first.wait()
        await asyncio.sleep(0)  # let `first`'s done-callback run
        assert runner.is_running(_RUN_ID)  # `second` is still live; `first` must not evict it

        release_second.set()
        await runner.wait_all()
        assert not runner.is_running(_RUN_ID)

    asyncio.run(scenario())
