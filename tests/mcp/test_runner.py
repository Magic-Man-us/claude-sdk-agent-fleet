from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from agent_fleet.models.agent import RunRecord, TeammateRunStatus
from agent_fleet_mcp.runner import TeammateRunner, run_status

_RUN_ID = "7c9e6679-7425-40de-944b-e07fc1f90ae7"


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


def test_failing_task_deregisters_without_raising() -> None:
    async def scenario() -> None:
        runner = TeammateRunner()

        async def job() -> str:
            raise RuntimeError("boom")

        runner.spawn(_RUN_ID, job())
        await runner.wait_all()
        assert not runner.is_running(_RUN_ID)

    asyncio.run(scenario())
