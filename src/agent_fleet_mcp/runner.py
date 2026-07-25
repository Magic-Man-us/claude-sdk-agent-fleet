from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine

from agent_fleet.models.agent import RunId, RunRecord, TeammateRunStatus

logger = logging.getLogger(__name__)


class TeammateRunner:
    """In-process registry of live background runs, keyed by run id.

    Holds each spawned `asyncio.Task` until it completes; membership is what distinguishes a
    `running` run from a `stale` one (unfinished in the database, unknown here — the server died
    mid-run). Results are not read from the task: the run coroutine persists its own outcome via
    `finish_run`, so a completed task's value is already in the pool."""

    def __init__(self) -> None:
        self._tasks: dict[RunId, asyncio.Task[object]] = {}

    def spawn(self, run_id: RunId, coro: Coroutine[object, object, object]) -> None:
        """Start `coro` as a background task registered under `run_id`."""
        task = asyncio.get_running_loop().create_task(coro)
        self._tasks[run_id] = task
        task.add_done_callback(lambda finished: self._deregister(run_id, finished))

    def _deregister(self, run_id: RunId, task: asyncio.Task[object]) -> None:
        """Drop the finished task; a failure is logged, not raised (the run row stays unfinished,
        so `run_status` reports it `stale`)."""
        self._tasks.pop(run_id, None)
        if not task.cancelled() and task.exception() is not None:
            logger.error("teammate run %s failed", run_id, exc_info=task.exception())

    def is_running(self, run_id: RunId) -> bool:
        """Whether `run_id` is a currently-registered live task."""
        return run_id in self._tasks

    async def wait_all(self) -> None:
        """Wait for every registered task to complete (failures swallowed) — shutdown and tests."""
        await asyncio.gather(*list(self._tasks.values()), return_exceptions=True)


def run_status(run: RunRecord | None, runner: TeammateRunner) -> TeammateRunStatus:
    """Derive the status of a teammate's latest run; `None` (no runs yet) is `idle`."""
    if run is None:
        return TeammateRunStatus.idle
    if run.finished_at is not None:
        return TeammateRunStatus.finished
    return TeammateRunStatus.running if runner.is_running(run.run_id) else TeammateRunStatus.stale
