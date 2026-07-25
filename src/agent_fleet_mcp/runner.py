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

    def spawn(self, run_id: RunId, coro: Coroutine[object, object, object]) -> asyncio.Task[object]:
        """Start `coro` as a background task registered under `run_id`, returning the task.

        A caller that must await this exact run (e.g. `message_teammate(wait=True)`) awaits the
        returned task, never a bare `await coro`: awaiting the coroutine directly would run it
        outside the registry's bookkeeping, so a concurrent `check_teammate` would see nothing
        live — `stale`, not `running` — for the whole duration of the wait.
        """
        task = asyncio.get_running_loop().create_task(coro)
        self._tasks[run_id] = task
        task.add_done_callback(lambda finished: self._deregister(run_id, finished))
        return task

    def _deregister(self, run_id: RunId, task: asyncio.Task[object]) -> None:
        """Drop the finished task, unless `run_id` was already reassigned to a newer task (a
        duplicate `spawn` under the same id): only the entry that still points at this exact task
        is removed, so a stale finish can't evict a live run. A failure is logged, not raised (the
        run row stays unfinished, so `run_status` reports it `stale`)."""
        if self._tasks.get(run_id) is task:
            del self._tasks[run_id]
        if not task.cancelled() and task.exception() is not None:
            logger.error("teammate run %s failed", run_id, exc_info=task.exception())

    def is_running(self, run_id: RunId) -> bool:
        """Whether `run_id` is a currently-registered live task."""
        return run_id in self._tasks

    def task_for(self, run_id: RunId) -> asyncio.Task[object] | None:
        """The currently-registered live task for `run_id`, or None when it isn't registered —
        already finished (deregistered), or never spawned in this process. Lets a caller that
        discovers a run is already live (rather than having just spawned it itself) still await
        its completion, e.g. `message_teammate(wait=True)` against a run `spawn_teammate` opened.
        """
        return self._tasks.get(run_id)

    async def wait_all(self) -> None:
        """Wait for every registered task to complete, including tasks a running task spawns via
        the runner while this call is waiting — shutdown and tests. A single `gather` over a
        one-time snapshot would miss those: each pass gathers the current snapshot (failures
        swallowed) and loops until no task remains registered."""
        while self._tasks:
            await asyncio.gather(*list(self._tasks.values()), return_exceptions=True)


def run_status(run: RunRecord | None, runner: TeammateRunner) -> TeammateRunStatus:
    """Derive the status of a teammate's latest run; `None` (no runs yet) is `idle`."""
    if run is None:
        return TeammateRunStatus.idle
    if run.finished_at is not None:
        return TeammateRunStatus.failed if run.error is not None else TeammateRunStatus.finished
    return TeammateRunStatus.running if runner.is_running(run.run_id) else TeammateRunStatus.stale
