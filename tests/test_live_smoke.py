"""Live smoke test for `run_teammate`, gated behind `AGENT_FLEET_LIVE=1`.

Spends real tokens against the `claude` CLI, so it is skipped by default — see
`docs/live-smoke.md`. `scripts/smoke_teammate.py` runs the same path standalone.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from agent_fleet.models.agent import Provider, TeammateRunStatus
from agent_fleet_mcp import context

_ENABLE_VAR = "AGENT_FLEET_LIVE"
_TASK = "Reply with the exact literal text OK-SMOKE-TEST and nothing else."
_ROSTER_TOML = """\
[[teammates]]
name = "smoke-tester"
brief = "Reply with the exact literal text OK-SMOKE-TEST and nothing else."
model = "haiku"
"""

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get(_ENABLE_VAR) != "1",
        reason=f"set {_ENABLE_VAR}=1 to run against the real claude CLI; spends real tokens",
    ),
    pytest.mark.skipif(shutil.which("claude") is None, reason="the claude CLI is not on PATH"),
]

_CACHED_ACCESSORS = (
    context.pool,
    context.roster_in_force,
    context.source,
    context.capability_router,
)


@pytest.fixture
def _throwaway_roster_and_pool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Point the process-wide, `@cache`'d context accessors at a throwaway roster and pool db —
    never the real ones — for the duration of one test."""
    roster_path = tmp_path / "roster.toml"
    roster_path.write_text(_ROSTER_TOML, encoding="utf-8")
    monkeypatch.setenv("AGENT_FLEET_ROSTER", str(roster_path))
    monkeypatch.setenv("AGENT_FLEET_POOL_DB", str(tmp_path / "pool.db"))
    monkeypatch.chdir(tmp_path)
    for accessor in _CACHED_ACCESSORS:
        accessor.cache_clear()
    yield
    for accessor in _CACHED_ACCESSORS:
        accessor.cache_clear()


def test_run_teammate_against_the_real_claude_cli(_throwaway_roster_and_pool: None) -> None:
    from agent_fleet_mcp import teammate_server

    status = asyncio.run(
        teammate_server.run_teammate(
            name="smoke-tester",
            task=_TASK,
            wait=True,
            provider=Provider.claude,
        )
    )

    assert status.status is TeammateRunStatus.finished, status.error
    assert status.output
