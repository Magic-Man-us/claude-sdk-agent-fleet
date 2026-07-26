"""Live smoke test: run one throwaway teammate through the real `run_teammate` path.

Spends real tokens against the Claude Agent SDK (via the `claude` CLI). Never touches the real
pool db or roster — a throwaway roster TOML and pool db are created under a temp directory and
pointed to via `AGENT_FLEET_ROSTER`/`AGENT_FLEET_POOL_DB`, and the process cwd is switched into
that temp directory before the teammate is built. Defaults to `haiku`; never defaults to `opus`.

Usage:
    uv run python scripts/smoke_teammate.py [--model haiku|sonnet]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
from pathlib import Path

_TASK = "Reply with the exact literal text OK-SMOKE-TEST and nothing else."

_ROSTER_TOML = """\
[[teammates]]
name = "smoke-tester"
brief = "Reply with the exact literal text OK-SMOKE-TEST and nothing else."
model = "{model}"
"""

_CHEAP_MODELS = ("haiku", "sonnet")


def _run(model: str) -> int:
    with tempfile.TemporaryDirectory(prefix="agent-fleet-smoke-") as tmp:
        tmp_path = Path(tmp)
        roster_path = tmp_path / "roster.toml"
        roster_path.write_text(_ROSTER_TOML.format(model=model), encoding="utf-8")
        # Set before the first call into agent_fleet_mcp.context: pool()/roster_in_force() are
        # @cache'd off AgentFleetSettings(), read once on first use in this process.
        os.environ["AGENT_FLEET_ROSTER"] = str(roster_path)
        os.environ["AGENT_FLEET_POOL_DB"] = str(tmp_path / "pool.db")
        os.chdir(tmp_path)

        from agent_fleet.models.agent import Provider, TeammateRunStatus
        from agent_fleet_mcp import teammate_server

        status = asyncio.run(
            teammate_server.run_teammate(
                name="smoke-tester",
                task=_TASK,
                wait=True,
                provider=Provider.claude,
            )
        )

    print(f"status: {status.status}")
    print(f"run_id: {status.run_id}")
    print(f"session_id: {status.session_id}")
    print(f"cost_usd: {status.total_cost_usd}")
    print(f"output: {status.output!r}")
    if status.error is not None:
        print(f"error: {status.error}", file=sys.stderr)

    if status.status is not TeammateRunStatus.finished or not status.output:
        print("FAIL: teammate did not finish with real output", file=sys.stderr)
        return 1
    print("OK: real output came back through run_teammate")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="haiku",
        choices=_CHEAP_MODELS,
        help="Cheap model to run the throwaway teammate on (default: haiku).",
    )
    args = parser.parse_args()
    return _run(args.model)


if __name__ == "__main__":
    raise SystemExit(main())
