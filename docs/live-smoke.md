# Live smoke test

Everything else in this repo runs `run_teammate` against a fake `query()` message stream — it
proves the wiring, not that a teammate actually talks to Claude. This is the one path that runs
for real.

## What it proves

That `run_teammate` — the exact function the `teammate-mcp` server exposes — drives a real
`claude` CLI invocation through the Claude Agent SDK and gets real text back. It builds a
throwaway roster (one teammate, a trivial brief) and a throwaway pool db under a temp directory,
never the real `~/.claude/agent-fleet/pool.db` or a real roster file.

## What it costs

One turn on a cheap model (`haiku` by default, `sonnet` optionally) replying to a ~15-word prompt.
A fraction of a cent. Never defaults to `opus`.

## Running it

Standalone script, real tokens, prints the outcome, exits non-zero on failure:

```
uv run python scripts/smoke_teammate.py
uv run python scripts/smoke_teammate.py --model sonnet
```

As a pytest test — skipped unless both are true: the `claude` CLI is on `PATH`, and
`AGENT_FLEET_LIVE=1` is set:

```
AGENT_FLEET_LIVE=1 uv run pytest -m live
```

It carries the `live` marker (registered in `pyproject.toml`), so a plain `uv run pytest` never
runs it and never spends tokens.
