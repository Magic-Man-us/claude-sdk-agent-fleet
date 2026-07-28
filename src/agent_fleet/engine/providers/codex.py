"""The Codex provider adapter: one blocking `codex exec` turn, run off the event loop.

Ported from an internal Codex-invocation reference implementation — its scope, isolation, and
change-detection primitives already live in `engine.scope` and `engine.worktree`, so this only
adds the Codex-specific transport: argv construction, JSONL parsing, and the subprocess call
itself. The packet/criterion/worker-result-schema policy layered on those primitives there is
that caller's own contract, not a generic "run a turn" capability, and stays with it.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import signal
import subprocess
from pathlib import Path

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from ...models.agent import (
    AgentKey,
    CodexRunRequest,
    ProviderSessionId,
    RunMode,
    RunOutcome,
    RunOutput,
    RunRecord,
    TaskBrief,
)
from ..codex_tool import enforce_codex_policy
from ..pool import AgentPool
from ..scope import violations
from ..worktree import (
    actual_changes,
    forbidden_state,
    require_isolated_worktree,
    scrub_secret_env,
)

#: Seconds granted to a terminated-but-not-yet-dead process before it is force-killed.
_TERMINATE_GRACE_SECONDS = 5
_SANDBOX_BY_MODE = {RunMode.read: "read-only", RunMode.write: "workspace-write"}

_DEVELOPER_INSTRUCTIONS_ADAPTER: TypeAdapter[str] = TypeAdapter(str)
_RUN_OUTPUT_ADAPTER: TypeAdapter[RunOutput] = TypeAdapter(RunOutput)
_SESSION_ADAPTER: TypeAdapter[ProviderSessionId] = TypeAdapter(ProviderSessionId)


class CodexRunError(RuntimeError):
    """A Codex `exec` turn failed transport-, scope-, or timeout-wise.

    Raised rather than returned as a `{"status": "blocked"}` dict, matching this codebase's
    exception-based failure convention (`context.record_failure`): a non-zero exit, a scope
    violation, a malformed thread count, or a timeout all surface here so the caller's own
    failure-recording wrapper stamps the run `failed` and re-raises.
    """


class _CodexItem(BaseModel):
    """The `item` payload of an `item.completed` event — only the fields this adapter reads."""

    model_config = ConfigDict(extra="ignore")

    type: str | None = None
    text: str | None = None


class _CodexEvent(BaseModel):
    """One parsed line of Codex's `exec --json` output — only the fields this adapter reads.

    Codex's JSONL protocol is external and evolving; `extra="ignore"` plus all-optional fields let
    an event type this adapter does not recognize pass through without raising — unrecognized
    events are silently skipped during parsing, not treated as transport errors. A hard
    Literal-tagged discriminated union would instead reject any future event type outright, which
    is the wrong failure mode for a forward-compatible external stream.
    """

    model_config = ConfigDict(extra="ignore")

    type: str | None = None
    thread_id: str | None = None
    item: _CodexItem | None = None


class _ParsedStream(BaseModel):
    """What the JSONL stream told us: the thread id, the final agent message, terminal state."""

    model_config = ConfigDict(frozen=True)

    thread_id: str | None = None
    thread_count: int = 0
    final_message: str = ""
    parse_errors: list[str] = []
    turn_completed: bool = False
    turn_failed: bool = False


def _parse_jsonl(stdout: str) -> _ParsedStream:
    """Extract the thread id, final agent message, and terminal state from Codex's JSONL stream."""
    thread_id: str | None = None
    thread_count = 0
    final_message = ""
    errors: list[str] = []
    turn_completed = False
    turn_failed = False
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            event = _CodexEvent.model_validate_json(line)
        except ValidationError:
            errors.append(f"non-JSON output: {line[:200]}")
            continue
        if event.type == "thread.started" and event.thread_id is not None:
            thread_count += 1
            thread_id = event.thread_id
        elif (
            event.type == "item.completed"
            and event.item is not None
            and event.item.type == "agent_message"
            and event.item.text is not None
        ):
            final_message = event.item.text
        elif event.type == "turn.completed":
            turn_completed = True
        elif event.type == "turn.failed":
            turn_failed = True
            errors.append("codex reported turn.failed")
    return _ParsedStream(
        thread_id=thread_id,
        thread_count=thread_count,
        final_message=final_message,
        parse_errors=errors,
        turn_completed=turn_completed,
        turn_failed=turn_failed,
    )


def _argv(request: CodexRunRequest, executable: str) -> list[str]:
    """The `codex` command line for one `exec` turn — a "-" prompt-on-stdin invocation."""
    developer_instructions = _DEVELOPER_INSTRUCTIONS_ADAPTER.dump_json(
        request.developer_instructions
    ).decode()
    argv = [
        executable,
        "--model",
        request.model.value,
        "--sandbox",
        _SANDBOX_BY_MODE[request.scope.mode],
        "--ask-for-approval",
        "never",
        "--config",
        f"developer_instructions={developer_instructions}",
        "exec",
        "--json",
        "--ephemeral",
    ]
    if request.output_schema_path is not None:
        argv += ["--output-schema", str(request.output_schema_path)]
    argv.append("-")
    return argv


def _diagnostic_changes(cwd: Path, forbidden_paths: list[str], before: dict[str, str]) -> list[str]:
    """Best-effort `actual_changes` for an error message; swallowed on failure so a diagnostic
    can never mask the real error it is attached to."""
    try:
        return actual_changes(cwd, forbidden_paths, before)
    except (OSError, RuntimeError):
        return []


def _run_codex_sync(request: CodexRunRequest, task: TaskBrief) -> str:
    """Run one Codex `exec` turn synchronously: blocking subprocess I/O plus scope enforcement.

    Dispatched off the event loop by `run_codex_capture` via `asyncio.to_thread` — this function's
    `subprocess.Popen(...).communicate(...)` call genuinely blocks for up to `request.timeout_s`
    seconds, and calling it directly on the loop would freeze every other run the MCP server is
    serving.

    Raises:
        CodexRunError: The `codex` executable is missing, the turn timed out, exited non-zero, or
            the worktree's actual changes violate `request.scope`.
    """
    executable = shutil.which("codex")
    if executable is None:
        raise CodexRunError("codex CLI is not installed or not on PATH")
    environment = scrub_secret_env(os.environ)
    forbidden_before = forbidden_state(request.cwd, request.scope.forbidden_paths)
    process = subprocess.Popen(  # noqa: S603
        _argv(request, executable),
        cwd=request.cwd,
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, _stderr = process.communicate(task.encode(), timeout=request.timeout_s)
    except subprocess.TimeoutExpired as exc:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.communicate(timeout=_TERMINATE_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
        changed = _diagnostic_changes(request.cwd, request.scope.forbidden_paths, forbidden_before)
        raise CodexRunError(
            f"codex timed out after {request.timeout_s} seconds "
            f"(worktree changes so far: {changed})"
        ) from exc
    actual_paths = actual_changes(request.cwd, request.scope.forbidden_paths, forbidden_before)
    problems = violations(request.scope, actual_paths)
    if problems:
        raise CodexRunError("; ".join(problems))
    if process.returncode != 0:
        raise CodexRunError(f"codex exited {process.returncode}")
    return stdout.decode(errors="replace")


async def run_codex_capture(
    pool: AgentPool,
    agent_key: AgentKey,
    task: TaskBrief,
    request: CodexRunRequest,
    *,
    run: RunRecord | None = None,
) -> RunOutcome:
    """Run one Codex `exec` turn live and record it as the run's sole agent.

    One-shot by construction: Codex's `exec` mode dispatches no subagents and takes no thread-id
    input, so exactly one `agent_runs` row is written — under the pool entry's own bookkeeping
    UUID (`PoolEntry.session_id`), the same identity-not-conversation role that field already plays
    for the Claude path. Codex's own `thread_id` (parsed here, `ProviderSessionId`-validated) is
    NOT forced into that UUID-shaped field; persisting it per-run on `RunRecord` is out of this
    module's scope (`RunRecord`/`AgentPool.finish_run` are not among the files this change touches)
    and is left for whoever owns that model.

    Args:
        pool: The pool the run and its captured agent are recorded in.
        agent_key: The pooled agent this run belongs to; its entry supplies the recorded session id.
        task: The run's recorded task and the literal prompt sent on stdin.
        request: The Codex run's cwd, scope, model, timeout, and developer instructions.
        run: An already-started run record, or None to start one here.

    Raises:
        KeyError: `agent_key` has no pool entry.
        ValueError: `request.cwd` is not a clean, detached-HEAD, linked worktree.
        CodexRunError: The codex CLI is missing, the run timed out, exited non-zero, the worktree's
            actual changes violate `request.scope`, or the JSONL stream did not end in exactly one
            completed turn.
    """
    entry = pool.get_by_key(agent_key)
    if entry is None:
        raise KeyError(agent_key)
    run = run if run is not None else pool.start_run(agent_key, task)
    # The operator's Codex boundary applies to every way Codex is reachable, not just the mounted
    # tool — otherwise `provider=codex` would be a way around AGENT_FLEET_CODEX_ENABLED and the
    # allowed-roots list. This also canonicalizes: `forbidden_state` globs its patterns from the
    # path it is handed, so a caller-supplied subdirectory would silently fingerprint the wrong
    # subtree and report no forbidden change at all — the deletions and symlink swaps git cannot
    # see are exactly what that fingerprinting exists to catch.
    root, timeout_s = enforce_codex_policy(request.cwd, request.timeout_s)
    request = request.model_copy(update={"cwd": root, "timeout_s": timeout_s})
    require_isolated_worktree(request.cwd)
    stdout = await asyncio.to_thread(_run_codex_sync, request, task)
    parsed = _parse_jsonl(stdout)
    if parsed.thread_count != 1 or parsed.thread_id is None:
        raise CodexRunError("codex did not emit exactly one usable thread.started event")
    _SESSION_ADAPTER.validate_python(parsed.thread_id)
    if not parsed.turn_completed or parsed.turn_failed:
        raise CodexRunError(f"codex did not complete the turn cleanly: {parsed.parse_errors}")
    output = _RUN_OUTPUT_ADAPTER.validate_python(parsed.final_message)
    pool.record_agent_run(run.run_id, entry.session_id)
    finished = pool.finish_run(run.run_id, output=output, total_cost_usd=None)
    return RunOutcome(
        output=output,
        run=finished,
        agent_runs=pool.list_agent_runs(run.run_id),
        structured_output=None,
        total_cost_usd=None,
    )
