"""The Codex provider adapter: argv/JSONL parsing units, then live runs against a fake `codex`."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from agent_fleet import AgentPool, AgentSpec
from agent_fleet.engine.providers.codex import (
    CodexRunError,
    _argv,
    _parse_jsonl,
    run_codex_capture,
)
from agent_fleet.models.agent import CodexModelId, CodexRunRequest, RunMode, RunScope

_AGENT_KEY = "PROJ-CODEX"
_TASK = "audit the codebase for security vulnerabilities now"
_DEVELOPER_INSTRUCTIONS = "You are auditor. Audit the code for vulnerabilities and stop." * 2


def _spec() -> AgentSpec:
    return AgentSpec(
        name="auditor",
        description="Audits code for vulnerabilities.",
        system_prompt=_DEVELOPER_INSTRUCTIONS,
    )


def _pool(tmp_path: Path) -> AgentPool:
    return AgentPool(tmp_path / "pool.db")


def _request(cwd: Path, *, mode: RunMode = RunMode.read, timeout_s: int = 30) -> CodexRunRequest:
    return CodexRunRequest(
        cwd=cwd,
        scope=RunScope(mode=mode, allowed_paths=["allowed"] if mode is RunMode.write else []),
        model=CodexModelId.gpt_5_6_sol,
        timeout_s=timeout_s,
        developer_instructions=_DEVELOPER_INSTRUCTIONS,
    )


# ---------------------------------------------------------------------------
# _argv / _parse_jsonl unit tests
# ---------------------------------------------------------------------------


def test_argv_maps_read_mode_to_read_only_sandbox(tmp_path: Path) -> None:
    argv = _argv(_request(tmp_path, mode=RunMode.read), "codex")
    assert "--sandbox" in argv
    assert argv[argv.index("--sandbox") + 1] == "read-only"
    assert "--output-schema" not in argv


def test_argv_maps_write_mode_to_workspace_write_sandbox(tmp_path: Path) -> None:
    argv = _argv(_request(tmp_path, mode=RunMode.write), "codex")
    assert argv[argv.index("--sandbox") + 1] == "workspace-write"


def test_argv_embeds_developer_instructions_as_quoted_json(tmp_path: Path) -> None:
    argv = _argv(_request(tmp_path), "codex")
    config = next(a for a in argv if a.startswith("developer_instructions="))
    encoded = config.removeprefix("developer_instructions=")
    assert TypeAdapter(str).validate_json(encoded) == _DEVELOPER_INSTRUCTIONS


def test_argv_includes_output_schema_only_when_given(tmp_path: Path) -> None:
    schema = tmp_path / "schema.json"
    with_schema = _request(tmp_path).model_copy(update={"output_schema_path": schema})
    argv = _argv(with_schema, "codex")
    assert "--output-schema" in argv
    assert argv[argv.index("--output-schema") + 1] == str(schema)


def test_parse_jsonl_extracts_thread_and_final_message() -> None:
    stream = "\n".join(
        [
            '{"type": "thread.started", "thread_id": "fixture-thread"}',
            '{"type": "item.completed", "item": {"type": "agent_message", "text": "done"}}',
            '{"type": "turn.completed"}',
        ]
    )
    parsed = _parse_jsonl(stream)
    assert parsed.thread_id == "fixture-thread"
    assert parsed.thread_count == 1
    assert parsed.final_message == "done"
    assert parsed.turn_completed
    assert not parsed.turn_failed
    assert parsed.parse_errors == []


def test_parse_jsonl_tolerates_unrecognized_event_types() -> None:
    """Codex's protocol is external and evolving; an event type this adapter doesn't know about
    must not be treated as a parse error."""
    stream = '{"type": "session_configured", "model": "gpt-5.6-sol"}'
    parsed = _parse_jsonl(stream)
    assert parsed.parse_errors == []


def test_parse_jsonl_records_non_json_lines_as_errors() -> None:
    parsed = _parse_jsonl("not json at all")
    assert len(parsed.parse_errors) == 1
    assert "non-JSON" in parsed.parse_errors[0]


def test_parse_jsonl_flags_turn_failed() -> None:
    stream = '{"type": "turn.failed"}'
    parsed = _parse_jsonl(stream)
    assert parsed.turn_failed
    assert parsed.parse_errors == ["codex reported turn.failed"]


# ---------------------------------------------------------------------------
# run_codex_capture: live runs against a fake `codex` executable on PATH
# ---------------------------------------------------------------------------


def _run_git(root: Path, *arguments: str) -> None:
    subprocess.run(["git", "-C", str(root), *arguments], check=True, capture_output=True)  # noqa: S603, S607


def _init_repo(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _run_git(root, "init", "-q")
    _run_git(root, "config", "user.email", "test@example.com")
    _run_git(root, "config", "user.name", "Test")
    (root / "tracked.txt").write_text("original\n")
    _run_git(root, "add", "-A")
    _run_git(root, "commit", "-q", "-m", "initial")


@pytest.fixture
def worktree(tmp_path: Path) -> Path:
    """A clean, detached-HEAD linked worktree — the shape `run_codex_capture` requires."""
    repo = tmp_path / "repo"
    _init_repo(repo)
    linked = tmp_path / "linked"
    _run_git(repo, "worktree", "add", "--detach", "-q", str(linked))
    return linked


def _install_fake_codex(bin_dir: Path, monkeypatch: pytest.MonkeyPatch, body: str) -> None:
    """Put an executable named `codex` on PATH that runs `body` as its own script."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    script = bin_dir / "codex"
    script.write_text(f"#!{sys.executable}\n{body}")
    script.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")


_SUCCESS_SCRIPT = """
import sys
sys.stdin.read()
print('{"type": "thread.started", "thread_id": "fixture-thread"}')
print('{"type": "session_configured", "unrelated": true}')
print('{"type": "item.completed", "item": {"type": "agent_message", "text": "codex reply text"}}')
print('{"type": "turn.completed"}')
"""


def test_run_codex_capture_records_the_run_and_one_agent_row(
    worktree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_codex(tmp_path / "bin", monkeypatch, _SUCCESS_SCRIPT)
    pool = _pool(tmp_path)
    entry = pool.save(_AGENT_KEY, _spec())

    outcome = asyncio.run(run_codex_capture(pool, _AGENT_KEY, _TASK, _request(worktree)))

    assert outcome.output == "codex reply text"
    assert outcome.run.finished_at is not None
    assert outcome.total_cost_usd is None
    assert outcome.structured_output is None
    assert len(outcome.agent_runs) == 1
    assert (
        outcome.agent_runs[0].session_id == entry.session_id
    )  # pool bookkeeping UUID, not codex's thread id
    assert outcome.agent_runs[0].tool_use_id is None

    stored = pool.get_run(outcome.run.run_id)
    assert stored is not None
    assert stored.output == "codex reply text"


def test_run_codex_capture_raises_on_missing_agent_key(worktree: Path, tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    with pytest.raises(KeyError):
        asyncio.run(run_codex_capture(pool, "nobody-here", _TASK, _request(worktree)))


def test_run_codex_capture_rejects_a_non_isolated_worktree(tmp_path: Path) -> None:
    dirty = tmp_path / "repo"
    _init_repo(dirty)  # the main worktree, not a linked one
    _run_git(dirty, "checkout", "-q", "--detach", "HEAD")
    pool = _pool(tmp_path)
    pool.save(_AGENT_KEY, _spec())
    with pytest.raises(ValueError, match="linked disposable worktree"):
        asyncio.run(run_codex_capture(pool, _AGENT_KEY, _TASK, _request(dirty)))


def test_run_codex_capture_raises_on_nonzero_exit(
    worktree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_codex(
        tmp_path / "bin",
        monkeypatch,
        "import sys\nsys.stdin.read()\nsys.exit(1)\n",
    )
    pool = _pool(tmp_path)
    pool.save(_AGENT_KEY, _spec())
    with pytest.raises(CodexRunError, match="codex exited 1"):
        asyncio.run(run_codex_capture(pool, _AGENT_KEY, _TASK, _request(worktree)))


def test_run_codex_capture_raises_when_thread_started_is_missing(
    worktree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_codex(
        tmp_path / "bin",
        monkeypatch,
        'import sys\nsys.stdin.read()\nprint(\'{"type": "turn.completed"}\')\n',
    )
    pool = _pool(tmp_path)
    pool.save(_AGENT_KEY, _spec())
    with pytest.raises(CodexRunError, match="exactly one usable thread\\.started"):
        asyncio.run(run_codex_capture(pool, _AGENT_KEY, _TASK, _request(worktree)))


def test_run_codex_capture_raises_on_turn_failed(
    worktree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_codex(
        tmp_path / "bin",
        monkeypatch,
        "import sys\nsys.stdin.read()\n"
        'print(\'{"type": "thread.started", "thread_id": "fixture-thread"}\')\n'
        'print(\'{"type": "turn.failed"}\')\n',
    )
    pool = _pool(tmp_path)
    pool.save(_AGENT_KEY, _spec())
    with pytest.raises(CodexRunError, match="did not complete the turn cleanly"):
        asyncio.run(run_codex_capture(pool, _AGENT_KEY, _TASK, _request(worktree)))


def test_run_codex_capture_raises_when_codex_is_not_on_path(
    worktree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # git must stay reachable (require_isolated_worktree needs it); only "codex" is absent
    git_executable = shutil.which("git")
    assert git_executable is not None
    monkeypatch.setenv("PATH", str(Path(git_executable).parent))
    pool = _pool(tmp_path)
    pool.save(_AGENT_KEY, _spec())
    with pytest.raises(CodexRunError, match="not installed"):
        asyncio.run(run_codex_capture(pool, _AGENT_KEY, _TASK, _request(worktree)))


def test_run_codex_capture_raises_on_a_forbidden_write(
    worktree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A codex run that writes outside `allowed_paths` is caught by the post-run scope diff, even
    though the transport itself reported success."""
    script = (
        "import sys\nsys.stdin.read()\n"
        "open('outside.txt', 'w').write('surprise')\n"
        'print(\'{"type": "thread.started", "thread_id": "fixture-thread"}\')\n'
        'print(\'{"type": "item.completed", "item": {"type": "agent_message", '
        '"text": "done"}}\')\n'
        'print(\'{"type": "turn.completed"}\')\n'
    )
    _install_fake_codex(tmp_path / "bin", monkeypatch, script)
    pool = _pool(tmp_path)
    pool.save(_AGENT_KEY, _spec())
    with pytest.raises(CodexRunError, match="outside allowed_paths"):
        asyncio.run(
            run_codex_capture(pool, _AGENT_KEY, _TASK, _request(worktree, mode=RunMode.write))
        )


def test_run_codex_capture_raises_on_timeout(
    worktree: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_codex(
        tmp_path / "bin",
        monkeypatch,
        "import time\ntime.sleep(5)\n",
    )
    pool = _pool(tmp_path)
    pool.save(_AGENT_KEY, _spec())
    with pytest.raises(CodexRunError, match="timed out"):
        asyncio.run(run_codex_capture(pool, _AGENT_KEY, _TASK, _request(worktree, timeout_s=1)))
