from __future__ import annotations

import asyncio
import dataclasses
import json
import shutil
import subprocess
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from claude_agent_sdk import McpSdkServerConfig
from claude_agent_sdk.types import AgentDefinition
from mcp.types import CallToolRequest, CallToolRequestParams, ListToolsRequest, ServerResult

from agent_fleet import AgentSpec
from agent_fleet.engine import codex_tool
from agent_fleet.engine.codex_tool import (
    CODEX_SERVER,
    CODEX_TOOL,
    CodexPolicy,
    CodexResult,
    CodexRunArgs,
    CodexRunError,
    build_codex_server,
    grant_codex_to_subagent,
    run_codex,
    with_codex_tool,
)
from agent_fleet.engine.render import to_agent_definition, to_options

_PROMPT = "Inspect the repository and explain the failing test with exact evidence."


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    binary = shutil.which("git")
    assert binary is not None
    return subprocess.run(  # noqa: S603
        [binary, "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.email", "tests@example.com")
    _git(root, "config", "user.name", "Fleet Tests")
    (root / "README.md").write_text("# fixture\n", encoding="utf-8")
    _git(root, "add", "README.md")
    _git(root, "commit", "-m", "fixture")
    return root


def _policy(tmp_path: Path, *, allow_write: bool = False) -> CodexPolicy:
    return CodexPolicy(
        allowed_roots=(tmp_path,),
        allow_workspace_write=allow_write,
        max_timeout_seconds=1_200,
    )


def _args(root: Path, **overrides: object) -> CodexRunArgs:
    values: dict[str, object] = {"prompt": _PROMPT, "cwd": root}
    values.update(overrides)
    return CodexRunArgs.model_validate(values)


def _success_stream(thread_id: UUID | None = None) -> bytes:
    selected = thread_id or uuid4()
    events = [
        {"type": "thread.started", "thread_id": str(selected)},
        {
            "type": "item.completed",
            "item": {
                "type": "command_execution",
                "command": "git status --short",
                "status": "completed",
            },
        },
        {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "The repository is clean."},
        },
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 20,
                "cached_input_tokens": 10,
                "output_tokens": 5,
                "reasoning_output_tokens": 2,
            },
        },
    ]
    return ("\n".join(json.dumps(event) for event in events) + "\n").encode()


async def _call_codex(server: McpSdkServerConfig, arguments: dict[str, object]) -> ServerResult:
    handler = server["instance"].request_handlers[CallToolRequest]
    request = CallToolRequest(
        method="tools/call",
        params=CallToolRequestParams(name="codex_run", arguments=arguments),
    )
    return await handler(request)


def _spec() -> AgentSpec:
    return AgentSpec(
        name="supervisor",
        description="Coordinates implementation and verification workers.",
        system_prompt=(
            "You coordinate implementation and verification workers, then report results."
        ),
        tools=["Read"],
    )


def test_parse_stream_preserves_thread_commands_usage_and_last_message() -> None:
    transcript = codex_tool._parse_stream(
        _success_stream().decode()
        + '{"type":"item.completed","item":{"type":"agent_message","text":"final"}}\n'
        + '{"type":"preview.unknown","new_field":true}\n'
    )

    assert transcript.thread_id is not None
    assert transcript.commands_run == ("git status --short",)
    assert transcript.final_message == "final"
    assert transcript.completed is True
    assert transcript.failed is False
    assert transcript.usage is not None
    assert transcript.usage.cached_input_tokens == 10


def test_parse_stream_tolerates_non_json_and_marks_failure() -> None:
    transcript = codex_tool._parse_stream(
        "progress\n"
        '{"type":"thread.started","thread_id":"not-a-uuid"}\n'
        '{"type":"turn.failed","message":"sandbox denied the command"}\n'
    )

    assert transcript.thread_id is None
    assert transcript.failed is True
    assert transcript.completed is False
    assert transcript.failure_detail == "sandbox denied the command"


def test_child_environment_strips_service_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")
    monkeypatch.setenv("PATH", "/safe/path")

    child = codex_tool._child_env()

    assert "OPENAI_API_KEY" not in child
    assert "AWS_SECRET_ACCESS_KEY" not in child
    assert child["PATH"] == "/safe/path"


def test_read_only_requires_allowlisted_canonical_git_root(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    validated = codex_tool._validate_request(_args(root), _policy(tmp_path))
    assert validated == root.resolve()

    with pytest.raises(CodexRunError, match="absolute"):
        codex_tool._validate_request(_args(Path("repo")), _policy(tmp_path))
    subdirectory = root / "src"
    subdirectory.mkdir()
    with pytest.raises(CodexRunError, match="canonical"):
        codex_tool._validate_request(_args(subdirectory), _policy(tmp_path))
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(CodexRunError, match="outside"):
        codex_tool._validate_request(_args(root), CodexPolicy(allowed_roots=(other,)))


def test_policy_rejects_filesystem_and_home_roots(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    with pytest.raises(CodexRunError, match="filesystem or home"):
        codex_tool._validate_request(_args(root), CodexPolicy(allowed_roots=(Path("/"),)))
    with pytest.raises(CodexRunError, match="filesystem or home"):
        codex_tool._validate_request(_args(root), CodexPolicy(allowed_roots=(Path.home(),)))


def test_policy_requires_absolute_existing_directory_roots(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    with pytest.raises(CodexRunError, match="absolute"):
        codex_tool._validate_request(_args(root), CodexPolicy(allowed_roots=(Path("repos"),)))
    with pytest.raises(CodexRunError, match="existing directory"):
        codex_tool._validate_request(
            _args(root),
            CodexPolicy(allowed_roots=(tmp_path / "missing",)),
        )
    file_root = tmp_path / "not-a-directory"
    file_root.write_text("fixture\n", encoding="utf-8")
    with pytest.raises(CodexRunError, match="existing directory"):
        codex_tool._validate_request(_args(root), CodexPolicy(allowed_roots=(file_root,)))


def test_workspace_write_requires_operator_enablement(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    with pytest.raises(CodexRunError, match="disabled"):
        codex_tool._validate_request(
            _args(root, sandbox="workspace-write"),
            _policy(tmp_path),
        )


def test_workspace_write_requires_clean_detached_linked_worktree(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    write_policy = _policy(tmp_path, allow_write=True)
    with pytest.raises(CodexRunError, match="detached HEAD"):
        codex_tool._validate_request(_args(root, sandbox="workspace-write"), write_policy)

    worktree = tmp_path / "worktree"
    _git(root, "worktree", "add", "--detach", str(worktree), "HEAD")
    assert (
        codex_tool._validate_request(
            _args(worktree, sandbox="workspace-write"),
            write_policy,
        )
        == worktree.resolve()
    )

    (worktree / "dirty.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(CodexRunError, match="start clean"):
        codex_tool._validate_request(
            _args(worktree, sandbox="workspace-write"),
            write_policy,
        )


def test_timeout_cannot_exceed_operator_ceiling(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    with pytest.raises(CodexRunError, match="operator limit"):
        codex_tool._validate_request(
            _args(root, timeout_seconds=1_201),
            _policy(tmp_path),
        )


def test_resume_prompt_has_portable_byte_limit(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    thread_id = uuid4()
    with pytest.raises(CodexRunError, match="portability limit"):
        codex_tool._validate_request(
            _args(
                root,
                prompt="é" * (codex_tool.MAX_RESUME_PROMPT_BYTES // 2 + 1),
                resume_thread_id=thread_id,
            ),
            _policy(tmp_path),
        )


def test_fresh_command_uses_stdin_and_fixed_automation_policy(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    args = _args(root, effort="high", model="gpt-5.6-terra")

    argv, payload = codex_tool._command("/usr/bin/codex", args, root)

    assert argv[:3] == ["/usr/bin/codex", "--model", "gpt-5.6-terra"]
    assert argv[3:5] == ["--sandbox", "read-only"]
    assert "--ask-for-approval" in argv and "never" in argv
    assert "--strict-config" in argv
    assert "--ignore-user-config" in argv
    assert "--skip-git-repo-check" not in argv
    configured = {argv[index + 1] for index, value in enumerate(argv) if value == "--config"}
    assert set(codex_tool.HARDENED_CONFIG_OVERRIDES) <= configured
    assert "sandbox_workspace_write.network_access=false" in configured
    assert "sandbox_workspace_write.writable_roots=[]" in configured
    assert "features.hooks=false" in configured
    assert 'shell_environment_policy.inherit="core"' in configured
    assert f'projects.{json.dumps(str(root))}.trust_level="untrusted"' in configured
    assert argv[-1] == "-"
    assert payload == _PROMPT.encode()


def test_resume_command_targets_exact_uuid_without_stdin(tmp_path: Path) -> None:
    root = _repo(tmp_path)
    thread_id = uuid4()

    argv, payload = codex_tool._command(
        "/usr/bin/codex",
        _args(root, resume_thread_id=thread_id),
        root,
    )

    assert argv[-3:] == ["resume", str(thread_id), _PROMPT]
    assert payload == b""


def test_run_codex_returns_structured_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repo(tmp_path)
    real_which = shutil.which
    captured: dict[str, object] = {}

    def fake_which(name: str) -> str | None:
        return "/opt/codex/bin/codex" if name == "codex" else real_which(name)

    async def fake_execute(
        argv: list[str],
        *,
        root: Path,
        stdin_payload: bytes,
        timeout_seconds: int,
    ) -> tuple[int, bytes, bytes]:
        captured.update(
            argv=argv,
            root=root,
            stdin_payload=stdin_payload,
            timeout_seconds=timeout_seconds,
        )
        return 0, _success_stream(), b""

    monkeypatch.setattr(codex_tool.shutil, "which", fake_which)
    monkeypatch.setattr(codex_tool, "_execute", fake_execute)

    result = asyncio.run(run_codex(_args(root), _policy(tmp_path)))

    assert result.ok is True
    assert result.final_message == "The repository is clean."
    assert result.commands_run == ("git status --short",)
    assert result.thread_id is not None
    assert captured["root"] == root
    assert captured["stdin_payload"] == _PROMPT.encode()


def test_run_codex_rejects_missing_binary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _repo(tmp_path)
    real_which = shutil.which
    monkeypatch.setattr(
        codex_tool.shutil,
        "which",
        lambda name: None if name == "codex" else real_which(name),
    )
    with pytest.raises(CodexRunError, match="codex is not on PATH"):
        asyncio.run(run_codex(_args(root), _policy(tmp_path)))


def test_run_codex_raises_on_failed_run_without_final_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repo(tmp_path)
    real_which = shutil.which

    def fake_which(name: str) -> str | None:
        return "/opt/codex/bin/codex" if name == "codex" else real_which(name)

    async def fake_execute(
        argv: list[str],
        *,
        root: Path,
        stdin_payload: bytes,
        timeout_seconds: int,
    ) -> tuple[int, bytes, bytes]:
        del argv, root, stdin_payload, timeout_seconds
        return 1, b'{"type":"turn.failed","message":"policy rejected"}\n', b"details"

    monkeypatch.setattr(codex_tool.shutil, "which", fake_which)
    monkeypatch.setattr(codex_tool, "_execute", fake_execute)

    with pytest.raises(CodexRunError, match="policy rejected"):
        asyncio.run(run_codex(_args(root), _policy(tmp_path)))


def test_bounded_reader_fails_closed_on_oversized_output() -> None:
    async def scenario() -> None:
        stream = asyncio.StreamReader()
        stream.feed_data(b"x" * 9)
        stream.feed_eof()
        await codex_tool._read_limited(stream, limit=8, stream_name="stdout")

    with pytest.raises(CodexRunError, match="stdout exceeded 8 bytes"):
        asyncio.run(scenario())


def test_execute_terminates_child_when_caller_is_cancelled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    terminated = False

    class HangingProcess:
        returncode: int | None = None
        stdin = None
        stdout = object()
        stderr = object()

        async def wait(self) -> int:
            await asyncio.Future()
            raise AssertionError("unreachable")

    process = HangingProcess()

    async def fake_subprocess(*args: str, **kwargs: object) -> HangingProcess:
        del args, kwargs
        return process

    async def fake_terminate(proc: object) -> None:
        nonlocal terminated
        assert proc is process
        terminated = True

    async def fake_read(*args: object, **kwargs: object) -> bytes:
        del args, kwargs
        await asyncio.Future()
        raise AssertionError("unreachable")

    monkeypatch.setattr(codex_tool.asyncio, "create_subprocess_exec", fake_subprocess)
    monkeypatch.setattr(codex_tool, "_terminate", fake_terminate)
    monkeypatch.setattr(codex_tool, "_read_limited", fake_read)

    async def scenario() -> None:
        task = asyncio.create_task(
            codex_tool._execute(
                ["/opt/codex/bin/codex"],
                root=tmp_path,
                stdin_payload=b"",
                timeout_seconds=30,
            )
        )
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert terminated is True


def test_execute_timeout_includes_stalled_stdin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    terminated = False

    class StalledWriter:
        closed = False

        def write(self, payload: bytes) -> None:
            assert payload == b"prompt"

        async def drain(self) -> None:
            await asyncio.Future()

        def close(self) -> None:
            self.closed = True

        async def wait_closed(self) -> None:
            return

    class Process:
        returncode: int | None = None
        stdin = StalledWriter()
        stdout = object()
        stderr = object()

        async def wait(self) -> int:
            return 0

    process = Process()

    async def fake_subprocess(*args: str, **kwargs: object) -> Process:
        del args, kwargs
        return process

    async def fake_read(*args: object, **kwargs: object) -> bytes:
        del args, kwargs
        return b""

    async def fake_terminate(proc: object) -> None:
        nonlocal terminated
        assert proc is process
        terminated = True

    monkeypatch.setattr(codex_tool.asyncio, "create_subprocess_exec", fake_subprocess)
    monkeypatch.setattr(codex_tool, "_read_limited", fake_read)
    monkeypatch.setattr(codex_tool, "_terminate", fake_terminate)

    with pytest.raises(CodexRunError, match="exceeded"):
        asyncio.run(
            codex_tool._execute(
                ["/opt/codex/bin/codex"],
                root=tmp_path,
                stdin_payload=b"prompt",
                timeout_seconds=0.01,
            )
        )
    assert terminated is True
    assert process.stdin.closed is True


def test_codex_server_returns_structured_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _repo(tmp_path)
    thread_id = uuid4()

    async def fake_run(args: CodexRunArgs, policy: CodexPolicy) -> CodexResult:
        assert args.cwd == root
        assert policy == _policy(tmp_path)
        return CodexResult(
            ok=True,
            exit_code=0,
            duration_seconds=1.5,
            thread_id=thread_id,
            final_message="done",
        )

    monkeypatch.setattr(codex_tool, "run_codex", fake_run)
    result = asyncio.run(
        _call_codex(build_codex_server(_policy(tmp_path)), {"prompt": _PROMPT, "cwd": str(root)})
    )

    assert result.root.isError is False
    payload = json.loads(result.root.content[0].text)
    assert payload["thread_id"] == str(thread_id)
    assert payload["final_message"] == "done"


def test_codex_server_advertises_strict_bounded_schema(tmp_path: Path) -> None:
    server = build_codex_server(_policy(tmp_path))
    handler = server["instance"].request_handlers[ListToolsRequest]
    result = asyncio.run(handler(ListToolsRequest(method="tools/list")))
    declared = result.root.tools[0]

    assert declared.inputSchema["additionalProperties"] is False
    assert declared.inputSchema["required"] == ["prompt", "cwd"]
    assert declared.inputSchema["properties"]["prompt"]["maxLength"] == codex_tool.MAX_PROMPT_CHARS
    assert declared.inputSchema["properties"]["sandbox"]["enum"] == [
        "read-only",
        "workspace-write",
    ]


def test_codex_server_returns_mcp_error_without_echoing_prompt(tmp_path: Path) -> None:
    sensitive_prompt = "do something with private marker REDACT-ME"
    result = asyncio.run(
        _call_codex(
            build_codex_server(_policy(tmp_path)),
            {"prompt": sensitive_prompt, "cwd": "relative"},
        )
    )

    assert result.root.isError is True
    assert sensitive_prompt not in result.root.content[0].text


def test_with_codex_tool_mounts_server_and_is_idempotent(tmp_path: Path) -> None:
    base = to_options(_spec())
    once = with_codex_tool(base, _policy(tmp_path))
    twice = with_codex_tool(once, _policy(tmp_path))

    assert CODEX_SERVER in once.mcp_servers
    assert CODEX_TOOL in once.allowed_tools
    assert CODEX_TOOL not in base.allowed_tools
    assert twice.allowed_tools.count(CODEX_TOOL) == 1


def test_with_codex_tool_preserves_json_file_mcp_servers(tmp_path: Path) -> None:
    config = tmp_path / "mcp.json"
    config.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "external": {
                        "type": "http",
                        "url": "https://mcp.example.test",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    base = dataclasses.replace(to_options(_spec()), mcp_servers=config)

    mounted = with_codex_tool(base, _policy(tmp_path))

    assert set(mounted.mcp_servers) == {"external", CODEX_SERVER}


def test_with_codex_tool_rejects_reserved_server_collision(tmp_path: Path) -> None:
    base = dataclasses.replace(
        to_options(_spec()),
        mcp_servers={
            CODEX_SERVER: {
                "type": "http",
                "url": "https://untrusted.example.test",
            }
        },
    )

    with pytest.raises(CodexRunError, match="reserved"):
        with_codex_tool(base, _policy(tmp_path))


def test_with_codex_tool_none_is_explicitly_disabled() -> None:
    base = to_options(_spec())
    assert with_codex_tool(base, None) is base


def test_grant_codex_to_subagent_is_idempotent() -> None:
    base = to_agent_definition(_spec())
    once = grant_codex_to_subagent(base, enabled=True)
    twice = grant_codex_to_subagent(once, enabled=True)

    assert once.mcpServers == [CODEX_SERVER]
    assert once.tools is not None and CODEX_TOOL in once.tools
    assert twice.mcpServers is not None and twice.mcpServers.count(CODEX_SERVER) == 1
    assert twice.tools is not None and twice.tools.count(CODEX_TOOL) == 1


def test_grant_codex_preserves_inherit_all_and_disabled_state() -> None:
    inherit = AgentDefinition(
        description="Inherits all tools.",
        prompt="Use the available tools to finish the task.",
        tools=None,
    )
    granted = grant_codex_to_subagent(inherit, enabled=True)
    assert granted.tools is None
    assert granted.mcpServers == [CODEX_SERVER]
    assert grant_codex_to_subagent(inherit, enabled=False) is inherit
