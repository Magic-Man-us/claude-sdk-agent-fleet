from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from claude_agent_sdk import ClaudeAgentOptions
from pydantic import ConfigDict, Field, SkipValidation

from capdisc.base import FrozenModel

from .scope import RunScope
from .types import CodexModelId, CodexTimeoutSeconds, PromptBody, Provider, TaskBrief


class ClaudeRunRequest(FrozenModel):
    """The Claude Agent SDK variant of a provider run request: already-built live options plus an
    optional literal first-turn override. `options` is an SDK dataclass, not domain data, so it
    stays an opaque, `SkipValidation`-marked field — the same boundary treatment `engine.dispatch`
    already gives it. `SkipValidation` (rather than `arbitrary_types_allowed` alone) is required
    here: `ClaudeAgentOptions` carries a `Protocol`-typed field (`session_store`), and Pydantic's
    isinstance-schema builder cannot introspect a plain `Protocol` — recursing into the dataclass
    to build a schema fails at class-definition time, not just at validation time.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    provider: Literal[Provider.claude] = Provider.claude
    options: SkipValidation[ClaudeAgentOptions]
    prompt: TaskBrief | None = None


class CodexRunRequest(FrozenModel):
    """One Codex `exec` turn: a disposable, caller-prepared worktree, the scope it is held to, and
    the developer instructions derived from the teammate's own assembled system prompt."""

    provider: Literal[Provider.codex] = Provider.codex
    cwd: Path
    scope: RunScope
    model: CodexModelId
    timeout_s: CodexTimeoutSeconds
    developer_instructions: PromptBody
    output_schema_path: Path | None = None


ProviderRunRequest = Annotated[
    ClaudeRunRequest | CodexRunRequest,
    Field(discriminator="provider"),
]


class CodexRunConfig(FrozenModel):
    """Caller-facing Codex run settings for `run_teammate` — everything a Codex turn needs except
    the developer instructions, which are derived from the teammate's own assembled system prompt
    rather than supplied by the caller."""

    cwd: Path
    scope: RunScope
    model: CodexModelId
    timeout_s: CodexTimeoutSeconds
    output_schema_path: Path | None = None
