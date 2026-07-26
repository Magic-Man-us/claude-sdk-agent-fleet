"""Models for the dependency-light fixed-agent runtime."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

PermissionMode = Literal["default", "acceptEdits", "plan", "bypassPermissions", "dontAsk"]


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RuntimeAgentSpec(_FrozenModel):
    """Fixed role definition stored by :class:`RuntimeAgentPool`.

    The host remains responsible for applying its final tool, sandbox,
    structured-output, environment, and credential policy to the returned
    Claude Agent SDK options.
    """

    name: str = Field(min_length=1, max_length=64)
    description: str = Field(min_length=1, max_length=1000)
    system_prompt: str = Field(min_length=1, max_length=100_000)
    model: str = Field(default="inherit", min_length=1, max_length=200)
    max_turns: int | None = Field(default=None, ge=1, le=1_000)
    permission_mode: PermissionMode | None = None


class RuntimePoolEntry(_FrozenModel):
    agent_key: str = Field(min_length=1, max_length=200)
    spec: RuntimeAgentSpec
    session_id: str = Field(min_length=1, max_length=200)
    cwd: Path
    created_at: datetime
    updated_at: datetime


class RuntimeRunRecord(_FrozenModel):
    run_id: str = Field(min_length=1, max_length=200)
    agent_key: str = Field(min_length=1, max_length=200)
    task: str = Field(min_length=1, max_length=8_000)
    started_at: datetime
    finished_at: datetime | None = None


class RuntimeAgentRunRecord(_FrozenModel):
    run_id: str
    session_id: str
    recorded_at: datetime
    tool_use_id: str | None = None
    agent_name: str | None = None
    agent_id: str | None = None


class RuntimeRunOutcome(_FrozenModel):
    output: str
    run: RuntimeRunRecord
    agent_runs: list[RuntimeAgentRunRecord]
    structured_output: JsonValue | None = None
    total_cost_usd: float | None = None
