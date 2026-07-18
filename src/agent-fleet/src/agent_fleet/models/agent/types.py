from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import Field
from pydantic.functional_validators import BeforeValidator

from capabilities_discovery.tokens import token_bounds


class ModelId(StrEnum):
    """Which model the agent runs on; `inherit` defers to the caller's model."""

    inherit = "inherit"
    opus = "opus"
    sonnet = "sonnet"
    haiku = "haiku"


class AgentEffort(StrEnum):
    """Mirrors the SDK's `EffortLevel` so any level the agent runtime accepts can be emitted."""

    low = "low"
    medium = "medium"
    high = "high"
    xhigh = "xhigh"
    max = "max"


class MemoryScope(StrEnum):
    """Which memory store the agent reads and writes — the user's home, the project, or a
    machine-local scope. Mirrors the SDK's `AgentDefinition.memory` choices."""

    user = "user"
    project = "project"
    local = "local"


class PermissionMode(StrEnum):
    """How the agent resolves tool-permission prompts. Values are the exact SDK `PermissionMode`
    literals (SDK 0.2.108); member names are their snake_case spellings."""

    default = "default"
    accept_edits = "acceptEdits"
    plan = "plan"
    bypass_permissions = "bypassPermissions"
    dont_ask = "dontAsk"
    auto = "auto"


class Isolation(StrEnum):
    """How the subagent's working tree is isolated — currently only a temporary git worktree."""

    worktree = "worktree"


class AgentColor(StrEnum):
    """Display color for the subagent in the task list and transcript."""

    red = "red"
    blue = "blue"
    green = "green"
    yellow = "yellow"
    purple = "purple"
    orange = "orange"
    pink = "pink"
    cyan = "cyan"


class SectionTag(StrEnum):
    """The tagged blocks a templated system prompt is built from, in render order."""

    role = "role"
    problem = "problem"
    tools = "tools"
    skills = "skills"
    instructions = "instructions"


TASK_TOKEN_MAX = 2400
PROMPT_TOKEN_MAX = 6000
PROMPT_MIN = 40
PROMPT_MAX = 20000

_DESCRIPTION_SUMMARY_MAX = 200


def _summarize(text: str) -> str:
    """Collapse whitespace and cap length so a long task brief fits AgentDescription."""
    return " ".join(text.split())[:_DESCRIPTION_SUMMARY_MAX]


AgentName = Annotated[
    str,
    Field(
        pattern=r"^[a-z0-9][a-z0-9-]{0,63}$",
        title="Agent name",
        description="The agent's unique name — lowercase letters, digits, and dashes.",
        examples=["changelog-curator"],
    ),
]
AgentKey = Annotated[
    str,
    Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$",
        title="Agent key",
        description="Stable external identifier for the pooled agent a caller manages — assign "
        "your own scheme (a ticket id, a slug, a UUID, anything meaningful to your use case). The "
        "pool's logical lookup key; distinct from the human-readable display name, and distinct "
        "from the harness's per-dispatch `AgentId`.",
        examples=["PROJ-4821", "nightly-triage", "3f2504e0-4f89"],
    ),
]
SessionId = Annotated[
    str,
    Field(
        pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        title="Session ID",
        description="The Claude Agent SDK session UUID a pool entry pins and resumes — a "
        "lowercase-hex UUID, generated internally, never the human-readable pool name.",
        examples=["3f2504e0-4f89-41d3-9a0c-0305e82c3301"],
    ),
]
RunId = Annotated[
    str,
    Field(
        pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        title="Run ID",
        description="Identifier for one invocation of a pooled agent — a lowercase-hex UUID minted "
        "per run, distinct from the session UUID: a run is a single dispatch, a session is the "
        "resumable conversation a run continues.",
        examples=["7c9e6679-7425-40de-944b-e07fc1f90ae7"],
    ),
]
AgentId = Annotated[
    str,
    Field(
        pattern=r"^[0-9a-f]{8,32}$",
        title="Agent ID",
        description="The harness's own id for one subagent dispatched via the `Agent` tool — "
        "captured from `TaskStartedMessage.task_id` when the dispatch launches. A short lowercase-"
        "hex token (no dashes), NOT a UUID: it is the durable, addressable handle for that one "
        "backgrounded subagent, and the only way to reach it later for a `SendMessage`-based "
        "resume. Distinct from `SessionId` — every subagent a run dispatches shares its parent's "
        "session id, so the session id cannot identify an individual subagent; this can.",
        examples=["a6bd0388ee89ccf94"],
    ),
]
AgentDescription = Annotated[
    str,
    BeforeValidator(_summarize),
    Field(
        min_length=8,
        title="Agent description",
        description=(
            "What the generated agent does and when to invoke it — a whitespace-collapsed "
            "summary capped at 200 characters."
        ),
        examples=["Proposes changelog entries from a diff."],
    ),
]
TaskBrief = Annotated[
    str,
    Field(
        min_length=12,
        max_length=8000,
        title="Task",
        description="What the generated agent should do — the job it is being built to perform.",
        examples=["Summarize the git commit history into grouped changelog entries."],
    ),
    token_bounds(TASK_TOKEN_MAX),
]
InitialPrompt = Annotated[
    str,
    Field(
        min_length=1,
        max_length=8000,
        title="Initial prompt",
        description="First user turn auto-submitted when the agent runs as the main session "
        "agent (--agent / the `agent` setting), prepended to the user's prompt; inert on normal "
        "delegated invocation.",
        examples=["Start by listing the files you will change."],
    ),
]
PromptBody = Annotated[
    str,
    Field(
        min_length=PROMPT_MIN,
        max_length=PROMPT_MAX,
        title="System prompt",
        description="The generated agent's system prompt body.",
    ),
    token_bounds(PROMPT_TOKEN_MAX),
]
FindingContent = Annotated[
    str,
    Field(
        min_length=1,
        max_length=20000,
        title="Finding",
        description="One finding a lens (or the supervisor) records for a pooled agent — the "
        "text preserved in the shared findings document. Append-only; never overwritten.",
        examples=["Unbounded user input reaches the SQL query in reports.py:42 without escaping."],
    ),
]
MaxTurns = Annotated[
    int,
    Field(
        ge=1,
        le=1000,
        title="Max turns",
        description="Most agent turns before the run is cut off.",
        examples=[20],
    ),
]
ToolBudget = Annotated[
    int,
    Field(ge=1, le=40, title="Tool budget", description="Max tools the agent may carry."),
]
SkillBudget = Annotated[
    int,
    Field(ge=0, le=16, title="Skill budget", description="Max skills the agent may carry."),
]
TeamSlug = Annotated[
    str,
    Field(
        pattern=r"^[a-z0-9][a-z0-9-]{0,63}$",
        title="Team",
        description="Name of the team that owns the generated agent.",
    ),
]

DEFAULT_TEAM = "default"
DEFAULT_TOOL_BUDGET = 8
DEFAULT_SKILL_BUDGET = 4
