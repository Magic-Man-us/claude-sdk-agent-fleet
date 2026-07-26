from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import Field, model_validator

from capdisc.base import FrozenModel

#: Path segments a run must never be granted: version-control internals, virtualenvs, and caches.
#: Naming one in `allowed_paths` is always a mistake; naming one in `forbidden_paths` is the point.
PROTECTED_PARTS = frozenset(
    {
        ".git",
        ".hg",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".venv",
        "__pycache__",
        "node_modules",
    }
)
#: Characters that make a scope pattern a glob rather than a literal path.
GLOB_CHARS = frozenset("*?[")
#: The whole repository. Legal to read; a write run naming it is rejected.
WHOLE_REPO = "."

ScopePattern = Annotated[
    str,
    Field(
        min_length=1,
        max_length=200,
        title="Scope pattern",
        description="A repository-relative path or glob. `*` and `?` stay inside one segment, "
        "`**` spans any number, and a glob-free pattern covers that path and everything under it.",
        examples=["src/agent_fleet", "src/**/*.py", "."],
    ),
]


class RunMode(StrEnum):
    """Whether a run may change the working tree. `read` runs that change anything have failed."""

    read = "read"
    write = "write"


class RunScope(FrozenModel):
    """The paths a run may touch, and whether it may write at all.

    Declared up front and checked afterwards against what the working tree actually shows, so an
    agent that ignores its brief is caught by the diff rather than trusted to report honestly.
    An empty `allowed_paths` grants nothing, which is why a scope is optional rather than default:
    a run with no scope is unrestricted, a run with one is held to it.
    """

    mode: RunMode = RunMode.read
    allowed_paths: list[ScopePattern] = []
    forbidden_paths: list[ScopePattern] = []

    @model_validator(mode="after")
    def _is_enforceable(self) -> RunScope:
        """Reject a scope that cannot mean what it says.

        Escaping patterns (absolute, or containing `..`) are rejected everywhere: a scope is
        repository-relative by definition, and one that reaches outside cannot be checked against
        a repository-relative diff. Protected internals are rejected in `allowed_paths` only —
        forbidding them is the entire point of the other list. Overlap is rejected because a path
        that is both allowed and forbidden has no defined verdict, and a write run naming the
        whole repository is rejected because that is not a scope.
        """
        for label, patterns in (
            ("allowed_paths", self.allowed_paths),
            ("forbidden_paths", self.forbidden_paths),
        ):
            for pattern in patterns:
                parts = pattern.replace("\\", "/").split("/")
                if pattern.startswith("/") or ".." in parts:
                    raise ValueError(f"{label} entry {pattern!r} escapes the repository")
                if label == "allowed_paths" and (
                    PROTECTED_PARTS & set(parts)
                    or any(part == ".env" or part.startswith(".env.") for part in parts)
                ):
                    raise ValueError(f"allowed_paths entry {pattern!r} names protected content")

        if self.mode is RunMode.write and WHOLE_REPO in self.allowed_paths:
            raise ValueError("a write run cannot be granted the whole repository")
        return self
