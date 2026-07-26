from __future__ import annotations

import re
from functools import cache

from ..models.agent import GLOB_CHARS, WHOLE_REPO, RunMode, RunScope, ScopePattern


def _segment_regex(segment: ScopePattern) -> str:
    """One path segment's glob, where `*` and `?` never cross a separator."""
    parts: list[str] = []
    index = 0
    while index < len(segment):
        char = segment[index]
        if char == "*":
            parts.append("[^/]*")
        elif char == "?":
            parts.append("[^/]")
        elif char == "[":
            close = segment.find("]", index + 1)
            if close != -1:
                body = segment[index + 1 : close]
                parts.append("[" + ("^" + body[1:] if body.startswith("!") else body) + "]")
                index = close + 1
                continue
            parts.append(re.escape(char))
        else:
            parts.append(re.escape(char))
        index += 1
    return "".join(parts)


@cache
def _pattern_regex(pattern: ScopePattern) -> re.Pattern[str]:
    """Compile one scope pattern.

    `PurePosixPath.match` is right-anchored with no recursive form, which reads `"src"` as nothing
    at all, `"*.py"` as every `.py` at any depth, and `"src/**/*"` as exactly two levels down.
    Scope is the boundary a run is held to, so the semantics are spelled out instead: a glob-free
    pattern covers that path and everything beneath it, `*` and `?` stay inside one segment, and
    `**` spans any number of them.
    """
    if pattern == WHOLE_REPO:
        return re.compile(".*", re.DOTALL)
    if not any(char in pattern for char in GLOB_CHARS):
        return re.compile(re.escape(pattern) + "(?:/.*)?", re.DOTALL)
    parts: list[str] = []
    segments = pattern.split("/")
    for index, segment in enumerate(segments):
        last = index == len(segments) - 1
        if segment == "**":
            parts.append("(?:[^/]+(?:/[^/]+)*)?" if last else "(?:[^/]+/)*")
            continue
        parts.append(_segment_regex(segment))
        if not last:
            parts.append("/")
    return re.compile("".join(parts), re.DOTALL)


def matches(path: str, patterns: list[ScopePattern]) -> bool:
    """Whether `path` is covered by any of `patterns`."""
    candidate = path.replace("\\", "/")
    return any(
        pattern == candidate or _pattern_regex(pattern).fullmatch(candidate) is not None
        for pattern in patterns
    )


def overlaps(allowed: list[ScopePattern], forbidden: list[ScopePattern]) -> bool:
    """Whether any allowed pattern and forbidden pattern cover common ground.

    A path that is both allowed and forbidden has no defined verdict, so a scope declaring one is
    rejected before a run starts. The whole-repo grant is exempt: it is the read-everything case
    where the forbidden list is precisely the carve-out.
    """
    return any(
        allowed_pattern != WHOLE_REPO
        and (
            allowed_pattern == forbidden_pattern
            or matches(allowed_pattern, [forbidden_pattern])
            or matches(forbidden_pattern, [allowed_pattern])
        )
        for allowed_pattern in allowed
        for forbidden_pattern in forbidden
    )


def violations(scope: RunScope, changed_paths: list[str]) -> list[str]:
    """The reasons `changed_paths` breaks `scope` — empty when the run stayed inside it.

    Checked against what the working tree actually shows rather than what the run reported, so an
    agent that exceeded its brief is caught by the diff instead of trusted to admit it.

    Args:
        scope: The declared scope the run was held to.
        changed_paths: Repository-relative paths the run actually changed.

    Returns:
        One message per distinct violation, in a stable order; empty when the run complied.
    """
    if not changed_paths:
        return []
    problems: list[str] = []
    if scope.mode is RunMode.read:
        problems.append(f"read-only run changed files: {sorted(changed_paths)}")
    outside = sorted(path for path in changed_paths if not matches(path, scope.allowed_paths))
    if outside:
        problems.append(f"changes outside allowed_paths: {outside}")
    forbidden = sorted(path for path in changed_paths if matches(path, scope.forbidden_paths))
    if forbidden:
        problems.append(f"changes touched forbidden_paths: {forbidden}")
    return problems
