"""Run scope: what a scope may declare, and whether a run stayed inside it."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_fleet.engine.scope import matches, overlaps, violations
from agent_fleet.models.agent import RunMode, RunScope


def test_a_glob_free_pattern_covers_everything_beneath_it() -> None:
    """PurePosixPath.match reads "src" as nothing at all; scope has to mean the subtree."""
    assert matches("src/agent_fleet/pool.py", ["src"])
    assert matches("src", ["src"])
    assert not matches("srcs/other.py", ["src"])


def test_single_star_stays_inside_one_segment() -> None:
    assert matches("src/pool.py", ["src/*.py"])
    assert not matches("src/engine/pool.py", ["src/*.py"])


def test_double_star_spans_any_number_of_segments() -> None:
    assert matches("src/pool.py", ["src/**/*.py"])
    assert matches("src/a/b/c/pool.py", ["src/**/*.py"])


def test_whole_repo_covers_everything() -> None:
    assert matches("anything/at/all.py", ["."])


def test_overlap_is_detected_in_both_directions() -> None:
    assert overlaps(["src"], ["src/secret.py"])
    assert overlaps(["src/secret.py"], ["src"])
    assert not overlaps(["src"], ["docs"])


def test_whole_repo_does_not_count_as_overlap() -> None:
    """Read-everything with carve-outs is the point of the forbidden list, not a contradiction."""
    assert not overlaps(["."], ["secrets.env"])


def test_a_compliant_run_reports_nothing() -> None:
    scope = RunScope(mode=RunMode.write, allowed_paths=["src/**/*.py"])
    assert violations(scope, ["src/a/b.py"]) == []
    assert violations(scope, []) == []


def test_changes_outside_and_inside_forbidden_are_both_reported() -> None:
    scope = RunScope(
        mode=RunMode.write, allowed_paths=["src/**/*.py"], forbidden_paths=["src/secret.py"]
    )
    outside = violations(scope, ["docs/readme.md"])
    assert outside == ["changes outside allowed_paths: ['docs/readme.md']"]

    forbidden = violations(scope, ["src/secret.py"])
    assert any("forbidden_paths" in problem for problem in forbidden)


def test_a_read_run_that_changed_anything_has_failed() -> None:
    scope = RunScope(mode=RunMode.read, allowed_paths=["."])
    assert "read-only run changed files" in violations(scope, ["src/a.py"])[0]


def test_escaping_patterns_are_rejected() -> None:
    for pattern in ("/etc/passwd", "../outside"):
        with pytest.raises(ValidationError, match="escapes the repository"):
            RunScope(allowed_paths=[pattern])


def test_protected_content_cannot_be_granted() -> None:
    for pattern in (".git/config", "src/.venv", "secrets/.env.production"):
        with pytest.raises(ValidationError, match="protected content"):
            RunScope(allowed_paths=[pattern])


def test_protected_content_may_be_forbidden() -> None:
    """Naming .git in forbidden_paths is the entire point of that list."""
    assert RunScope(forbidden_paths=[".git", ".env"]).forbidden_paths == [".git", ".env"]


def test_a_write_run_cannot_take_the_whole_repository() -> None:
    with pytest.raises(ValidationError, match="whole repository"):
        RunScope(mode=RunMode.write, allowed_paths=["."])
    assert RunScope(mode=RunMode.read, allowed_paths=["."]).allowed_paths == ["."]
