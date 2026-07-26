"""Isolation and change detection: what a run may touch, proven against real git repositories."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agent_fleet.engine.worktree import (
    SECRET_ENV_NAMES,
    actual_changes,
    canonical_root,
    fingerprint,
    forbidden_state,
    git_changed_paths,
    require_isolated_worktree,
    scrub_secret_env,
)


def _run_git(root: Path, *arguments: str) -> None:
    subprocess.run(["git", "-C", str(root), *arguments], check=True, capture_output=True)  # noqa: S603, S607


def _init_repo(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _run_git(root, "init", "-q")
    _run_git(root, "config", "user.email", "test@example.com")
    _run_git(root, "config", "user.name", "Test")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    _init_repo(root)
    (root / "tracked.txt").write_text("original\n")
    (root / "secret.env").write_text("secret\n")
    _run_git(root, "add", "-A")
    _run_git(root, "commit", "-q", "-m", "initial")
    return root


def test_canonical_root_resolves_a_subdirectory_to_the_worktree_top(repo: Path) -> None:
    subdir = repo / "sub"
    subdir.mkdir()
    assert canonical_root(subdir) == repo.resolve()


def test_canonical_root_rejects_a_non_git_directory(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    with pytest.raises(RuntimeError, match="not a git repository"):
        canonical_root(plain)


def test_git_changed_paths_reports_a_tracked_modification(repo: Path) -> None:
    (repo / "tracked.txt").write_text("modified\n")
    assert git_changed_paths(repo) == {"tracked.txt"}


def test_git_changed_paths_reports_an_untracked_file(repo: Path) -> None:
    (repo / "new.txt").write_text("new\n")
    assert git_changed_paths(repo) == {"new.txt"}


def test_git_changed_paths_is_empty_on_a_clean_tree(repo: Path) -> None:
    assert git_changed_paths(repo) == set()


def test_fingerprint_differs_for_different_content(repo: Path) -> None:
    path = repo / "tracked.txt"
    before = fingerprint(path)
    path.write_text("different\n")
    assert fingerprint(path) != before


def test_forbidden_state_snapshots_matching_files(repo: Path) -> None:
    state = forbidden_state(repo, ["secret.env"])
    assert set(state) == {"secret.env"}


def test_forbidden_state_skips_git_internals(repo: Path) -> None:
    assert forbidden_state(repo, [".git/config"]) == {}
    assert all(not path.startswith(".git/") for path in forbidden_state(repo, ["**"]))


def test_actual_changes_reports_a_tracked_modification(repo: Path) -> None:
    before = forbidden_state(repo, ["secret.env"])
    (repo / "tracked.txt").write_text("modified\n")
    assert actual_changes(repo, ["secret.env"], before) == ["tracked.txt"]


def test_actual_changes_catches_a_deletion_git_diff_alone_misses(repo: Path) -> None:
    """A deletion of a gitignored/untracked forbidden file never shows in `git diff --name-only`;
    only the fingerprint snapshot diff catches it."""
    ignored = repo / "ignored.env"
    ignored.write_text("secret\n")
    (repo / ".gitignore").write_text("ignored.env\n")
    _run_git(repo, "add", ".gitignore")
    _run_git(repo, "commit", "-q", "-m", "ignore secret")
    before = forbidden_state(repo, ["ignored.env"])
    assert git_changed_paths(repo) == set()

    ignored.unlink()

    assert git_changed_paths(repo) == set()
    assert actual_changes(repo, ["ignored.env"], before) == ["ignored.env"]


def test_actual_changes_catches_a_symlink_swap_git_diff_alone_misses(repo: Path) -> None:
    """An ignored/untracked forbidden file swapped for a symlink: invisible to `git diff
    --name-only` either way, caught only by the fingerprint's type-tagged digest."""
    ignored = repo / "ignored.env"
    ignored.write_text("secret\n")
    target = repo / "target.txt"
    target.write_text("secret\n")
    (repo / ".gitignore").write_text("ignored.env\ntarget.txt\n")
    _run_git(repo, "add", ".gitignore")
    _run_git(repo, "commit", "-q", "-m", "ignore secret and target")
    before = forbidden_state(repo, ["ignored.env"])

    ignored.unlink()
    ignored.symlink_to(target)

    assert git_changed_paths(repo) == set()
    assert actual_changes(repo, ["ignored.env"], before) == ["ignored.env"]


def test_actual_changes_is_empty_on_a_clean_tree(repo: Path) -> None:
    before = forbidden_state(repo, ["secret.env"])
    assert actual_changes(repo, ["secret.env"], before) == []


@pytest.fixture
def linked_worktree(repo: Path, tmp_path: Path) -> Path:
    """A clean, detached-HEAD linked worktree off `repo` — the shape a run is meant to run in."""
    worktree = tmp_path / "linked"
    _run_git(repo, "worktree", "add", "--detach", "-q", str(worktree))
    return worktree


def test_require_isolated_worktree_accepts_a_clean_detached_linked_worktree(
    linked_worktree: Path,
) -> None:
    require_isolated_worktree(linked_worktree)


def test_require_isolated_worktree_rejects_a_dirty_tree(linked_worktree: Path) -> None:
    (linked_worktree / "tracked.txt").write_text("dirty\n")
    with pytest.raises(ValueError, match="clean"):
        require_isolated_worktree(linked_worktree)


def test_require_isolated_worktree_rejects_a_non_detached_head(repo: Path, tmp_path: Path) -> None:
    worktree = tmp_path / "branched"
    _run_git(repo, "branch", "other")
    _run_git(repo, "worktree", "add", "-q", str(worktree), "other")
    with pytest.raises(ValueError, match="detached-HEAD"):
        require_isolated_worktree(worktree)


def test_require_isolated_worktree_rejects_the_main_worktree(repo: Path) -> None:
    _run_git(repo, "checkout", "-q", "--detach", "HEAD")
    with pytest.raises(ValueError, match="linked disposable worktree"):
        require_isolated_worktree(repo)


def test_scrub_secret_env_removes_every_secret_name() -> None:
    env = dict.fromkeys(SECRET_ENV_NAMES, "leaked") | {"PATH": "/usr/bin"}
    scrubbed = scrub_secret_env(env)
    assert not SECRET_ENV_NAMES & scrubbed.keys()
    assert scrubbed == {"PATH": "/usr/bin"}
