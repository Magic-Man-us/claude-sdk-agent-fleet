"""Working-tree change detection: what a run actually touched, beyond what git alone reports.

`git diff --name-only` cannot see a deletion of a gitignored/forbidden file, a symlink swapped
for a regular file (or vice versa) at the same path, or a bare mode-bit change with unchanged
content. `forbidden_state`/`fingerprint` snapshot the forbidden-path subtree before and after a
run so `actual_changes` catches those too, unioned with git's own diff.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path, PurePosixPath

from ..models.agent import ScopePattern


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    executable = shutil.which("git")
    if executable is None:
        raise RuntimeError("git is required for change detection")
    return subprocess.run(  # noqa: S603
        [executable, "-C", str(root), *arguments],
        check=False,
        capture_output=True,
    )


def canonical_root(root: Path) -> Path:
    """Resolve `root` to its git worktree's top-level directory.

    `forbidden_state`'s `root.glob(pattern)` treats `patterns` as repo-root-relative, so callers
    that pass a subdirectory need the true root for forbidden-path fingerprinting to glob the
    right subtree. Unlike the skill this was ported from — which treats any mismatch between the
    given `root` and the canonical top-level as fatal because its `cwd` comes from untrusted
    request JSON — this does not reject a mismatch: agent-fleet's callers pass internal state
    (e.g. `PoolEntry.cwd`), not untrusted input, so a subdirectory is resolved rather than
    refused. Callers should use the returned path for subsequent calls in the sequence.

    Raises:
        RuntimeError: `root` is not inside a git repository.
    """
    probe = _git(root, "rev-parse", "--show-toplevel")
    if probe.returncode != 0:
        raise RuntimeError(f"{root} is not a git repository")
    return Path(os.fsdecode(probe.stdout).strip()).resolve()


def git_changed_paths(root: Path) -> set[str]:
    """Repo-relative paths git reports as changed: tracked diffs against HEAD plus untracked files.

    The baseline is `HEAD`, not the index, so both staged and unstaged changes are caught.
    """
    diff = _git(root, "diff", "--name-only", "-z", "HEAD", "--")
    untracked = _git(root, "ls-files", "--others", "--exclude-standard", "-z")
    if diff.returncode != 0 or untracked.returncode != 0:
        details = os.fsdecode(diff.stderr or untracked.stderr)[:500]
        raise RuntimeError(f"unable to inspect changed paths: {details}")
    return {
        os.fsdecode(item)
        for output in (diff.stdout, untracked.stdout)
        for item in output.split(b"\0")
        if item
    }


def fingerprint(path: Path) -> str:
    """A content+mode digest for `path`: mode bits, then a type-tagged payload.

    The payload is the symlink target for a symlink, the file bytes (streamed) for a regular
    file, or nothing for anything else — tagged by type so a symlink and a regular file that
    happen to hash to the same bytes never collide.
    """
    stat = path.lstat()
    digest = hashlib.sha256(f"{stat.st_mode:o}\0".encode())
    if path.is_symlink():
        digest.update(b"symlink\0")
        digest.update(os.fsencode(path.readlink()))
    elif path.is_file():
        digest.update(b"file\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    else:
        digest.update(b"other")
    return digest.hexdigest()


def forbidden_state(root: Path, patterns: list[ScopePattern]) -> dict[str, str]:
    """Snapshot `{repo-relative path: fingerprint}` for every file matching a forbidden pattern.

    Taken before and after a run; the diff of the two snapshots is where `actual_changes` finds
    deletions, symlink swaps, and mode-only changes that `git_changed_paths` cannot see. `.git` is
    excluded twice over: any pattern naming it is skipped outright (so `root.glob` never walks
    `.git` internals), and any matched candidate under it is skipped too, for patterns like
    `.git/**` or `**` that could still reach it. Only files are fingerprinted; directories are
    skipped, matching what `fingerprint` supports.
    """
    state: dict[str, str] = {}
    for pattern in patterns:
        if ".git" in PurePosixPath(pattern).parts:
            continue
        for candidate in root.glob(pattern):
            relative = candidate.relative_to(root)
            if ".git" in relative.parts or candidate.is_dir():
                continue
            state[relative.as_posix()] = fingerprint(candidate)
    return state


def actual_changes(
    root: Path,
    forbidden_paths: list[ScopePattern],
    forbidden_before: dict[str, str],
) -> list[str]:
    """Every repo-relative path actually changed since `forbidden_before` was snapshotted.

    Unions `git_changed_paths(root)` with every path whose forbidden-pattern fingerprint differs
    between `forbidden_before` and a fresh `forbidden_state` snapshot. Comparing via `.get()`
    (which returns None for an absent key) means a deletion (present before, absent after) and a
    forbidden-pattern creation (absent before, present after) both trip the inequality with no
    special-case code — the reason `fingerprint`/`forbidden_state` exist alongside git's own diff.
    """
    changed = git_changed_paths(root)
    forbidden_after = forbidden_state(root, forbidden_paths)
    changed.update(
        path
        for path in set(forbidden_before) | set(forbidden_after)
        if forbidden_before.get(path) != forbidden_after.get(path)
    )
    return sorted(changed)
