from __future__ import annotations

from pathlib import Path


def touch(path: Path, text: str = "x") -> None:
    """Create `path` and any missing parents, writing `text`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_skill(root: Path, name: str, body: str) -> None:
    """Write `<root>/.claude/skills/<name>/SKILL.md` with `body`."""
    touch(root / ".claude" / "skills" / name / "SKILL.md", body)


def standalone_set(claude_dir: Path) -> None:
    """One 'foo' of each standalone kind (skill, agent, command) under a `.claude` directory."""
    touch(claude_dir / "skills" / "foo" / "SKILL.md")
    touch(claude_dir / "agents" / "foo.md")
    touch(claude_dir / "commands" / "foo.md")
