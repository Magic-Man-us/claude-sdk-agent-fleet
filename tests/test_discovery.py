from __future__ import annotations

from pathlib import Path

import pytest

from agent_fleet import BUILTIN_TOOLS, scan_environment, scan_skills
from capabilities_discovery.scope import ScopeRoots
from helpers import write_skill

_REAL_SKILLS = Path.home() / ".claude" / "skills"


def _roots(root: Path) -> ScopeRoots:
    (root / ".git").mkdir(exist_ok=True)  # bound the project walk-up at this dir
    return ScopeRoots.discover(start=root)


def test_scan_skills_parses_frontmatter(tmp_path: Path) -> None:
    write_skill(
        tmp_path,
        "changelog-curator",
        "---\nname: changelog-curator\n"
        "description: Curate changelog entries from a diff.\n---\n\nBody.",
    )
    cards = scan_skills(_roots(tmp_path))
    assert len(cards) == 1
    assert cards[0].ref == "changelog-curator"
    assert cards[0].id == "skill.changelog-curator"
    assert "changelog" in cards[0].description


def test_scan_skills_parses_tags(tmp_path: Path) -> None:
    write_skill(
        tmp_path,
        "audit",
        "---\nname: audit\ndescription: Audit code for vulnerabilities.\n"
        "tags: [audit, pentest]\n---\n",
    )
    card = scan_skills(_roots(tmp_path))[0]
    assert "audit" in card.tags
    assert "pentest" in card.tags


def test_scan_skills_untagged_skill_still_scans(tmp_path: Path) -> None:
    write_skill(tmp_path, "plain", "---\nname: plain\ndescription: A plain untagged skill.\n---\n")
    card = scan_skills(_roots(tmp_path))[0]
    assert card.tags == []


def test_scan_skills_falls_back_to_dir_name(tmp_path: Path) -> None:
    write_skill(tmp_path, "fallback", "---\ndescription: A skill without a name field.\n---\n")
    assert scan_skills(_roots(tmp_path))[0].ref == "fallback"


def test_scan_skills_skips_malformed(tmp_path: Path) -> None:
    (tmp_path / "no-skill-md").mkdir()
    write_skill(tmp_path, "no-frontmatter", "Just a body, no frontmatter.\n")
    write_skill(tmp_path, "no-description", "---\nname: no-description\n---\n")
    assert scan_skills(_roots(tmp_path)) == []


def test_scan_skills_missing_root_is_empty(tmp_path: Path) -> None:
    assert scan_skills(ScopeRoots.discover(start=tmp_path / "does-not-exist")) == []


def test_builtin_tools_carry_correct_flags() -> None:
    by_ref = {tool.ref: tool for tool in BUILTIN_TOOLS}
    assert by_ref["Read"].read_only is True
    assert by_ref["Write"].read_only is False
    assert by_ref["WebFetch"].needs_network is True


def test_scan_environment_indexes_skills_not_builtins(tmp_path: Path) -> None:
    write_skill(
        tmp_path, "demo", "---\nname: demo\ndescription: A demo skill for testing the scan.\n---\n"
    )
    catalog = scan_environment(_roots(tmp_path))
    refs = {entry.ref for entry in catalog.entries}
    assert "demo" in refs
    # built-in tools are provisioned at selection time, not recalled, so they are not in the catalog
    assert all(entry.kind != "tool" for entry in catalog.entries)


@pytest.mark.skipif(not _REAL_SKILLS.is_dir(), reason="no installed skills to scan")
def test_real_environment_scan_finds_skills() -> None:
    catalog = scan_environment(ScopeRoots.discover(start=Path.cwd(), home_dir=Path.home()))
    assert sum(entry.kind == "skill" for entry in catalog.entries) >= 1
