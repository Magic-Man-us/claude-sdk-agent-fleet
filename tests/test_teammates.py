"""The roster: where it is loaded from, what a file may say, and what a template expands into."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_fleet.engine.teammates import (
    DEFAULT_ROSTER,
    PROJECT_ROSTER_DIR,
    ROSTER_FILENAME,
    build_teammate,
    load_roster,
    resolve_template,
    roster_path,
)
from agent_fleet.models.agent import RosterFile, TeammateTemplate, Toolkit
from agent_fleet.settings import AgentFleetSettings

_ROSTER_TOML = """
[[toolkits]]
name = "python-review"
skills = ["pydantic-type-discipline", "code-reviewer"]
mcp_servers = ["dq"]
tools = ["Read", "Grep"]
agents = ["researcher"]

[[teammates]]
name = "researcher"
brief = "Research questions against the local code and documentation indexes."

[[teammates]]
name = "reviewer"
brief = "Review changes for correctness, regressions, and project conventions."
toolkits = ["python-review"]
model = "sonnet"
"""

_BRIEF = "Review changes for correctness and regressions in this repository."


def _write(path: Path, text: str = _ROSTER_TOML) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_explicit_setting_wins_over_a_project_file(tmp_path: Path) -> None:
    explicit = _write(tmp_path / "explicit.toml")
    _write(tmp_path / PROJECT_ROSTER_DIR / ROSTER_FILENAME)
    settings = AgentFleetSettings.model_validate({"AGENT_FLEET_ROSTER": explicit})

    assert roster_path(settings, start=tmp_path) == explicit


def test_an_explicit_path_is_returned_even_when_absent(tmp_path: Path) -> None:
    """A typo in AGENT_FLEET_ROSTER must surface, not fall through to a different roster."""
    missing = tmp_path / "typo.toml"
    settings = AgentFleetSettings.model_validate({"AGENT_FLEET_ROSTER": missing})

    assert roster_path(settings, start=tmp_path) == missing
    with pytest.raises(FileNotFoundError):
        load_roster(roster_path(settings, start=tmp_path))


def test_a_project_file_is_found_when_no_setting_is_given(tmp_path: Path) -> None:
    project = _write(tmp_path / PROJECT_ROSTER_DIR / ROSTER_FILENAME)
    assert roster_path(AgentFleetSettings(), start=tmp_path) == project


def test_no_file_falls_back_to_the_shipped_roster() -> None:
    assert load_roster(None) == DEFAULT_ROSTER
    assert [mate.name for mate in DEFAULT_ROSTER.teammates] == ["researcher", "reviewer"]


def test_loading_a_file_validates_every_entry(tmp_path: Path) -> None:
    roster = load_roster(_write(tmp_path / "roster.toml"))

    assert [mate.name for mate in roster.teammates] == ["researcher", "reviewer"]
    kit = roster.toolkits[0]
    assert kit.skills == ["pydantic-type-discipline", "code-reviewer"]
    assert kit.mcp_servers == ["dq"]
    assert kit.tools == ["Read", "Grep"]
    assert kit.agents == ["researcher"]


def test_malformed_toml_names_the_file(tmp_path: Path) -> None:
    path = _write(tmp_path / "roster.toml", "[[teammates]\nname =")
    with pytest.raises(ValueError, match=path.name):
        load_roster(path)


def test_a_teammate_referencing_an_unknown_toolkit_is_rejected() -> None:
    with pytest.raises(ValidationError, match="unknown toolkit"):
        RosterFile(
            teammates=[TeammateTemplate(name="reviewer", brief=_BRIEF, toolkits=["nonexistent"])]
        )


def test_duplicate_teammate_names_are_rejected() -> None:
    template = TeammateTemplate(name="reviewer", brief=_BRIEF)
    with pytest.raises(ValidationError, match="duplicate teammate names: reviewer"):
        RosterFile(teammates=[template, template])


def test_a_toolkit_naming_a_non_teammate_agent_is_rejected() -> None:
    with pytest.raises(ValidationError, match="not teammates"):
        RosterFile(
            toolkits=[Toolkit(name="kit", agents=["ghost"])],
            teammates=[TeammateTemplate(name="reviewer", brief=_BRIEF)],
        )


def test_a_toolkit_that_grants_nothing_is_rejected() -> None:
    with pytest.raises(ValidationError, match="grants nothing"):
        Toolkit(name="empty")


def test_build_routes_each_grant_kind_to_its_own_mechanism(tmp_path: Path) -> None:
    roster = load_roster(_write(tmp_path / "roster.toml"))
    build = build_teammate(resolve_template("reviewer", roster), roster)

    # skills and mcp servers become pinned catalog ids; tools and agents travel separately
    assert build.request.pinned == [
        "skill.pydantic-type-discipline",
        "skill.code-reviewer",
        "mcp.dq",
    ]
    assert build.request.tools == ["Read", "Grep"]
    assert build.agents == ["researcher"]
    assert build.request.name == "reviewer"


def test_the_template_model_is_used_when_no_override_is_set(tmp_path: Path) -> None:
    roster = load_roster(_write(tmp_path / "roster.toml"))
    assert build_teammate(resolve_template("reviewer", roster), roster).request.model == "sonnet"


def test_the_subagent_model_override_wins_over_the_template(tmp_path: Path) -> None:
    roster = load_roster(_write(tmp_path / "roster.toml"))
    build = build_teammate(resolve_template("reviewer", roster), roster, subagent_model="haiku")
    assert build.request.model == "haiku"


def test_inherit_is_treated_as_no_override(tmp_path: Path) -> None:
    """Claude Code reads CLAUDE_CODE_SUBAGENT_MODEL=inherit as unset; this mirrors that."""
    roster = load_roster(_write(tmp_path / "roster.toml"))
    build = build_teammate(resolve_template("reviewer", roster), roster, subagent_model="inherit")
    assert build.request.model == "sonnet"


def test_an_unrepresentable_override_is_reported_and_ignored(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A full model id cannot be represented here, but must not stop a teammate from starting."""
    roster = load_roster(_write(tmp_path / "roster.toml"))
    with caplog.at_level("WARNING"):
        build = build_teammate(
            resolve_template("reviewer", roster), roster, subagent_model="claude-sonnet-5"
        )

    assert build.request.model == "sonnet"  # the template's choice, unchanged
    assert "claude-sonnet-5" in caplog.text


def test_resolving_an_unknown_name_lists_the_roster(tmp_path: Path) -> None:
    roster = load_roster(_write(tmp_path / "roster.toml"))
    with pytest.raises(ValueError, match="researcher, reviewer"):
        resolve_template("nobody", roster)
