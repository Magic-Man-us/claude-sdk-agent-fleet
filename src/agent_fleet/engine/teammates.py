from __future__ import annotations

import logging
import tomllib
from pathlib import Path

from pydantic import validate_call

from capdisc.catalog import CatalogEntryId, SkillRef, ToolRef, catalog_id

from ..models.agent import (
    AgentName,
    ModelId,
    ProblemRequest,
    RosterFile,
    TeammateBuild,
    TeammateTemplate,
    Toolkit,
    ToolkitName,
)
from ..settings import AgentFleetSettings

logger = logging.getLogger(__name__)

ROSTER_FILENAME = "roster.toml"
#: Per-project roster directory, resolved against the working directory.
PROJECT_ROSTER_DIR = ".agent-fleet"
#: User-wide roster directory.
USER_ROSTER_DIR = Path.home() / ".config" / "agent-fleet"

#: The roster used when no file is configured. Deliberately minimal: it exists so the tools work
#: out of the box, not as a place to keep a personal crew — this module ships inside an installed
#: package, so an edit here is undone by the next install. Write a roster file instead.
DEFAULT_ROSTER = RosterFile(
    teammates=[
        TeammateTemplate(
            name="researcher",
            brief=(
                "Research questions against the local code and documentation indexes and report "
                "cited findings."
            ),
        ),
        TeammateTemplate(
            name="reviewer",
            brief=(
                "Review code changes for correctness, regressions, and adherence to project "
                "conventions."
            ),
        ),
    ],
)


def roster_path(
    settings: AgentFleetSettings | None = None, *, start: Path | None = None
) -> Path | None:
    """The roster file to load, or None when none is configured anywhere.

    Resolution runs most-specific first: an explicit `AGENT_FLEET_ROSTER`, then this project's
    `.agent-fleet/roster.toml`, then the user's `~/.config/agent-fleet/roster.toml`. An explicit
    path is returned even when it does not exist, so a typo in the variable surfaces as a named
    error instead of silently falling through to a different roster.

    Args:
        settings: Settings supplying the explicit override; read from the environment when
            omitted.
        start: Directory to look for a project roster in; the working directory when omitted.

    Returns:
        The first roster file found, or None to use the shipped default.
    """
    settings = settings if settings is not None else AgentFleetSettings()
    if settings.roster is not None:
        return settings.roster.expanduser()
    project = (start if start is not None else Path.cwd()) / PROJECT_ROSTER_DIR / ROSTER_FILENAME
    if project.is_file():
        return project
    user = USER_ROSTER_DIR / ROSTER_FILENAME
    return user if user.is_file() else None


def load_roster(path: Path | None) -> RosterFile:
    """Read and validate the roster at `path`, or return the shipped default when it is None.

    The parsed TOML is handed straight to `RosterFile`, so every name, ref, and model in the file
    is checked at load time — a typo is a named error here rather than a confusing failure later.

    Args:
        path: The roster file to read, or None for the shipped default.

    Returns:
        The validated roster.

    Raises:
        FileNotFoundError: When `path` is given but absent; an explicit path that does not exist
            is a configuration error, not a reason to fall back.
        ValueError: When the file is not valid TOML, named with its path.
    """
    if path is None:
        return DEFAULT_ROSTER
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"roster file {path} is not valid TOML: {exc}") from exc
    return RosterFile.model_validate(raw)


def current_roster(settings: AgentFleetSettings | None = None) -> RosterFile:
    """The roster in force: the configured file when there is one, else the shipped default."""
    return load_roster(roster_path(settings))


@validate_call
def resolve_template(name: AgentName, roster: RosterFile) -> TeammateTemplate:
    """The roster template addressed by `name`.

    Decorated so `AgentName`'s pattern actually fires: this is the roster gate every teammate tool
    calls first, and a bare annotation would let an unvalidated name through to a pool lookup.

    Raises:
        ValidationError: When `name` is not a well-formed agent name.
        ValueError: When `name` is not on the roster; the message lists the valid names.
    """
    for template in roster.teammates:
        if template.name == name:
            return template
    known = ", ".join(sorted(mate.name for mate in roster.teammates)) or "<empty roster>"
    raise ValueError(f"unknown teammate {name!r} (roster: {known})")


def _resolved_model(template: TeammateTemplate, override: str | None) -> ModelId:
    """The model a teammate runs on: the subagent-model override when usable, else the template's.

    Claude Code puts `CLAUDE_CODE_SUBAGENT_MODEL` at the top of its resolution order and reads
    `inherit` as "no override"; this mirrors that. A value this package cannot represent (a full
    model id rather than an alias) is reported and ignored rather than raised on, since an
    unrelated environment variable must not stop a teammate from starting.
    """
    if override is None or override == ModelId.inherit:
        return template.model
    try:
        return ModelId(override)
    except ValueError:
        logger.warning(
            "ignoring CLAUDE_CODE_SUBAGENT_MODEL=%r: not one of %s",
            override,
            ", ".join(sorted(member.value for member in ModelId)),
        )
        return template.model


def _toolkits_for(template: TeammateTemplate, roster: RosterFile) -> list[Toolkit]:
    """The toolkit objects a template names, in the order it names them.

    Raises:
        ValueError: When the template names a toolkit the roster does not define.
    """
    by_name: dict[ToolkitName, Toolkit] = {kit.name: kit for kit in roster.toolkits}
    missing = sorted(set(template.toolkits) - set(by_name))
    if missing:
        raise ValueError(
            f"template {template.name!r} references unknown toolkit(s): {', '.join(missing)}"
        )
    return [by_name[name] for name in template.toolkits]


def build_teammate(
    template: TeammateTemplate,
    roster: RosterFile,
    *,
    subagent_model: str | None = None,
) -> TeammateBuild:
    """Expand a template and its toolkits into everything needed to stand the teammate up.

    The grant kinds travel separately because different machinery applies them: skills and MCP
    servers become pinned catalog ids that survive the relevance filter, tools ride on the request
    and are granted outright, and agents come back for the caller to wire as dispatchable
    subagents at run time.

    Args:
        template: The roster template to expand.
        roster: The roster its toolkits resolve against.
        subagent_model: A `CLAUDE_CODE_SUBAGENT_MODEL` value to apply over the template's model.

    Returns:
        The request the generation pipeline consumes, plus the subagent names to wire in.

    Raises:
        ValueError: When the template names a toolkit the roster does not define.
    """
    kits = _toolkits_for(template, roster)
    pinned: list[CatalogEntryId] = []
    tools: list[ToolRef] = []
    agents: list[AgentName] = []
    skills: list[SkillRef] = []
    for kit in kits:
        pinned.extend(catalog_id("skill", ref) for ref in kit.skills)
        pinned.extend(catalog_id("mcp", ref) for ref in kit.mcp_servers)
        skills.extend(ref for ref in kit.skills if ref not in skills)
        tools.extend(ref for ref in kit.tools if ref not in tools)
        agents.extend(name for name in kit.agents if name not in agents)
    # Naming skills is an exact answer; naming none leaves relevance selection to do its job —
    # auto-wiring a researcher with doc-search is the point of the capability router, not waste.
    # Pin skills only when you want exactly those (which also skips loading the settings tree).
    declared_skills = skills or None
    request = ProblemRequest(
        task=template.brief,
        name=template.name,
        tags=template.tags,
        model=_resolved_model(template, subagent_model),
        pinned=pinned,
        tools=tools,
        skills=declared_skills,
        system_prompt=template.system_prompt,
    )
    return TeammateBuild(request=request, agents=agents)
