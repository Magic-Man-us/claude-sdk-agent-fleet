from __future__ import annotations

from collections.abc import Sequence

from capdisc.catalog import CatalogEntryId

from ..models.agent import (
    AgentName,
    ProblemRequest,
    TeammateTemplate,
    Toolkit,
    ToolkitName,
)

TOOLKITS: tuple[Toolkit, ...] = ()

ROSTER: tuple[TeammateTemplate, ...] = (
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
)

_TEMPLATE_BY_NAME: dict[AgentName, TeammateTemplate] = {t.name: t for t in ROSTER}
if len(_TEMPLATE_BY_NAME) != len(ROSTER):
    raise ValueError("duplicate teammate names in ROSTER")


def resolve_template(name: AgentName) -> TeammateTemplate:
    """The roster template addressed by `name`.

    Raises:
        ValueError: When `name` is not on the roster; the message lists the valid names.
    """
    template = _TEMPLATE_BY_NAME.get(name)
    if template is None:
        known = ", ".join(sorted(_TEMPLATE_BY_NAME))
        raise ValueError(f"unknown teammate {name!r} (roster: {known})")
    return template


def template_request(
    template: TeammateTemplate, toolkits: Sequence[Toolkit] = TOOLKITS
) -> ProblemRequest:
    """Translate a template into the `ProblemRequest` the existing create path consumes,
    flattening its toolkits into pinned capability ids.

    Raises:
        ValueError: When the template references a toolkit name absent from `toolkits`.
    """
    by_name: dict[ToolkitName, Toolkit] = {kit.name: kit for kit in toolkits}
    pinned: list[CatalogEntryId] = []
    for kit_name in template.toolkits:
        kit = by_name.get(kit_name)
        if kit is None:
            raise ValueError(f"template {template.name!r} references unknown toolkit {kit_name!r}")
        pinned.extend(kit.entries)
    return ProblemRequest(
        task=template.brief,
        name=template.name,
        tags=template.tags,
        model=template.model,
        pinned=pinned,
        system_prompt=template.system_prompt,
    )
