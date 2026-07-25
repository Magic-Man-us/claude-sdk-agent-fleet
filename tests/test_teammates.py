from __future__ import annotations

import pytest

from agent_fleet.engine.teammates import (
    ROSTER,
    _validate_roster,
    resolve_template,
    template_request,
)
from agent_fleet.models.agent import TeammateTemplate, Toolkit


def test_resolve_known_template() -> None:
    template = resolve_template(ROSTER[0].name)
    assert template is ROSTER[0]


def test_resolve_unknown_lists_roster() -> None:
    with pytest.raises(ValueError, match=ROSTER[0].name):
        resolve_template("nobody-here")


def test_template_request_maps_fields() -> None:
    template = ROSTER[0]
    request = template_request(template)
    assert request.task == template.brief
    assert request.name == template.name
    assert request.model == template.model


def test_template_request_flattens_toolkits() -> None:
    kits = (
        Toolkit(name="kit-a", entries=["skill-alpha", "tool-beta"]),
        Toolkit(name="kit-b", entries=["skill-gamma"]),
    )
    template = TeammateTemplate(
        name="specialist",
        brief="Exercise toolkit flattening in template resolution.",
        toolkits=["kit-a", "kit-b"],
    )
    request = template_request(template, toolkits=kits)
    assert request.pinned == ["skill-alpha", "tool-beta", "skill-gamma"]


def test_template_with_dangling_toolkit_raises() -> None:
    template = TeammateTemplate(
        name="specialist",
        brief="Exercise the dangling toolkit reference error.",
        toolkits=["kit-missing"],
    )
    with pytest.raises(ValueError, match="kit-missing"):
        template_request(template, toolkits=())


def test_template_request_carries_tags_and_system_prompt() -> None:
    template = TeammateTemplate(
        name="specialist",
        brief="Exercise tag and system-prompt propagation onto the ProblemRequest.",
        tags=["security", "documentation"],
        system_prompt=(
            "You are a specialist teammate focused on reviewing security-sensitive changes "
            "before they ship."
        ),
    )
    request = template_request(template)
    assert request.tags == template.tags
    assert request.system_prompt == template.system_prompt


def test_validate_roster_rejects_duplicate_names() -> None:
    duplicate = TeammateTemplate(
        name="researcher", brief="Duplicate the researcher name to exercise the guard."
    )
    with pytest.raises(ValueError, match="researcher"):
        _validate_roster((duplicate, duplicate), ())


def test_validate_roster_rejects_dangling_toolkit() -> None:
    template = TeammateTemplate(
        name="specialist",
        brief="Exercise the dangling toolkit reference guard at import time.",
        toolkits=["kit-missing"],
    )
    with pytest.raises(ValueError, match="kit-missing"):
        _validate_roster((template,), ())
