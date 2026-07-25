from __future__ import annotations

import pytest

from agent_fleet.engine.teammates import ROSTER, resolve_template, template_request
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
