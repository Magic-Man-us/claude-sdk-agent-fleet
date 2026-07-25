from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from agent_fleet.models.agent import CostUsd, RunOutput, ToolkitName

_TOOLKIT_NAME = TypeAdapter(ToolkitName)
_COST = TypeAdapter(CostUsd)
_OUTPUT = TypeAdapter(RunOutput)


def test_toolkit_name_accepts_slug_and_rejects_uppercase() -> None:
    assert _TOOLKIT_NAME.validate_python("pydantic-review") == "pydantic-review"
    with pytest.raises(ValidationError):
        _TOOLKIT_NAME.validate_python("Pydantic Review")


def test_cost_rejects_negative() -> None:
    assert _COST.validate_python(0.42) == 0.42
    with pytest.raises(ValidationError):
        _COST.validate_python(-0.01)


def test_run_output_accepts_text() -> None:
    assert _OUTPUT.validate_python("done") == "done"
