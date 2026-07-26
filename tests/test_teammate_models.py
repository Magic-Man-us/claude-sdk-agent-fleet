from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from agent_fleet.models.agent import (
    RUN_ERROR_MAX,
    RUN_OUTPUT_MAX,
    AgentKey,
    CodexModelId,
    CostUsd,
    Provider,
    RunError,
    RunOutput,
    TeammateRunStatus,
    TeammateTemplate,
    Toolkit,
    ToolkitName,
    teammate_key,
)

_TOOLKIT_NAME = TypeAdapter(ToolkitName)
_COST = TypeAdapter(CostUsd)
_OUTPUT = TypeAdapter(RunOutput)
_AGENT_KEY = TypeAdapter(AgentKey)
_RUN_ERROR = TypeAdapter(RunError)


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


def test_run_output_truncates_overlong_text() -> None:
    overlong = "x" * (RUN_OUTPUT_MAX + 500)
    truncated = _OUTPUT.validate_python(overlong)
    assert len(truncated) == RUN_OUTPUT_MAX
    assert truncated == "x" * RUN_OUTPUT_MAX


def test_run_error_truncates_overlong_text() -> None:
    overlong = "x" * (RUN_ERROR_MAX + 500)
    truncated = _RUN_ERROR.validate_python(overlong)
    assert len(truncated) == RUN_ERROR_MAX
    assert truncated == "x" * RUN_ERROR_MAX


def test_run_error_rejects_empty() -> None:
    with pytest.raises(ValidationError):
        _RUN_ERROR.validate_python("")


def test_teammate_key_is_a_valid_agent_key() -> None:
    key = teammate_key("reviewer")
    assert key == "teammate.reviewer"
    assert _AGENT_KEY.validate_python(key) == key


def test_template_defaults() -> None:
    template = TeammateTemplate(
        name="reviewer",
        brief="Review code changes for correctness and regressions.",
    )
    assert template.toolkits == []
    assert template.model.value == "inherit"
    assert template.provider is Provider.claude
    assert template.codex_model is None


def test_template_accepts_a_codex_provider_with_its_model() -> None:
    template = TeammateTemplate(
        name="reviewer",
        brief="Review code changes for correctness and regressions.",
        provider=Provider.codex,
        codex_model=CodexModelId.gpt_5_6_sol,
    )
    assert template.provider is Provider.codex
    assert template.codex_model is CodexModelId.gpt_5_6_sol


def test_template_rejects_codex_model_without_codex_provider() -> None:
    with pytest.raises(ValidationError, match="codex_model must be set"):
        TeammateTemplate(
            name="reviewer",
            brief="Review code changes for correctness and regressions.",
            codex_model=CodexModelId.gpt_5_6_sol,
        )


def test_template_rejects_codex_provider_without_a_codex_model() -> None:
    with pytest.raises(ValidationError, match="codex_model must be set"):
        TeammateTemplate(
            name="reviewer",
            brief="Review code changes for correctness and regressions.",
            provider=Provider.codex,
        )


def test_toolkit_holds_each_grant_kind_separately() -> None:
    kit = Toolkit(
        name="pydantic-review",
        skills=["pydantic-type-discipline"],
        mcp_servers=["dq"],
        tools=["Read"],
        agents=["researcher"],
    )
    assert kit.skills == ["pydantic-type-discipline"]
    assert kit.mcp_servers == ["dq"]
    assert kit.tools == ["Read"]
    assert kit.agents == ["researcher"]


def test_teammate_key_validates_the_name_it_is_given() -> None:
    """@validate_call, so AgentName's pattern fires on a plain call instead of minting a bad key."""
    assert teammate_key("reviewer") == "teammate.reviewer"
    with pytest.raises(ValidationError):
        teammate_key("Not A Name")


def test_toolkit_rejects_empty_entries() -> None:
    with pytest.raises(ValidationError):
        Toolkit(name="pydantic-review", entries=[])


def test_status_enum_members() -> None:
    assert {s.value for s in TeammateRunStatus} == {
        "unspawned",
        "idle",
        "running",
        "finished",
        "failed",
        "stale",
    }
