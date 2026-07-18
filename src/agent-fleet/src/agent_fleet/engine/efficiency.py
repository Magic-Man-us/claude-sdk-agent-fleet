from __future__ import annotations

from enum import StrEnum

from pydantic import computed_field

from capdisc.base import FrozenModel

from ..models.agent import (
    DEFAULT_SKILL_BUDGET,
    DEFAULT_TOOL_BUDGET,
    PROMPT_MAX,
    PROMPT_MIN,
    AgentSpec,
    SkillBudget,
    ToolBudget,
)


class EfficiencyDimension(StrEnum):
    """The axes a spec is scored on — tool count, skill count, and prompt size."""

    tool_count = "tool_count"
    skill_count = "skill_count"
    prompt_size = "prompt_size"


class DimensionResult(FrozenModel):
    """The pass/fail outcome for one efficiency dimension, with a human-readable `detail`."""

    dimension: EfficiencyDimension
    passed: bool
    detail: str


class EfficiencyConfig(FrozenModel):
    """The budgets a spec is scored against: tool/skill counts and the prompt-size band."""

    tool_budget: ToolBudget = DEFAULT_TOOL_BUDGET
    skill_budget: SkillBudget = DEFAULT_SKILL_BUDGET
    prompt_min: int = PROMPT_MIN
    prompt_max: int = PROMPT_MAX


class EfficiencyReport(FrozenModel):
    """Per-dimension pass/fail for one assembled spec; `passed` holds only when every
    dimension lands within its budget."""

    results: list[DimensionResult]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        """True only when every dimension passed."""
        return all(result.passed for result in self.results)


def score(spec: AgentSpec, config: EfficiencyConfig | None = None) -> EfficiencyReport:
    """Score an assembled spec on each efficiency dimension.

    Args:
        spec: The spec to score, by its tool count, skill count, and prompt length.
        config: The budgets to score against; the defaults when None. Note an empty tool grant
            fails `tool_count` — it inherits every tool, which is not minimal.

    Returns:
        The per-dimension pass/fail report.
    """
    cfg = config or EfficiencyConfig()
    n_tools = len(spec.tools)
    n_skills = len(spec.skills)
    n_prompt = len(spec.system_prompt)
    tool_detail = (
        "0 tools — an empty grant inherits every tool (not minimal)"
        if n_tools == 0
        else f"{n_tools}/{cfg.tool_budget} tools"
    )
    results = [
        DimensionResult(
            dimension=EfficiencyDimension.tool_count,
            passed=1 <= n_tools <= cfg.tool_budget,
            detail=tool_detail,
        ),
        DimensionResult(
            dimension=EfficiencyDimension.skill_count,
            passed=n_skills <= cfg.skill_budget,
            detail=f"{n_skills}/{cfg.skill_budget} skills",
        ),
        DimensionResult(
            dimension=EfficiencyDimension.prompt_size,
            passed=cfg.prompt_min <= n_prompt <= cfg.prompt_max,
            detail=f"{n_prompt} chars",
        ),
    ]
    return EfficiencyReport(results=results)
