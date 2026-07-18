from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import Field

from capdisc.base import FrozenModel


class ThinkingDisplay(StrEnum):
    """Whether the model's thinking text is returned to the caller. On Opus 4.7+ the default is
    `omitted` (the `ThinkingBlock` arrives with empty text); `summarized` returns a readable
    summary of the reasoning. Visibility only — thinking runs and is billed the same either way,
    and the raw chain of thought is never exposed on any model."""

    summarized = "summarized"
    omitted = "omitted"


ThinkingBudget = Annotated[
    int,
    Field(
        ge=1024,
        title="Thinking budget",
        description="Fixed token budget for the model's thinking, for the `enabled` variant.",
        examples=[8000],
    ),
]


class AdaptiveThinking(FrozenModel):
    """Claude decides when and how much to think — the recommended setting on current models.
    Pair with `display=summarized` to surface a summary of the reasoning."""

    type: Literal["adaptive"] = "adaptive"
    display: ThinkingDisplay | None = None


class EnabledThinking(FrozenModel):
    """Think with a fixed token budget. Opus 4.7+ models reject `budget_tokens` and accept only
    the `adaptive` variant; this is kept to mirror the full SDK `ThinkingConfig` surface."""

    type: Literal["enabled"] = "enabled"
    budget_tokens: ThinkingBudget
    display: ThinkingDisplay | None = None


class DisabledThinking(FrozenModel):
    """No extended thinking."""

    type: Literal["disabled"] = "disabled"


ThinkingConfig = Annotated[
    AdaptiveThinking | EnabledThinking | DisabledThinking,
    Field(
        discriminator="type",
        title="Thinking config",
        description="How the generated agent reasons, mirroring the SDK `ThinkingConfig` union; "
        "set `display=summarized` to capture the reasoning summary in `ThinkingBlock` output.",
    ),
]
