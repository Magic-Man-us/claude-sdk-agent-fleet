from __future__ import annotations

import re

from pydantic import validate_call

from ..models.agent import AgentName, TaskBrief

_SLUG_WORD = re.compile(r"[a-z0-9]+")
_MAX_SLUG_WORDS = 6
_MAX_SLUG_LEN = 64  # AgentName's pattern caps the slug at 64 characters
_FALLBACK_SLUG = "agent"


@validate_call(validate_return=True)
def slugify_name(task: TaskBrief) -> AgentName:
    """Derive a display-name slug from the first several words of a task brief.

    Lowercases, keeps the leading alphanumeric words, joins them with dashes, and caps the result
    at `AgentName`'s length — so the return always satisfies the `AgentName` pattern (enforced by
    `validate_return`). The slug is a label, not a key, so it need not be unique.

    Args:
        task: The task brief to name after.

    Returns:
        An `AgentName`-valid slug, or a fallback when the task carries no alphanumeric words.
    """
    words = _SLUG_WORD.findall(task.lower())[:_MAX_SLUG_WORDS]
    slug = "-".join(words)[:_MAX_SLUG_LEN].rstrip("-")
    return slug or _FALLBACK_SLUG
