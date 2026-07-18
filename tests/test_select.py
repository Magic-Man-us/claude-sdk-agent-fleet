from __future__ import annotations

from agent_fleet import (
    Catalog,
    InMemoryCatalogSource,
    ProblemRequest,
    RecallQuery,
    select,
)
from agent_fleet.engine.select import DEFAULT_TOOLS


def _recall(catalog: Catalog, text: str) -> list:
    return InMemoryCatalogSource(catalog).recall(RecallQuery(text=text))


def test_select_keeps_relevant_drops_irrelevant(catalog: Catalog) -> None:
    text = "Summarize the git commit history into grouped changelog entries"
    selection = select(_recall(catalog, text), ProblemRequest(task=text, name="changelog-curator"))
    assert selection.skills == ["changelog"]  # the irrelevant error-handling skill is dropped


def test_every_agent_gets_the_fixed_default_toolset(catalog: Catalog) -> None:
    text = "Summarize the git commit history into grouped changelog entries"
    selection = select(_recall(catalog, text), ProblemRequest(task=text, name="changelog-curator"))
    # tools are provisioned, not recalled — never empty, regardless of the task's wording
    assert selection.tools == DEFAULT_TOOLS


def test_pinned_bypasses_threshold(catalog: Catalog) -> None:
    text = "Summarize the git commit history into grouped changelog entries"
    request = ProblemRequest(task=text, name="changelog-curator", pinned=("skill.error_handling",))
    selection = select(_recall(catalog, text), request)
    assert "error-handling" in selection.skills
