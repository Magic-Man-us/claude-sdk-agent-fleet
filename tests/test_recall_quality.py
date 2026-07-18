from __future__ import annotations

import pytest

from agent_fleet import BUILTIN_TOOLS, Catalog, InMemoryCatalogSource, RecallQuery

# (problem statement, the builtin tool we expect to surface near the top of recall).
# This is the "are the ones we want showing up?" check — when a case fails, the tool's
# keyword tags are too thin for that phrasing and need enriching.
CASES = [
    ("read the project configuration file", "Read"),
    ("generate a report file and save it to disk", "Write"),
    ("edit a specific line in an existing source file", "Edit"),
    ("find all files matching a glob pattern", "Glob"),
    ("search the codebase for where a function is defined", "Grep"),
    ("run a shell command to build the project", "Bash"),
    ("fetch a web page over the network and read it", "WebFetch"),
]


def _ranked_refs(problem: str) -> list[str]:
    catalog = Catalog(entries=list(BUILTIN_TOOLS))
    query = RecallQuery(text=problem)
    return [candidate.entry.ref for candidate in InMemoryCatalogSource(catalog).recall(query)]


@pytest.mark.parametrize("problem,expected", CASES)
def test_expected_tool_in_top_three(problem: str, expected: str) -> None:
    ranked = _ranked_refs(problem)
    assert expected in ranked[:3], f"{expected!r} not in top-3 {ranked[:3]} for {problem!r}"
