from __future__ import annotations

from agent_fleet import Catalog, InMemoryCatalogSource, RecallQuery


def _query(text: str) -> RecallQuery:
    return RecallQuery(text=text)


def test_recall_ranks_relevant_entries_first(catalog: Catalog) -> None:
    source = InMemoryCatalogSource(catalog)
    out = source.recall(_query("Summarize the git commit history into grouped changelog entries"))
    assert out[0].entry.id == "skill.changelog"  # the on-topic skill ranks first
    assert out[0].score > 0.0


def test_scores_are_bounded(catalog: Catalog) -> None:
    out = InMemoryCatalogSource(catalog).recall(_query("git commit history changelog"))
    assert all(0.0 <= c.score <= 1.0 for c in out)


def test_recall_is_deterministic(catalog: Catalog) -> None:
    source = InMemoryCatalogSource(catalog)
    query = _query("git commit history changelog")
    assert source.recall(query) == source.recall(query)
