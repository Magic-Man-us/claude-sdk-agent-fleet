from __future__ import annotations

import re
from collections import Counter
from math import log
from typing import Protocol

from capdisc.base import FrozenModel
from capdisc.catalog import (
    DEFAULT_RECALL_LIMIT,
    Catalog,
    CatalogEntry,
    RecallLimit,
    RelevanceScore,
    Tag,
)

from ..models.agent import TaskBrief

_TOKEN = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "into",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "then",
        "this",
        "to",
        # domain fillers beyond standard English: catalog descriptions are full of
        # "use X to …" / "via the …", which would otherwise match nearly every entry
        "use",
        "using",
        "via",
        "with",
        "you",
        "your",
    }
)


def token_list(text: str) -> list[str]:
    """Tokenize text into lowercased words, dropping stopwords and keeping duplicates.

    Duplicates are kept so term frequencies survive for BM25 scoring.
    """
    return [tok for tok in _TOKEN.findall(text.lower()) if tok not in _STOPWORDS]


class RecallQuery(FrozenModel):
    """A task brief plus optional free `tags` routing, normalized into the query a
    `CatalogSource` ranks against."""

    text: TaskBrief
    tags: list[Tag] = []
    limit: RecallLimit = DEFAULT_RECALL_LIMIT


class Candidate(FrozenModel):
    """One catalog entry paired with its relevance score for a query — the ranked unit
    `recall()` returns for `select()` to threshold and pick from."""

    entry: CatalogEntry
    score: RelevanceScore


class CatalogSource(Protocol):
    """Ranks a catalog against a `RecallQuery`. The retrieval swap seam: an in-memory
    stand-in today, a real backend later — callers depend only on `recall()`."""

    def recall(self, query: RecallQuery) -> list[Candidate]: ...


class Ranker(Protocol):
    """Scores each entry against a query — the swap point between lexical (keyword
    overlap) and semantic (meaning-based) matching. Returns scored, unsorted candidates."""

    def rank(self, query: RecallQuery, entries: list[CatalogEntry]) -> list[Candidate]: ...


def _entry_text(entry: CatalogEntry) -> str:
    return entry.search_text


BM25_K1 = 1.5
BM25_B = 0.75


def bm25_normalized(
    query_terms: list[str], docs: list[list[str]], k1: float = BM25_K1, b: float = BM25_B
) -> list[float]:
    """Okapi BM25 of each tokenized doc against the query terms, normalized to [0,1].

    Args:
        query_terms: Already-tokenized, deduplicated query terms (sorted for stable summation).
        docs: One token list per document, in caller order.
        k1: BM25 term-frequency saturation.
        b: BM25 length-normalization strength.

    Returns:
        One score per doc (same order), divided by the top raw score — best match 1.0, non-match
        0.0. All-zero when there are no query terms or no docs.
    """
    if not query_terms or not docs:
        return [0.0 for _ in docs]
    freqs = [Counter(doc) for doc in docs]
    lengths = [sum(freq.values()) for freq in freqs]
    avgdl = sum(lengths) / len(lengths) or 1.0
    n = len(docs)
    df = {term: sum(term in freq for freq in freqs) for term in query_terms}
    idf = {term: log(1 + (n - df[term] + 0.5) / (df[term] + 0.5)) for term in query_terms}
    raw: list[float] = []
    for freq, length in zip(freqs, lengths, strict=True):
        norm = k1 * (1 - b + b * length / avgdl)
        raw.append(
            sum(
                idf[term] * freq[term] * (k1 + 1) / (freq[term] + norm)
                for term in query_terms
                if term in freq
            )
        )
    top = max(raw)
    return [value / top if top > 0.0 else 0.0 for value in raw]


class BM25Ranker:
    """Okapi BM25 over each entry's text. Rare query terms dominate (IDF) and length
    normalization keeps a verbose description from winning on size alone — sharper than raw
    token overlap, still deterministic and offline. Raw scores are normalized to [0,1] by the
    top score in the candidate set: the best match is 1.0, a non-match 0.0."""

    def __init__(self, k1: float = BM25_K1, b: float = BM25_B) -> None:
        self._k1 = k1
        self._b = b

    def rank(self, query: RecallQuery, entries: list[CatalogEntry]) -> list[Candidate]:
        """Score each entry by BM25 against the query terms.

        Args:
            query: The recall query; only its `text` is scored here.
            entries: The candidate entries to score, in caller order.

        Returns:
            One candidate per entry (same order), scored in [0,1] by the top raw score — best
            match 1.0, non-match 0.0. All-zero when the query has no terms or there are no entries.
        """
        terms = sorted(set(token_list(query.text)))  # sorted → sum order is hash-seed-stable
        docs = [token_list(_entry_text(entry)) for entry in entries]
        scores = bm25_normalized(terms, docs, self._k1, self._b)
        return [
            Candidate(entry=entry, score=score)
            for entry, score in zip(entries, scores, strict=True)
        ]


DEFAULT_PREFILTER_K = 15


def _tag_idf(entries: list[CatalogEntry]) -> dict[str, float]:
    """Inverse document frequency per tag across the candidate set.

    A tag in nearly every entry carries ~0 weight, a rare tag carries high weight.
    """
    n = len(entries) or 1
    df = Counter(tag for entry in entries for tag in entry.tags)
    return {tag: log(n / count) for tag, count in df.items()}


def _prefilter_score(query: RecallQuery, entry: CatalogEntry, idf: dict[str, float]) -> float:
    """Stage-1 sort key for one entry: its IDF-weighted tag overlap with the query, higher first."""
    return sum(idf.get(tag, 0.0) for tag in set(query.tags) & set(entry.tags))


class TwoStageRanker:
    """Two-stage retrieval. Stage 1 narrows the candidate set: a tags query ranks the tagged
    entries by IDF-weighted tag overlap and keeps the top-K (tuned for recall, untagged entries
    pass through). Stage 2 reranks the survivors by description, so the final pick is by meaning,
    not tags alone. A query with no tags skips stage 1 (pure BM25)."""

    def __init__(
        self, prefilter_k: int = DEFAULT_PREFILTER_K, rerank: Ranker | None = None
    ) -> None:
        self._k = prefilter_k
        self._rerank = rerank or BM25Ranker()

    def rank(self, query: RecallQuery, entries: list[CatalogEntry]) -> list[Candidate]:
        """Narrow by tags, then rerank the survivors by description.

        Args:
            query: The recall query; its `tags` drive stage 1, its `text` drives stage 2.
            entries: The candidate entries to rank.

        Returns:
            Scored candidates from the rerank ranker. A query with no tags skips stage 1 and
            reranks every entry (pure BM25).
        """
        if not query.tags:
            return self._rerank.rank(query, entries)
        idf = _tag_idf(entries)
        routed = [entry for entry in entries if entry.tags]
        passthrough = [entry for entry in entries if not entry.tags]
        routed.sort(key=lambda entry: _prefilter_score(query, entry, idf), reverse=True)
        return self._rerank.rank(query, routed[: self._k] + passthrough)


class InMemoryCatalogSource:
    """A `CatalogSource` over one in-memory catalog, scored by a pluggable `Ranker`
    (BM25 by default). Ranks every entry and trims to the query limit."""

    def __init__(self, catalog: Catalog, ranker: Ranker | None = None) -> None:
        self._catalog = catalog
        self._ranker = ranker or BM25Ranker()

    def _top_k(self, matches: list[Candidate], limit: RecallLimit) -> list[Candidate]:
        """The `limit` highest-scoring candidates, ties broken by entry id for a stable order."""
        matches.sort(key=lambda candidate: (-candidate.score, candidate.entry.id))
        return matches[:limit]

    def recall(self, query: RecallQuery) -> list[Candidate]:
        """Rank the catalog against a query.

        Favors recall (find every plausibly relevant entry) over precision — the threshold in
        `select` trims later.

        Args:
            query: The recall query, carrying the text and optional tag routing.

        Returns:
            Up to `query.limit` candidates, ranked highest score first.
        """
        ranked = self._ranker.rank(query, list(self._catalog.entries))
        return self._top_k(ranked, query.limit)
