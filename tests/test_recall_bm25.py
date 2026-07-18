from __future__ import annotations

import pytest

from agent_fleet import Catalog, CatalogSkill, InMemoryCatalogSource, RecallQuery


def _skill(ref: str, description: str) -> CatalogSkill:
    return CatalogSkill(id=f"skill.{ref}", ref=ref, description=description)


def _query(text: str) -> RecallQuery:
    return RecallQuery(text=text)


def _catalog() -> Catalog:
    return Catalog(
        entries=[
            _skill("security-auditor", "Audit code for security vulnerabilities."),
            _skill("blog-writer", "Write blog posts and run an SEO audit before publishing."),
            _skill("doc-writer", "Write and publish project documentation."),
            _skill("k8s-deployer", "Deploy and scale services on a Kubernetes cluster."),
            _skill("ci-runner", "Run continuous integration builds and deploy artifacts."),
            _skill("changelog", "Curate changelog entries grouped from a commit diff."),
        ]
    )


# (query, the skill that must rank first). Each distractor shares a common term (audit / deploy)
# but lacks the rare ones — the "are the ones we want showing up?" quality lock.
CASES = [
    ("audit the code for security vulnerabilities", "security-auditor"),
    ("deploy a service to a kubernetes cluster", "k8s-deployer"),
    ("write and publish the project documentation", "doc-writer"),
]


@pytest.mark.parametrize("text,expected", CASES)
def test_bm25_ranks_focused_skill_first(text: str, expected: str) -> None:
    ranked = [c.entry.ref for c in InMemoryCatalogSource(_catalog()).recall(_query(text))]
    assert ranked[0] == expected, f"{expected!r} not first for {text!r}: {ranked[:3]}"


def test_bm25_prefers_rare_term_over_high_frequency_common_term() -> None:
    # 'deploy' is common across the corpus (low IDF); 'kubernetes' is rare (high IDF). Plain
    # overlap would tie — each candidate matches one query term — so this only passes because
    # BM25 weights the rare hit above the repeated common one.
    catalog = Catalog(
        entries=[
            _skill("k8s-guide", "Kubernetes basics and concepts."),
            _skill("deployer", "Deploy deploy deploy the build artifacts."),
            _skill("deploy-web", "Deploy the web service."),
            _skill("deploy-db", "Deploy the database service."),
            _skill("deploy-worker", "Deploy the worker service."),
        ]
    )
    source = InMemoryCatalogSource(catalog)
    ranked = [c.entry.ref for c in source.recall(_query("deploy kubernetes"))]
    assert ranked[0] == "k8s-guide"


def test_bm25_scores_are_normalized_to_unit_interval() -> None:
    source = InMemoryCatalogSource(_catalog())
    out = source.recall(_query("audit code for security vulnerabilities"))
    assert out[0].score == 1.0  # the top match is normalized to exactly 1.0
    assert all(0.0 <= c.score <= 1.0 for c in out)
