from __future__ import annotations

from agent_fleet import Catalog, CatalogSkill, InMemoryCatalogSource, RecallQuery
from agent_fleet.engine.source import TwoStageRanker


def _skill(ref: str, description: str, tags: list[str]) -> CatalogSkill:
    return CatalogSkill(id=f"skill.{ref}", ref=ref, description=description, tags=tags)


def _catalog() -> Catalog:
    return Catalog(
        entries=[
            _skill(
                "vuln-auditor",
                "Audit code for security vulnerabilities.",
                ["security", "audit"],
            ),
            _skill(
                "pentest-runner",
                "Run a penetration test against an endpoint.",
                ["security", "pentest"],
            ),
            _skill(
                "doc-writer",
                "Write and publish project documentation.",
                ["documentation", "docs"],
            ),
            _skill(
                "mermaid-maker",
                "Author mermaid diagrams for documentation.",
                ["diagramming", "mermaid"],
            ),
        ]
    )


def _query(text: str, tags: list[str]) -> RecallQuery:
    return RecallQuery(text=text, tags=tags)


def test_prefilter_keeps_top_k_tag_matches() -> None:
    source = InMemoryCatalogSource(_catalog(), ranker=TwoStageRanker(prefilter_k=2))
    out = source.recall(_query("run a penetration test", ["security"]))
    refs = {candidate.entry.ref for candidate in out}
    assert refs == {"vuln-auditor", "pentest-runner"}  # non-overlapping tags cut by stage 1


def test_prefilter_orders_survivors_by_description() -> None:
    source = InMemoryCatalogSource(_catalog(), ranker=TwoStageRanker(prefilter_k=2))
    out = source.recall(_query("run a penetration test", ["security"]))
    assert out[0].entry.ref == "pentest-runner"  # description rerank breaks the tag tie


def test_untagged_query_falls_back_to_lexical() -> None:
    source = InMemoryCatalogSource(_catalog(), ranker=TwoStageRanker())
    out = source.recall(_query("write and publish project documentation", []))
    assert out[0].entry.ref == "doc-writer"  # no tags → stage 1 skipped, pure description


def test_tag_query_keeps_untagged_skills_but_cuts_low_overlap() -> None:
    catalog = Catalog(
        entries=[
            _skill("vuln-auditor", "Audit code for security vulnerabilities.", ["security"]),
            CatalogSkill(
                id="skill.pattern-finder",
                ref="pattern-finder",
                description="Search code for a vulnerable pattern.",
            ),
            _skill("doc-writer", "Write project documentation.", ["documentation"]),
        ]
    )
    source = InMemoryCatalogSource(catalog, ranker=TwoStageRanker(prefilter_k=1))
    out = source.recall(_query("find a vulnerable pattern in code", ["security"]))
    refs = {candidate.entry.ref for candidate in out}
    assert "pattern-finder" in refs  # untagged → passthrough, never cut
    assert "doc-writer" not in refs  # non-overlapping tag, past top-K → cut
