from __future__ import annotations

from agent_fleet import Catalog, CatalogSkill, InMemoryCatalogSource, RecallQuery
from agent_fleet.engine.source import TwoStageRanker
from capabilities_discovery.catalog import DomainTag


def _skill(ref: str, description: str, domain: DomainTag, tags: list[str]) -> CatalogSkill:
    return CatalogSkill(
        id=f"skill.{ref}", ref=ref, description=description, domain=domain, tags=tags
    )


def _catalog() -> Catalog:
    return Catalog(
        entries=[
            _skill(
                "vuln-auditor",
                "Audit code for security vulnerabilities.",
                "security",
                ["audit"],
            ),
            _skill(
                "pentest-runner",
                "Run a penetration test against an endpoint.",
                "security",
                ["pentest"],
            ),
            _skill(
                "doc-writer",
                "Write and publish project documentation.",
                "documentation",
                ["docs"],
            ),
            _skill(
                "mermaid-maker",
                "Author mermaid diagrams for documentation.",
                "diagramming",
                ["mermaid"],
            ),
        ]
    )


def _query(text: str, domain: DomainTag | None) -> RecallQuery:
    return RecallQuery(text=text, domain=domain)


def test_prefilter_cuts_off_domain_entries() -> None:
    source = InMemoryCatalogSource(_catalog(), ranker=TwoStageRanker(prefilter_k=2))
    out = source.recall(_query("run a penetration test", domain="security"))
    refs = {candidate.entry.ref for candidate in out}
    assert refs == {"vuln-auditor", "pentest-runner"}  # off-domain skills cut by stage 1


def test_prefilter_orders_survivors_by_description() -> None:
    source = InMemoryCatalogSource(_catalog(), ranker=TwoStageRanker(prefilter_k=2))
    out = source.recall(_query("run a penetration test", domain="security"))
    assert out[0].entry.ref == "pentest-runner"  # description rerank breaks the domain tie


def test_untagged_query_falls_back_to_lexical() -> None:
    source = InMemoryCatalogSource(_catalog(), ranker=TwoStageRanker())
    out = source.recall(_query("write and publish project documentation", domain=None))
    assert out[0].entry.ref == "doc-writer"  # no domain → stage 1 skipped, pure description


def test_domain_query_keeps_untagged_skills_but_cuts_conflicting() -> None:
    catalog = Catalog(
        entries=[
            _skill("vuln-auditor", "Audit code for security vulnerabilities.", "security", []),
            CatalogSkill(
                id="skill.pattern-finder",
                ref="pattern-finder",
                description="Search code for a vulnerable pattern.",
            ),
            _skill("doc-writer", "Write project documentation.", "documentation", []),
        ]
    )
    source = InMemoryCatalogSource(catalog, ranker=TwoStageRanker())
    out = source.recall(_query("find a vulnerable pattern in code", domain="security"))
    refs = {candidate.entry.ref for candidate in out}
    assert "pattern-finder" in refs  # undeclared domain → agnostic, not cut
    assert "doc-writer" not in refs  # declares a conflicting domain → cut
