from __future__ import annotations

from agent_fleet import Catalog, InMemoryCatalogSource, ProblemRequest, assemble, generate
from agent_fleet.engine.select import DEFAULT_TOOLS

_PROBLEM = "Summarize the git commit history into grouped changelog entries"


def test_end_to_end_golden(catalog: Catalog) -> None:
    result = assemble(
        ProblemRequest(task=_PROBLEM, name="changelog-curator"), InMemoryCatalogSource(catalog)
    )
    assert result.selection.skills == ["changelog"]
    assert result.selection.tools == DEFAULT_TOOLS  # the fixed default grant
    assert result.spec.tools == DEFAULT_TOOLS
    assert result.spec.skills == ["changelog"]
    assert result.efficiency.passed

    rendered = generate(result.spec)
    compile(rendered, "<agent>", "exec")  # the SDK emitter produces runnable Python
    assert "ClaudeAgentOptions" in rendered
    assert f"allowed_tools={DEFAULT_TOOLS!r}" in rendered
    assert "skills=['changelog']" in rendered


def test_assembly_is_deterministic(catalog: Catalog) -> None:
    source = InMemoryCatalogSource(catalog)
    request = ProblemRequest(task=_PROBLEM, name="changelog-curator")
    assert assemble(request, source) == assemble(request, source)
    assert generate(assemble(request, source).spec) == generate(assemble(request, source).spec)
