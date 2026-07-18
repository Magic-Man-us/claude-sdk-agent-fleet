from __future__ import annotations

from pathlib import Path

from agent_fleet import (
    AgentSpec,
    Catalog,
    InMemoryCatalogSource,
    ProblemRequest,
    assemble,
    generate,
)
from agent_fleet.engine.select import DEFAULT_TOOLS
from capdisc.hooks import HookConfig

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


def test_generate_wires_hooks_settings_file_when_directory_given(tmp_path: Path) -> None:
    spec = AgentSpec(
        name="auditor",
        description="Audits code for vulnerabilities.",
        system_prompt="You are auditor. Audit the code for vulnerabilities and stop now.",
        tools=["Read"],
        hooks=HookConfig.model_validate(
            {"PreToolUse": [{"hooks": [{"type": "command", "command": "./g.sh"}]}]}
        ),
    )
    rendered = generate(spec, tmp_path)
    path = tmp_path / "auditor.hooks.json"
    assert path.exists()
    assert f"settings={str(path)!r}" in rendered


def test_generate_omits_hooks_without_directory(tmp_path: Path) -> None:
    spec = AgentSpec(
        name="auditor",
        description="Audits code for vulnerabilities.",
        system_prompt="You are auditor. Audit the code for vulnerabilities and stop now.",
        tools=["Read"],
        hooks=HookConfig.model_validate(
            {"PreToolUse": [{"hooks": [{"type": "command", "command": "./g.sh"}]}]}
        ),
    )
    rendered = generate(spec)  # stdout-only preview: nowhere to write the sidecar
    assert "settings=" not in rendered
    assert list(tmp_path.iterdir()) == []


def test_assembly_is_deterministic(catalog: Catalog) -> None:
    source = InMemoryCatalogSource(catalog)
    request = ProblemRequest(task=_PROBLEM, name="changelog-curator")
    assert assemble(request, source) == assemble(request, source)
    assert generate(assemble(request, source).spec) == generate(assemble(request, source).spec)
