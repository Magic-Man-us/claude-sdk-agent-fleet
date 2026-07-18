from __future__ import annotations

from pathlib import Path

from agent_fleet import AgentSpec, generate, write_agent

_SPEC = AgentSpec(
    name="auditor",
    description="Audits code for vulnerabilities.",
    system_prompt="You are auditor. Audit the code for vulnerabilities and stop now.",
    tools=("Read", "Grep"),
)
_SOURCE = generate(_SPEC)


def test_write_agent_persists_the_rendered_program(tmp_path: Path) -> None:
    path = write_agent(_SPEC.name, _SOURCE, tmp_path)
    assert path == tmp_path / "auditor.py"
    assert path.read_text(encoding="utf-8") == _SOURCE


def test_write_agent_creates_missing_parents(tmp_path: Path) -> None:
    path = write_agent(_SPEC.name, _SOURCE, tmp_path / "nested" / "agents")
    assert path.exists()


def test_write_agent_overwrites_a_previous_render(tmp_path: Path) -> None:
    stale = tmp_path / "auditor.py"
    stale.write_text("outdated", encoding="utf-8")
    assert write_agent(_SPEC.name, _SOURCE, tmp_path).read_text(encoding="utf-8") == _SOURCE
