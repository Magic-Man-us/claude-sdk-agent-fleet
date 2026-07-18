from __future__ import annotations

from agent_fleet import ProblemRequest, SelectedCapabilities
from agent_fleet.engine.compose import compose


def test_compose_maps_refs_and_identity() -> None:
    selection = SelectedCapabilities(
        skills=("changelog",),
        tools=("Bash(git log:*)",),
    )
    request = ProblemRequest(
        task="Summarize the git history into changelog entries", name="changelog-curator"
    )
    spec = compose(request, selection)
    assert spec.name == "changelog-curator"
    assert spec.tools == ["Bash(git log:*)"]
    assert spec.skills == ["changelog"]
    assert "Summarize the git history" in spec.system_prompt
    assert "Bash(git log:*)" in spec.system_prompt
    assert 8 <= len(spec.description) <= 1536


def test_caller_prompt_overrides_template() -> None:
    prompt = "You are a very specific agent. Follow these exact instructions precisely and stop."
    request = ProblemRequest(
        task="Do the specific thing with the file", name="doer", system_prompt=prompt
    )
    spec = compose(request, SelectedCapabilities(tools=("Read",)))
    assert spec.system_prompt == prompt


def test_compose_carries_domain_and_tags() -> None:
    request = ProblemRequest(
        task="Audit the codebase for vulnerabilities",
        name="auditor",
        domain="security",
        tags=["audit", "pentest"],
    )
    spec = compose(request, SelectedCapabilities())
    assert spec.domain == "security"
    assert spec.tags == ["audit", "pentest"]


def test_compose_summarizes_overlong_task() -> None:
    # a task far longer than AgentDescription's cap must not break compose; the description
    # is normalized down to a short summary instead of raising ValidationError
    long_task = "Summarize the repository changes for the release notes. " * 100
    spec = compose(ProblemRequest(task=long_task, name="summarizer"), SelectedCapabilities())
    assert len(spec.description) <= 200
    assert spec.description.startswith("Summarize the repository changes")
