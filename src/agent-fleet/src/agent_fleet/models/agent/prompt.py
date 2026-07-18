from __future__ import annotations

from pydantic import computed_field

from capabilities_discovery.base import FrozenModel
from capabilities_discovery.catalog import SkillRef, ToolRef

from .types import AgentName, PromptBody, SectionTag, TaskBrief

_EMPTY = "none"
_ROLE = "You are {name}, a focused subagent with a single job."
_TASK_HEADING = "## Task"
_TOOLS_HEADING = "## Tools granted"
_SKILLS_HEADING = "## Skills loaded"
_OPERATING_RULES = (
    "## Operating rules\n"
    "- Do only the task above, using only the tools granted.\n"
    "- Report the result and stop; do not expand scope.\n"
)


def _listed(refs: list[str]) -> str:
    """The refs as a comma-separated line, or `"none"` when empty."""
    return ", ".join(refs) or _EMPTY


class PromptSection(FrozenModel):
    """One `<tag> … </tag>` block of a system prompt."""

    tag: SectionTag
    body: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def text(self) -> str:
        """The block rendered as `<tag> body </tag>`."""
        return f"<{self.tag}> {self.body} </{self.tag}>"


class TemplatedPrompt(FrozenModel):
    """The default system prompt for a focused subagent, assembled from its tagged
    sections: the role line, the task, the granted tools and loaded skills, and the
    fixed operating rules. Loads from its fields and dumps the rendered `body`."""

    name: AgentName
    task: TaskBrief
    tools: list[ToolRef] = []
    skills: list[SkillRef] = []

    @property
    def sections(self) -> list[PromptSection]:
        """The prompt's tagged sections in order: role, task, tools, skills, operating rules."""
        return [
            PromptSection(tag=SectionTag.role, body=_ROLE.format(name=self.name)),
            PromptSection(tag=SectionTag.problem, body=f"{_TASK_HEADING}\n{self.task}"),
            PromptSection(tag=SectionTag.tools, body=f"{_TOOLS_HEADING}\n{_listed(self.tools)}"),
            PromptSection(tag=SectionTag.skills, body=f"{_SKILLS_HEADING}\n{_listed(self.skills)}"),
            PromptSection(tag=SectionTag.instructions, body=_OPERATING_RULES),
        ]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def body(self) -> PromptBody:
        """The full system prompt: every section's text joined by blank lines."""
        return "\n\n".join(section.text for section in self.sections)
