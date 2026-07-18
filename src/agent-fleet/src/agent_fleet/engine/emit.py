from __future__ import annotations

from pathlib import Path

from ..models.agent import AgentName


def write_agent(name: AgentName, source: str, directory: Path) -> Path:
    """Persist a rendered agent program as ``<directory>/<name>.py``.

    ``AgentName`` is a lowercase slug, so the filename is filesystem-safe by construction.
    Rewriting overwrites the previous render.

    Args:
        name: The agent name the file is keyed by.
        source: The rendered SDK program (`generate(spec)`).
        directory: The output directory; created (with parents) when missing.

    Returns:
        The path of the written program.
    """
    path = directory / f"{name}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path
