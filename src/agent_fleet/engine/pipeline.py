from __future__ import annotations

from pathlib import Path

from capdisc.base import FrozenModel

from ..models.agent import AgentSpec, ProblemRequest
from .compose import compose
from .efficiency import EfficiencyConfig, EfficiencyReport, score
from .render import render_claude_sdk, to_options, with_hooks
from .select import SelectedCapabilities, select
from .source import CatalogSource, RecallQuery


class AssemblyResult(FrozenModel):
    """Everything `assemble()` produces for one request: the AgentSpec, the capabilities
    chosen to build it, and the efficiency report scoring the result."""

    spec: AgentSpec
    selection: SelectedCapabilities
    efficiency: EfficiencyReport


def _query(request: ProblemRequest) -> RecallQuery:
    """The `RecallQuery` for a request: its task text plus optional tag routing."""
    return RecallQuery(text=request.task, tags=request.tags)


def assemble(
    request: ProblemRequest,
    source: CatalogSource,
    config: EfficiencyConfig | None = None,
) -> AssemblyResult:
    """Run the full assembly pipeline: recall → select → compose → score.

    Args:
        request: The problem request driving recall and selection.
        source: The catalog source to recall candidates from.
        config: Efficiency-scoring config; the scorer's default when None.

    Returns:
        The spec, the capabilities chosen to build it, and its efficiency report.
    """
    candidates = source.recall(_query(request))
    selection = select(candidates, request)
    spec = compose(request, selection)
    report = score(spec, config)
    return AssemblyResult(spec=spec, selection=selection, efficiency=report)


def generate(spec: AgentSpec, directory: Path | None = None) -> str:
    """Render an assembled spec into the source of a runnable Claude Agent SDK program.

    When `directory` is given and the spec declares hooks, a `<name>.hooks.json` settings file is
    written there (alongside the emitted program) and the emitted options load it via `settings=`.
    Without a directory — a stdout-only preview with nowhere persistent to write the sidecar — hooks
    are not wired.

    Args:
        spec: The assembled spec to render.
        directory: Where the program (and its hooks sidecar) is persisted; typically `agent_dir`.
    """
    options = to_options(spec)
    if directory is not None:
        options = with_hooks(options, spec, directory)
    return render_claude_sdk(spec, options)
