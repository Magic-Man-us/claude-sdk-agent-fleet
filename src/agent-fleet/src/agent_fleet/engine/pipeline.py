from __future__ import annotations

from capabilities_discovery.base import FrozenModel

from ..models.agent import AgentSpec, ProblemRequest
from .compose import compose
from .efficiency import EfficiencyConfig, EfficiencyReport, score
from .render import render_claude_sdk
from .select import SelectedCapabilities, select
from .source import CatalogSource, RecallQuery


class AssemblyResult(FrozenModel):
    """Everything `assemble()` produces for one request: the AgentSpec, the capabilities
    chosen to build it, and the efficiency report scoring the result."""

    spec: AgentSpec
    selection: SelectedCapabilities
    efficiency: EfficiencyReport


def _query(request: ProblemRequest) -> RecallQuery:
    """The `RecallQuery` for a request: its task text plus optional domain/tag routing."""
    return RecallQuery(text=request.task, domain=request.domain, tags=request.tags)


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


def generate(spec: AgentSpec) -> str:
    """Render an assembled spec into the source of a runnable Claude Agent SDK program."""
    return render_claude_sdk(spec)
