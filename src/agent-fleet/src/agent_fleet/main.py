from __future__ import annotations

import sys
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from capabilities_discovery.discovery import scan_environment

from .engine.emit import write_agent
from .engine.pipeline import assemble, generate
from .engine.source import InMemoryCatalogSource
from .models.agent import AgentName, ProblemRequest, TaskBrief
from .settings import AgentFleetSettings, current_discovery_scope

_DEFAULT_AGENT_NAME: AgentName = "generated-agent"
_TASK_ADAPTER: TypeAdapter[TaskBrief] = TypeAdapter(TaskBrief)
_MODULE_RUN = "python -m agent_fleet.main"
# argv[0] for a `python -m` run is the module file path (…/main.py), not the console-script name
_MODULE_ARGV0 = frozenset({"main.py", "__main__.py", "-c", ""})


def _prog() -> str:
    """The program name for usage/error text: the console-script name when invoked as one (argv[0]
    is the script, e.g. `agent-fleet`), or the `python -m` form for a module run."""
    name = Path(sys.argv[0]).name
    return _MODULE_RUN if name in _MODULE_ARGV0 else name


def _task_from_cli() -> TaskBrief:
    """The task brief from the CLI args, falling back to stdin.

    Returns:
        The task text, validated against the `TaskBrief` constraints.

    Raises:
        SystemExit: With code 2 when no task is supplied, or when it fails `TaskBrief`
            validation — the boundary translation of `ValidationError` into a clean CLI error.
    """
    raw = " ".join(sys.argv[1:]).strip() or sys.stdin.read().strip()
    if not raw:
        sys.stderr.write(f"usage: {_prog()} <task>\n")
        raise SystemExit(2)
    try:
        return _TASK_ADAPTER.validate_python(raw)
    except ValidationError as exc:
        reasons = "; ".join(error["msg"] for error in exc.errors())
        sys.stderr.write(f"{_prog()}: invalid task: {reasons}\n")
        raise SystemExit(2) from exc


def main() -> None:
    """Assemble an agent for the CLI task and write the generated SDK program to stdout,
    also persisting it to ``agent_dir`` when one is configured."""
    request = ProblemRequest(task=_task_from_cli(), name=_DEFAULT_AGENT_NAME)
    core = AgentFleetSettings()
    roots = current_discovery_scope().roots()
    catalog = scan_environment(roots)
    result = assemble(request, InMemoryCatalogSource(catalog))
    source = generate(result.spec, core.agent_dir)
    sys.stdout.write(source)
    if core.agent_dir is not None:
        sys.stderr.write(f"wrote {write_agent(result.spec.name, source, core.agent_dir)}\n")


if __name__ == "__main__":
    main()
