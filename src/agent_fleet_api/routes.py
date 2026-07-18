from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response, status

from agent_fleet.engine.dispatch import prepare_run, run_with_capture
from agent_fleet.engine.emit import write_agent
from agent_fleet.engine.orchestrate import OrchestrateOutcome, collect_orchestration
from agent_fleet.engine.pipeline import AssemblyResult, generate
from agent_fleet.models.agent import (
    AgentKey,
    AgentRunRecord,
    AgentSpec,
    Finding,
    PoolEntry,
    ProblemRequest,
    RunId,
    RunOutcome,
    RunRecord,
    TaskBrief,
)
from capdisc.catalog import DEFAULT_RECALL_LIMIT, Catalog, RecallLimit
from capdisc.report import EnvironmentReport

from .deps import (
    AuthDep,
    CapabilityRouterDep,
    CoreSettingsDep,
    EngineDep,
    PoolDep,
    ReportDep,
)
from .models import Health, OrchestrateRequest, PoolRunRequest, RenderedAgent

router = APIRouter()


@router.get("/healthz")
def healthz() -> Health:
    """Liveness probe — returns the static ok health payload. Unauthenticated by design."""
    return Health()


@router.get("/catalog", dependencies=[AuthDep])
def get_catalog(engine: EngineDep) -> Catalog:
    """The live capability corpus the pipeline draws from — drives the form's pin/tag options."""
    return engine.catalog


@router.get("/report", dependencies=[AuthDep])
def get_report_route(report: ReportDep) -> EnvironmentReport:
    """The discovery harvest captured at startup — scan roots, on-disk inventory, skills, builtin
    tools, plugins (with per-plugin component token cost), and MCP servers."""
    return report


@router.post("/generate", dependencies=[AuthDep])
def generate_agent(request: ProblemRequest, engine: EngineDep) -> AssemblyResult:
    """Assemble a spec for a problem statement: recall, select, compose, score."""
    return engine.assemble(request)


@router.post("/render", dependencies=[AuthDep])
def render_agent(spec: AgentSpec, core: CoreSettingsDep) -> RenderedAgent:
    """Render an assembled spec into a runnable Claude Agent SDK program, persisting it to the
    configured ``agent_dir`` when one is set."""
    source = generate(spec, core.agent_dir)
    written = write_agent(spec.name, source, core.agent_dir) if core.agent_dir is not None else None
    return RenderedAgent(source=source, path=written)


@router.post("/orchestrate", dependencies=[AuthDep])
async def orchestrate(request: OrchestrateRequest, caps: CapabilityRouterDep) -> OrchestrateOutcome:
    """Run the capability orchestrator to completion (review -> compose -> spawn) and return its
    outcome. Requires the claude CLI at runtime."""
    return await collect_orchestration(request.task, caps)


@router.post("/pool/{agent_key}", dependencies=[AuthDep])
async def save_pool_entry(
    agent_key: AgentKey,
    request: ProblemRequest,
    engine: EngineDep,
    pool: PoolDep,
    reset_session: bool = False,
) -> PoolEntry:
    """Assemble an agent for the problem and store it in the pool under `agent_key`, returning the
    stored entry. `reset_session=True` mints a fresh session UUID even when the id exists."""
    return await pool.create_agent(agent_key, request, engine.source, reset_session=reset_session)


@router.get("/pool/find", dependencies=[AuthDep])
async def find_pool_entries(
    query: TaskBrief, pool: PoolDep, limit: RecallLimit = DEFAULT_RECALL_LIMIT
) -> list[PoolEntry]:
    """Rediscover pooled entries by re-describing the problem, ranked most-relevant first."""
    return await pool.find(query, limit)


@router.get("/pool", dependencies=[AuthDep])
async def list_pool_entries(pool: PoolDep) -> list[PoolEntry]:
    """Every pooled entry, most-recently-updated first."""
    return await pool.list()


@router.get("/pool/{agent_key}", dependencies=[AuthDep])
async def get_pool_entry(agent_key: AgentKey, pool: PoolDep) -> PoolEntry:
    """The pooled entry stored under `agent_key`; 404 when the pool holds no such entry."""
    entry = await pool.get_by_key(agent_key)
    if entry is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"pool entry not found: {agent_key}"
        )
    return entry


@router.delete("/pool/{agent_key}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[AuthDep])
async def delete_pool_entry(agent_key: AgentKey, pool: PoolDep) -> Response:
    """Remove the entry stored under `agent_key`; 204 on removal, 404 when the id was absent."""
    if not await pool.delete(agent_key):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"pool entry not found: {agent_key}"
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/pool/{agent_key}/runs", dependencies=[AuthDep])
async def list_pool_runs(agent_key: AgentKey, pool: PoolDep) -> list[RunRecord]:
    """Every run recorded for `agent_key`, most-recently-started first."""
    return await pool.list_runs(agent_key)


@router.get("/pool/{agent_key}/runs/{run_id}/agents", dependencies=[AuthDep])
async def list_pool_run_agents(
    agent_key: AgentKey,  # noqa: ARG001 — path-scoping only; the run id alone identifies the agents
    run_id: RunId,
    pool: PoolDep,
) -> list[AgentRunRecord]:
    """Every agent (main plus dispatched subagents) that ran within `run_id`, in recorded order —
    this is how a caller discovers the independently-resumable `session_id` of each one."""
    return await pool.list_agent_runs(run_id)


@router.get("/pool/{agent_key}/findings", dependencies=[AuthDep])
async def list_pool_findings(
    agent_key: AgentKey, pool: PoolDep, run_id: RunId | None = None
) -> list[Finding]:
    """The pooled agent's findings, oldest-first; `run_id` narrows to findings recorded within
    that run."""
    return await pool.list_findings(agent_key, run_id=run_id)


@router.post("/pool/{agent_key}/run", dependencies=[AuthDep])
async def run_pool_entry(
    agent_key: AgentKey, request: PoolRunRequest, pool: PoolDep, caps: CapabilityRouterDep
) -> RunOutcome:
    """Run the pooled agent live and capture every agent's session id. The entry's first-ever run
    starts a fresh session; later runs resume it. Named subagents are resolved from their pool
    entries and wired in via `with_subagents`. Every run is also granted the run-scoped
    findings-writer (`with_findings_tool`) and dynamic capability acquisition
    (`with_acquire_tool`), matching an MCP-triggered run. `request.resume_agent_id`, when set,
    continues one specific previously-dispatched subagent via `with_agent_resume` and
    `build_resume_prompt`. Requires the claude CLI at runtime."""
    try:
        run, options, prompt = prepare_run(
            pool.pool,
            caps,
            agent_key,
            request.task,
            subagent_agent_keys=request.subagent_agent_keys,
            resume_agent_id=request.resume_agent_id,
        )
    except KeyError as exc:
        missing = exc.args[0]
        if missing == agent_key:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"pool entry not found: {agent_key}"
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"subagent pool entry not found: {missing}",
        ) from exc
    try:
        return await run_with_capture(
            pool.pool, agent_key, request.task, options, run=run, prompt=prompt
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"pool entry vanished during run: {agent_key}",
        ) from exc
