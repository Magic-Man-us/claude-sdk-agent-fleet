from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from agent_fleet.engine.pipeline import AssemblyResult, assemble
from agent_fleet.engine.pool import AsyncAgentPool
from agent_fleet.engine.source import CatalogSource, InMemoryCatalogSource
from agent_fleet.models.agent import ProblemRequest
from agent_fleet.router.capability import CapabilityRouter
from agent_fleet.settings import AgentFleetSettings
from capdisc.catalog import Catalog
from capdisc.report import EnvironmentReport

from .settings import ApiSettings


class Engine:
    """Holds the live catalog and its source, and runs the core pipeline. Built once at
    startup so each request reuses one scanned catalog rather than rescanning the environment."""

    def __init__(self, catalog: Catalog) -> None:
        """Wrap a scanned catalog in an in-memory source for the pipeline to recall against."""
        self._catalog = catalog
        self._source: CatalogSource = InMemoryCatalogSource(catalog)

    @property
    def catalog(self) -> Catalog:
        """The live capability catalog this engine serves."""
        return self._catalog

    @property
    def source(self) -> CatalogSource:
        """The catalog source this engine recalls against."""
        return self._source

    def assemble(self, request: ProblemRequest) -> AssemblyResult:
        """Run the assembly pipeline for a request against the held catalog source."""
        return assemble(request, self._source)


def get_engine(request: Request) -> Engine:
    """The `Engine` built at startup, for use as a FastAPI dependency.

    Args:
        request: The incoming request, carrying the app whose `state.engine` was set in lifespan.

    Returns:
        The process-wide engine.
    """
    engine = request.app.state.engine
    assert isinstance(engine, Engine)
    return engine


EngineDep = Annotated[Engine, Depends(get_engine)]


def get_report(request: Request) -> EnvironmentReport:
    """The discovery `EnvironmentReport` built at startup, for use as a FastAPI dependency.

    Args:
        request: The incoming request, carrying the app whose `state.report` was set in lifespan.

    Returns:
        The process-wide environment report.

    Raises:
        HTTPException: 503 when no report is available — the startup build failed, or the lifespan
            that stashes it never ran.
    """
    # app.state is Starlette's dynamic namespace; `report` is unset until the lifespan stashes it,
    # so read it reflectively to map "never set" to a 503 instead of an AttributeError 500
    report = getattr(request.app.state, "report", None)
    if not isinstance(report, EnvironmentReport):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="environment report unavailable",
        )
    return report


ReportDep = Annotated[EnvironmentReport, Depends(get_report)]


def get_capability_router(request: Request) -> CapabilityRouter:
    """The `CapabilityRouter` built at startup, for use as a FastAPI dependency.

    Args:
        request: The incoming request, carrying the app whose `state.capability_router` was set
            in lifespan.

    Returns:
        The process-wide capability router.
    """
    capability_router = request.app.state.capability_router
    assert isinstance(capability_router, CapabilityRouter)
    return capability_router


CapabilityRouterDep = Annotated[CapabilityRouter, Depends(get_capability_router)]


def get_pool(request: Request) -> AsyncAgentPool:
    """The `AsyncAgentPool` opened at startup, for use as a FastAPI dependency.

    Args:
        request: The incoming request, carrying the app whose `state.pool` was set in lifespan.

    Returns:
        The process-wide pool over the configured `pool_db`.
    """
    pool = request.app.state.pool
    assert isinstance(pool, AsyncAgentPool)
    return pool


PoolDep = Annotated[AsyncAgentPool, Depends(get_pool)]


def get_core_settings(request: Request) -> AgentFleetSettings:
    """The core `AgentFleetSettings` resolved at startup, for use as a FastAPI dependency.

    Args:
        request: The incoming request, carrying the app whose `state.core` was set in lifespan.

    Returns:
        The process-wide core settings.
    """
    core = request.app.state.core
    assert isinstance(core, AgentFleetSettings)
    return core


CoreSettingsDep = Annotated[AgentFleetSettings, Depends(get_core_settings)]


def verify_token(request: Request) -> None:
    """Enforce bearer-token auth when an API token is configured.

    A no-op when `api_token` is unset (the loopback-only default), so localhost usage stays
    auth-free. When a token is set, every protected endpoint requires a matching
    `Authorization: Bearer <token>`; the comparison is constant-time.

    Args:
        request: The incoming request, carrying the app whose `state.settings` holds the token.

    Raises:
        HTTPException: 401 when a token is configured and the request's bearer token is missing
            or wrong; 500 when `state.settings` is missing or not `ApiSettings` (fails closed).
    """
    settings = request.app.state.settings
    if not isinstance(settings, ApiSettings):
        # not an assert: this guards an auth decision and must hold under `python -O`
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API settings not configured",
        )
    if settings.api_token is None:
        return
    expected = f"Bearer {settings.api_token.get_secret_value()}"
    if not secrets.compare_digest(request.headers.get("Authorization", ""), expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing or invalid bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )


AuthDep = Depends(verify_token)
