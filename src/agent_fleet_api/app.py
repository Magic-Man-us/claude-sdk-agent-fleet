from __future__ import annotations

import asyncio
import ipaddress
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agent_fleet.engine.pool import AgentPool, AsyncAgentPool
from agent_fleet.router.capability import CapabilityRouter
from agent_fleet.settings import AgentFleetSettings, DiscoveryScope
from capdisc.discovery import scan_environment
from capdisc.mcp_catalog import enumerate_mcp_servers
from capdisc.mcp_harvest import (
    cache_is_stale,
    read_mcp_cache,
    refresh_in_background,
)
from capdisc.plugin_catalog import enumerate_plugins
from capdisc.report import write_report_on_start

from .deps import Engine
from .routes import router
from .settings import ApiSettings


def _core_settings(settings: ApiSettings) -> AgentFleetSettings:
    class _Core(AgentFleetSettings):
        # Mirrors the external capdisc.DiscoverySettings config directory
        # (unaffected by this repo's own agent-generator -> agent-fleet rename), just
        # parametrized by `settings.home_dir` instead of `Path.home()` for testability.
        model_config = {
            **AgentFleetSettings.model_config,
            "json_file": settings.home_dir / ".claude" / "capdisc" / "config.json",
        }

    return _Core()


def _build_capability_router(settings: ApiSettings, scope: DiscoveryScope) -> CapabilityRouter:
    """Build the capability router using the same discovery recipe as the MCP server entry-point.

    Separated from the lifespan to keep the async context manager readable and to make the
    recipe easy to spot and compare against mcp_server._router().

    Args:
        settings: The API settings supplying the home dir for the plugins root.
        scope: The already-resolved discovery scope whose roots the router indexes over; passed in
            so startup does not re-scan the filesystem to recompute it.

    Returns:
        A router indexed over the discovered scope roots, the enriched MCP cache (falling back to
        the live list), and the installed plugins.
    """
    plugins_root = settings.home_dir / ".claude" / "plugins"
    roots = scope.roots()
    mcp_servers = read_mcp_cache() or enumerate_mcp_servers()
    if cache_is_stale():
        refresh_in_background(plugins_root=plugins_root)
    return CapabilityRouter.from_environment(
        roots, mcp_servers=mcp_servers, plugins=enumerate_plugins(plugins_root)
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Scan the environment once at startup and stash the built `Engine` and `CapabilityRouter`
    on `app.state`.

    Args:
        app: The application; `state.settings` supplies the scan roots, and `state.engine` /
            `state.capability_router` are populated for request handlers.
    """
    settings = app.state.settings
    assert isinstance(settings, ApiSettings)
    plugins_root = settings.home_dir / ".claude" / "plugins"
    core = _core_settings(settings)
    scope = core.discovery_scope(
        start=settings.scan_root,
        home_dir=settings.home_dir,
        plugins_root=plugins_root,
    )
    app.state.core = core
    app.state.engine = Engine(scan_environment(scope.roots()))
    app.state.capability_router = _build_capability_router(settings, scope)
    app.state.pool = AsyncAgentPool(AgentPool(core.pool_db))
    # write_report_on_start's harvest chain calls asyncio.run internally, which raises if invoked
    # on the already-running loop; dispatch it to a worker thread (no running loop there), the same
    # off-loop pattern _build_capability_router uses for the MCP refresh.
    app.state.report = await asyncio.to_thread(write_report_on_start)
    yield


def _is_loopback(host: str) -> bool:
    """Whether `host` binds to the loopback interface only (`localhost`, `127.0.0.0/8`, `::1`)."""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _enforce_bind_safety(settings: ApiSettings) -> None:
    """Refuse to expose the API off the loopback interface without an explicit opt-in token.

    `/orchestrate` spawns a real agent, so binding to a non-loopback host is only safe as a
    deliberate choice. `AGENT_FLEET_API_TOKEN` serves both roles: it gates this bind (set it to
    bind off-loopback), and `deps.verify_token` / `AuthDep` check it per request on every route
    except `/healthz`, so once it is set requests without a matching bearer token are rejected.

    Raises:
        RuntimeError: When `settings.host` is non-loopback and no token is configured.
    """
    if _is_loopback(settings.host) or settings.api_token is not None:
        return
    raise RuntimeError(
        f"refusing to bind the unauthenticated API to non-loopback host {settings.host!r}: "
        "set AGENT_FLEET_API_TOKEN to bind off-loopback "
        "(and add request auth before exposing it)"
    )


def _package_version() -> str:
    """The installed `agent-fleet-api` version, or `"0.0.0"` when the package has no metadata.

    Falls back for a source/editable checkout run without an installed distribution, where
    `importlib.metadata` finds no record for the package.
    """
    try:
        return version("agent-fleet-api")
    except PackageNotFoundError:
        return "0.0.0"


def create_app(settings: ApiSettings | None = None, *, enforce_bind_safety: bool = True) -> FastAPI:
    """Build the FastAPI app — CORS, routes, and the startup catalog scan wired up.

    Args:
        settings: The HTTP-layer config; loaded from the environment when None.
        enforce_bind_safety: When True (the default), refuse to build an off-loopback app without
            `AGENT_FLEET_API_TOKEN`. Set False for offline uses that never bind a socket (e.g.
            OpenAPI schema export), where the host/token env vars are irrelevant.

    Returns:
        The configured application.
    """
    config = settings or ApiSettings()
    if enforce_bind_safety:
        _enforce_bind_safety(config)
    app = FastAPI(title="Agent Fleet API", version=_package_version(), lifespan=lifespan)
    app.state.settings = config
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.cors_origins),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


def main() -> None:
    """Run the API under uvicorn on the configured host and port."""
    config = ApiSettings()
    uvicorn.run(create_app(config), host=config.host, port=config.port)
