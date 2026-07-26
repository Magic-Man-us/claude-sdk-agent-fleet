from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import BeforeValidator, Field

from capdisc.base import FrozenModel
from capdisc.plugin_catalog import installed_plugin_dirs
from capdisc.scope import ScopeRoots, default_managed_dir
from capdisc.settings import DiscoverySettings

NotifyCommand = Annotated[
    str,
    Field(
        min_length=1,
        title="Notify command",
        description="Shell command a Stop hook runs when a teammate run finishes. Receives "
        "Claude Code's hook-input JSON on stdin, which identifies the finished session — a user "
        "script can disambiguate which teammate fired.",
        examples=["notify-send teammate-done"],
    ),
]


def _blank_to_none(value: object) -> object:
    """Map a blank/whitespace-only string to None so `notify_command` reads as unset. Any other
    value — including a non-string — passes through untouched, so a genuine type error (an int, a
    list, ...) still raises instead of silently disabling the hook."""
    if isinstance(value, str):
        return value if value.strip() else None
    return value


class DiscoveryScope(FrozenModel):
    """Runtime inputs for resolving Claude Code discovery scope roots."""

    start: Path
    home_dir: Path
    managed_dir: Path | None
    plugins_root: Path
    extra_plugin_dirs: list[Path] = []
    extra_scan_dirs: list[Path] = []

    def plugin_dirs(self) -> list[Path]:
        """Installed and configured plugin roots, de-duplicated in discovery order."""
        paths = installed_plugin_dirs(self.plugins_root) + self.extra_plugin_dirs
        return list(dict.fromkeys(paths))

    def add_dirs(self) -> list[Path]:
        """Configured extra scan roots, de-duplicated in discovery order."""
        return list(dict.fromkeys(self.extra_scan_dirs))

    def roots(self) -> ScopeRoots:
        """Resolved scope roots for scanning capabilities."""
        return ScopeRoots.discover(
            start=self.start,
            home_dir=self.home_dir,
            managed_dir=self.managed_dir,
            plugin_dirs=self.plugin_dirs(),
            add_dirs=self.add_dirs(),
        )


class AgentFleetSettings(DiscoverySettings):
    """Discovery settings plus the generator's output locations.

    Inherited fields keep DiscoverySettings' `CAPABILITIES_DISCOVERY_` env prefix, shared
    across every consumer of capdisc. This class's own three fields are
    outside that shared discovery namespace, so each carries an explicit `validation_alias`
    under `AGENT_FLEET_` instead — matching the sibling `AGENT_FLEET_API_`-prefixed ApiSettings.
    """

    agent_dir: Path | None = Field(
        default=None,
        description="Where generated agents are written; None until configured.",
        validation_alias="AGENT_FLEET_AGENT_DIR",
    )
    skill_dir: Path | None = Field(
        default=None,
        description="Where generated skills are written; None until configured.",
        validation_alias="AGENT_FLEET_SKILL_DIR",
    )
    pool_db: Path = Field(
        default_factory=lambda: Path.home() / ".claude" / "agent-fleet" / "pool.db",
        description="SQLite database backing the pool of named, resumable agent sessions.",
        validation_alias="AGENT_FLEET_POOL_DB",
    )
    notify_command: Annotated[
        NotifyCommand | None,
        BeforeValidator(_blank_to_none),
    ] = Field(
        default=None,
        description="Shell command a Stop hook runs when a teammate run finishes; None (or an "
        "empty/whitespace-only value) disables it.",
        validation_alias="AGENT_FLEET_NOTIFY_COMMAND",
    )
    roster: Path | None = Field(
        default=None,
        description="Explicit path to a roster TOML file; None falls back to the project file "
        "then the user file, then the roster shipped with this package.",
        validation_alias="AGENT_FLEET_ROSTER",
    )
    subagent_model: Annotated[
        str | None,
        BeforeValidator(_blank_to_none),
    ] = Field(
        default=None,
        description="Claude Code's own subagent-model override, honoured here because a teammate "
        "is a subagent of its caller. Wins over a template's model; 'inherit' (or unset) defers "
        "to the template.",
        validation_alias="CLAUDE_CODE_SUBAGENT_MODEL",
    )
    codex_enabled: bool = Field(
        default=True,
        description="Mount the guarded Codex MCP worker on live and generated Claude agents.",
        validation_alias="AGENT_FLEET_CODEX_ENABLED",
    )
    codex_allow_workspace_write: bool = Field(
        default=False,
        description="Allow Codex workspace-write calls, still restricted to clean detached linked "
        "worktrees under codex_allowed_roots.",
        validation_alias="AGENT_FLEET_CODEX_ALLOW_WORKSPACE_WRITE",
    )
    codex_allowed_roots: list[Path] = Field(
        default_factory=list,
        description="Roots Codex cwd values may live under; empty means the fleet process cwd.",
        validation_alias="AGENT_FLEET_CODEX_ALLOWED_ROOTS",
    )
    codex_max_timeout_seconds: int = Field(
        default=3_600,
        ge=30,
        le=7_200,
        description="Operator ceiling for one Codex worker call.",
        validation_alias="AGENT_FLEET_CODEX_MAX_TIMEOUT_SECONDS",
    )

    def discovery_scope(
        self,
        *,
        start: Path,
        home_dir: Path | None = None,
        managed_dir: Path | None = None,
        plugins_root: Path | None = None,
    ) -> DiscoveryScope:
        """Build the model that resolves runtime capability discovery roots."""
        return DiscoveryScope(
            start=start,
            home_dir=home_dir or Path.home(),
            managed_dir=managed_dir or default_managed_dir(),
            plugins_root=plugins_root or self.plugins_root,
            extra_plugin_dirs=self.extra_plugin_dirs,
            extra_scan_dirs=self.extra_scan_dirs,
        )


def current_discovery_scope() -> DiscoveryScope:
    """The discovery scope for this process: default `AgentFleetSettings`, rooted at `Path.cwd()`.

    Every process-wide router/source builder in the skill-router and pool MCP servers needs
    exactly this — resolve settings, then scope them to the current working directory. Centralized
    so that recipe exists in one place rather than being re-typed at each `@cache`-d builder.
    Callers that need a different root or a non-default settings source (e.g. the HTTP API, which
    is parametrized by request settings for testability) build a `DiscoveryScope` directly instead.
    """
    return AgentFleetSettings().discovery_scope(start=Path.cwd())
