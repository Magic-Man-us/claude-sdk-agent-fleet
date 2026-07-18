from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import SettingsConfigDict

from agent_fleet.settings import AgentFleetSettings
from capdisc.scope import ScopeInventory, ScopeKind, ScopeRoots
from capdisc.settings import ExtraSourceDir
from helpers import touch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _settings_with_json(json_file: Path, **env: str) -> AgentFleetSettings:
    """Build a settings instance that reads from `json_file`, with optional extra env vars."""

    class _Scoped(AgentFleetSettings):
        model_config = SettingsConfigDict(
            env_prefix="AGENT_FLEET_",
            env_file=json_file.parent / "nonexistent.env",
            extra="ignore",
            json_file=json_file,
        )

    keys = {"AGENT_FLEET_EXTRA_SCAN_DIRS", "AGENT_FLEET_EXTRA_PLUGIN_DIRS"} | set(env)
    old = {k: os.environ.pop(k, None) for k in keys}
    os.environ.update(env)
    try:
        return _Scoped()
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---------------------------------------------------------------------------
# ExtraSourceDir alias
# ---------------------------------------------------------------------------


def test_extra_source_dir_alias_carries_title() -> None:
    import typing

    meta = next(m for m in typing.get_args(ExtraSourceDir) if hasattr(m, "title"))
    assert meta.title == "Extra source directory"


# ---------------------------------------------------------------------------
# Core settings defaults
# ---------------------------------------------------------------------------


def test_defaults_are_empty_lists(tmp_path: Path) -> None:
    s = _settings_with_json(tmp_path / "nonexistent.json")
    assert s.extra_scan_dirs == []
    assert s.extra_plugin_dirs == []


def test_missing_json_config_file_is_tolerated(tmp_path: Path) -> None:
    s = _settings_with_json(tmp_path / "missing.json")
    assert s.extra_scan_dirs == []


# ---------------------------------------------------------------------------
# JSON config file source
# ---------------------------------------------------------------------------


def test_loads_extra_scan_dirs_from_json_file(tmp_path: Path) -> None:
    cfg = tmp_path / "config.json"
    d = tmp_path / "scans"
    d.mkdir()
    cfg.write_text(
        f'{{"extra_scan_dirs": ["{d}"]}}'.encode().decode(),
        encoding="utf-8",
    )
    s = _settings_with_json(cfg)
    assert s.extra_scan_dirs == [d]


def test_loads_extra_plugin_dirs_from_json_file(tmp_path: Path) -> None:
    cfg = tmp_path / "config.json"
    d = tmp_path / "plugins"
    d.mkdir()
    cfg.write_text(
        f'{{"extra_plugin_dirs": ["{d}"]}}'.encode().decode(),
        encoding="utf-8",
    )
    s = _settings_with_json(cfg)
    assert s.extra_plugin_dirs == [d]


# ---------------------------------------------------------------------------
# Env var source — list[Path] uses JSON array format
# ---------------------------------------------------------------------------


def test_loads_extra_scan_dirs_from_env(tmp_path: Path) -> None:
    d = tmp_path / "env-scan"
    d.mkdir()
    s = _settings_with_json(
        tmp_path / "nonexistent.json",
        AGENT_FLEET_EXTRA_SCAN_DIRS=f'["{d}"]',
    )
    assert s.extra_scan_dirs == [d]


def test_loads_extra_plugin_dirs_from_env(tmp_path: Path) -> None:
    d = tmp_path / "env-plugin"
    d.mkdir()
    s = _settings_with_json(
        tmp_path / "nonexistent.json",
        AGENT_FLEET_EXTRA_PLUGIN_DIRS=f'["{d}"]',
    )
    assert s.extra_plugin_dirs == [d]


# ---------------------------------------------------------------------------
# Precedence: env var wins over JSON file
# ---------------------------------------------------------------------------


def test_env_takes_precedence_over_json_file(tmp_path: Path) -> None:
    cfg = tmp_path / "config.json"
    file_dir = tmp_path / "from-file"
    file_dir.mkdir()
    env_dir = tmp_path / "from-env"
    env_dir.mkdir()
    cfg.write_text(
        f'{{"extra_scan_dirs": ["{file_dir}"]}}'.encode().decode(),
        encoding="utf-8",
    )
    s = _settings_with_json(cfg, AGENT_FLEET_EXTRA_SCAN_DIRS=f'["{env_dir}"]')
    assert s.extra_scan_dirs == [env_dir]


# ---------------------------------------------------------------------------
# Discovery wiring contract
# ---------------------------------------------------------------------------


def test_add_dirs_contributes_project_scope_root(tmp_path: Path) -> None:
    # --add-dir points at a directory outside the project tree; a sibling of `start`, not one
    # nested under it (a nested dir is already covered by the normal project/nested-skill walk).
    start = tmp_path / "proj"
    (start / ".git").mkdir(parents=True)
    extra = tmp_path / "extra"
    (extra / ".claude" / "agents").mkdir(parents=True)
    touch(extra / ".claude" / "agents" / "my-agent.md", "hello")

    roots = ScopeRoots.discover(start=start, add_dirs=[extra])
    inv = ScopeInventory.scan(roots)

    names = {a.name for a in inv.artifacts}
    assert "my-agent" in names
    scope = next(a for a in inv.artifacts if a.name == "my-agent").scope
    assert scope is ScopeKind.project


def test_plugin_dirs_contributes_plugin_scope_root(tmp_path: Path) -> None:
    plugin = tmp_path / "my-plugin"
    (plugin / "skills" / "demo").mkdir(parents=True)
    (plugin / "skills" / "demo" / "SKILL.md").write_text("skill", encoding="utf-8")

    roots = ScopeRoots.discover(start=tmp_path, plugin_dirs=[plugin])
    plugin_roots = [r for r in roots.roots if r.scope is ScopeKind.plugin]
    assert any(r.base == plugin for r in plugin_roots)


def test_discovery_scope_model_expands_and_dedupes_runtime_roots(tmp_path: Path) -> None:
    plugins_root = tmp_path / "plugins"
    installed = tmp_path / "installed"
    extra_plugin = tmp_path / "extra-plugin"
    extra_scan = tmp_path / "extra-scan"
    for directory in (plugins_root, installed, extra_plugin, extra_scan / ".claude"):
        directory.mkdir(parents=True)
    (plugins_root / "installed_plugins.json").write_text(
        f'{{"plugins": {{"demo@local": [{{"installPath": "{installed}"}}]}}}}',
        encoding="utf-8",
    )
    settings = AgentFleetSettings(
        plugins_root=plugins_root,
        extra_plugin_dirs=[installed, extra_plugin, extra_plugin],
        extra_scan_dirs=[extra_scan, extra_scan],
    )

    scope = settings.discovery_scope(start=tmp_path, home_dir=tmp_path)

    assert scope.plugin_dirs() == [installed, extra_plugin]
    assert scope.add_dirs() == [extra_scan]
    roots = scope.roots()
    assert any(root.scope is ScopeKind.plugin and root.base == installed for root in roots.roots)
    assert any(root.scope is ScopeKind.plugin and root.base == extra_plugin for root in roots.roots)


# ---------------------------------------------------------------------------
# Path fields — defaults and env override
# ---------------------------------------------------------------------------


def test_path_defaults_point_at_standard_locations(tmp_path: Path) -> None:
    s = _settings_with_json(tmp_path / "missing.json")
    assert s.plugins_root == Path.home() / ".claude" / "plugins"
    assert s.claude_json == Path.home() / ".claude.json"
    assert s.user_settings == Path.home() / ".claude" / "settings.json"
    # These two default to a directory owned by the external capdisc package
    # (DiscoverySettings), unaffected by this repo's agent-generator -> agent-fleet rename.
    assert s.mcp_cache == Path.home() / ".claude" / "capdisc" / "mcp-tools.json"
    assert s.report_dir == Path.home() / ".claude" / "capdisc"
    assert s.agent_dir is None  # output targets unset until configured
    assert s.skill_dir is None  # output targets unset until configured


def test_env_overrides_a_path(tmp_path: Path) -> None:
    s = _settings_with_json(
        tmp_path / "missing.json",
        AGENT_FLEET_MCP_CACHE=str(tmp_path / "cache.json"),
        AGENT_FLEET_AGENT_DIR=str(tmp_path / "agents"),
    )
    assert s.mcp_cache == tmp_path / "cache.json"
    assert s.agent_dir == tmp_path / "agents"
