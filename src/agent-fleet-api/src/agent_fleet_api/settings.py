from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from .types import CorsOrigin, Hostname, Port

_DEFAULT_CORS_ORIGINS: list[CorsOrigin] = []


class ApiSettings(BaseSettings):
    """HTTP-layer configuration. Loaded from `AGENT_FLEET_API_*` env vars and an optional
    `.env`; the scan roots default to the process cwd and home so the live catalog mirrors what
    the CLI would discover."""

    model_config = SettingsConfigDict(env_prefix="AGENT_FLEET_API_", env_file=".env")

    host: Hostname = "127.0.0.1"
    port: Port = 8000
    scan_root: Path = Field(default_factory=Path.cwd)
    home_dir: Path = Field(default_factory=Path.home)
    cors_origins: list[CorsOrigin] = _DEFAULT_CORS_ORIGINS
    api_token: SecretStr | None = None
