from __future__ import annotations

import pytest

from agent_fleet_api.app import create_app
from agent_fleet_api.settings import ApiSettings


def test_app_builds_and_exposes_the_expected_routes() -> None:
    """The FastAPI app assembles and its OpenAPI describes every endpoint. Catches a broken
    route registration or a model that won't serialize into the schema — without starting a
    server or running the scan-on-startup lifespan (which `create_app`/`openapi()` do not)."""
    schema = create_app().openapi()
    assert schema["info"]["title"] == "Agent Fleet API"
    assert {"/healthz", "/catalog", "/generate", "/render"} <= set(schema["paths"])


def test_openapi_exposes_the_pydantic_component_schemas() -> None:
    """The point of the API is to surface the core's Pydantic models as an OpenAPI schema, so the
    components section must be populated."""
    schemas = create_app().openapi().get("components", {}).get("schemas", {})
    assert schemas


def test_loopback_default_builds_without_a_token() -> None:
    # the default host is 127.0.0.1, so no token is required to build/serve.
    assert create_app(ApiSettings(host="127.0.0.1")) is not None


def test_non_loopback_without_token_is_refused() -> None:
    # binding the unauthenticated API to a public interface without an opt-in token must fail.
    with pytest.raises(RuntimeError, match="non-loopback"):
        create_app(ApiSettings(host="0.0.0.0", api_token=None))  # noqa: S104 — asserts refusal


def test_non_loopback_with_token_is_allowed() -> None:
    # an explicit token is the deliberate opt-in that unlocks off-loopback binding.
    assert create_app(ApiSettings(host="0.0.0.0", api_token="opt-in")) is not None  # noqa: S104, S106
