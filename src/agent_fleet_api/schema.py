from __future__ import annotations

import sys
from pathlib import Path

from pydantic_core import to_json

from .app import create_app


def build_openapi() -> bytes:
    """The app's OpenAPI document as JSON bytes.

    Derived entirely from the Pydantic models the routes declare. No environment scan happens
    (the lifespan that builds the catalog never runs) and bind safety is not enforced (the schema
    never binds a socket, so `AGENT_FLEET_API_HOST`/`AGENT_FLEET_API_TOKEN` are irrelevant here),
    so this is safe to call offline as a build step.
    """
    return to_json(create_app(enforce_bind_safety=False).openapi())


def main() -> None:
    """Write the OpenAPI JSON to the path in argv[1], or stdout when none is given."""
    data = build_openapi()
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_bytes(data)
    else:
        sys.stdout.buffer.write(data)
