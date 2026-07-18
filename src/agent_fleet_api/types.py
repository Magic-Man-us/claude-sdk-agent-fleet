from __future__ import annotations

from typing import Annotated

from pydantic import Field

SdkSource = Annotated[
    str,
    Field(
        title="SDK source",
        description="Rendered Claude Agent SDK program text for an assembled spec.",
    ),
]

Hostname = Annotated[
    str,
    Field(title="Host", description="Network interface the API binds to.", examples=["127.0.0.1"]),
]

Port = Annotated[
    int,
    Field(
        ge=1,
        le=65535,
        title="Port",
        description="TCP port the API listens on.",
        examples=[8000],
    ),
]

CorsOrigin = Annotated[
    str,
    Field(
        title="CORS origin",
        description="A browser origin permitted to call the API.",
        examples=["http://localhost:5173"],
    ),
]
