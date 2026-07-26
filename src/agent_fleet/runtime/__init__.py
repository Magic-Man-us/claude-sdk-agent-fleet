"""Dependency-light fixed-agent pool and capture runtime.

No capability discovery, agent assembly, skill routing, or catalog package is
imported by this namespace.
"""

from .capture import run_runtime_with_capture
from .models import (
    RuntimeAgentRunRecord,
    RuntimeAgentSpec,
    RuntimePoolEntry,
    RuntimeRunOutcome,
    RuntimeRunRecord,
)
from .pool import RuntimeAgentPool

__all__ = [
    "RuntimeAgentPool",
    "RuntimeAgentRunRecord",
    "RuntimeAgentSpec",
    "RuntimePoolEntry",
    "RuntimeRunOutcome",
    "RuntimeRunRecord",
    "run_runtime_with_capture",
]
