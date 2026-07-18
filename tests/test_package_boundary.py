from __future__ import annotations

import re
from pathlib import Path

import capdisc

_GENERATOR_IMPORT = re.compile(r"^\s*(?:from|import)\s+agent_fleet\b", re.MULTILINE)


def test_discovery_package_never_imports_the_generator() -> None:
    """The layering is one-way: the generator imports discovery, never the reverse."""
    package_root = Path(capdisc.__file__).parent
    offenders = [
        str(path.relative_to(package_root))
        for path in sorted(package_root.rglob("*.py"))
        if _GENERATOR_IMPORT.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []
