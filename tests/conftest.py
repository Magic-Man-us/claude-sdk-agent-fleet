from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from agent_fleet import Catalog
from capdisc.settings import get_settings

_FIXTURE = Path(__file__).parent / "fixtures" / "mock_catalog.json"

# The domain vocabulary the test suite's skill fixtures declare. `get_settings()` reads real
# config files off disk by default, so without this the domain-scan tests would depend on
# whatever ~/.claude/capdisc/config.{json,yaml} happens to hold on the machine
# running them. Set here instead, so the suite is hermetic regardless of the developer's own
# config.
_TEST_DOMAIN_TAGS = '{"security": {}, "documentation": {}, "web": {}}'


@pytest.fixture(autouse=True)
def _domain_tags_vocab(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("CAPABILITIES_DISCOVERY_DOMAIN_TAGS", _TEST_DOMAIN_TAGS)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def catalog() -> Catalog:
    return Catalog.model_validate_json(_FIXTURE.read_text(encoding="utf-8"))
