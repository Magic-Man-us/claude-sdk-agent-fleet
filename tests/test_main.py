from __future__ import annotations

import pytest

from agent_fleet.main import _MODULE_RUN, _prog


def test_prog_uses_console_script_name(monkeypatch: pytest.MonkeyPatch) -> None:
    # invoked as the installed console script: argv[0] is the script path
    monkeypatch.setattr("sys.argv", ["/usr/local/bin/agent-fleet", "do a thing"])
    assert _prog() == "agent-fleet"


def test_prog_falls_back_to_module_form_for_m_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    # `python -m agent_fleet.main` sets argv[0] to the module file path
    monkeypatch.setattr("sys.argv", ["/repo/src/agent_fleet/main.py"])
    assert _prog() == _MODULE_RUN


def test_prog_falls_back_when_argv0_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", [""])
    assert _prog() == _MODULE_RUN
