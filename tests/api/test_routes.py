from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent_fleet_api.app import create_app
from agent_fleet_api.settings import ApiSettings

_TOKEN = "test-bearer-token"  # noqa: S105 — test fixture, not a real secret


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app()) as started:
        yield started


def test_healthz(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}


def test_catalog_lists_entries(client: TestClient) -> None:
    body = client.get("/catalog").json()
    assert isinstance(body["entries"], list)


def test_report_returns_environment_report(client: TestClient) -> None:
    from capdisc.report import EnvironmentReport

    response = client.get("/report")
    assert response.status_code == 200
    EnvironmentReport.model_validate(response.json())  # the body is the typed snapshot


def test_report_stashed_on_app_state(client: TestClient) -> None:
    from capdisc.report import EnvironmentReport

    assert isinstance(client.app.state.report, EnvironmentReport)


def test_generate_then_render_round_trip(client: TestClient) -> None:
    generated = client.post(
        "/generate",
        json={"task": "read log files and summarize errors", "name": "log-summarizer"},
    )
    assert generated.status_code == 200
    spec = generated.json()["spec"]
    assert spec["name"] == "log-summarizer"

    # post /generate's output straight to /render — the spec carries no derived field, so the
    # same AgentSpec shape flows through both endpoints with no client-side surgery
    assert "allowed_tools" not in spec
    rendered = client.post("/render", json=spec)
    assert rendered.status_code == 200
    assert "ClaudeAgentOptions" in rendered.json()["source"]


def _rendered_body(client: TestClient) -> dict[str, object]:
    generated = client.post(
        "/generate",
        json={"task": "read log files and summarize errors", "name": "log-summarizer"},
    )
    assert generated.status_code == 200
    rendered = client.post("/render", json=generated.json()["spec"])
    assert rendered.status_code == 200
    body: dict[str, object] = rendered.json()
    return body


def test_render_persists_when_agent_dir_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    agent_dir = tmp_path / "agents"
    monkeypatch.setenv("AGENT_FLEET_AGENT_DIR", str(agent_dir))
    with TestClient(create_app(ApiSettings(home_dir=tmp_path))) as c:
        body = _rendered_body(c)
    assert body["path"] is not None
    written = Path(str(body["path"]))
    assert written == agent_dir / "log-summarizer.py"
    assert written.read_text(encoding="utf-8") == body["source"]


def test_render_skips_persistence_without_agent_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AGENT_FLEET_AGENT_DIR", raising=False)
    with TestClient(create_app(ApiSettings(home_dir=tmp_path))) as c:
        body = _rendered_body(c)
    assert body["path"] is None


def test_capability_router_stashed_on_app_state(client: TestClient) -> None:
    from agent_fleet.router.capability import CapabilityRouter

    # the /orchestrate dep resolves the router the lifespan stashes
    assert isinstance(client.app.state.capability_router, CapabilityRouter)


def test_orchestrate_returns_outcome(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from agent_fleet.engine.orchestrate import OrchestrateOutcome

    fixed = OrchestrateOutcome(output="done", spec=None)

    async def _fake_collect(task: object, router: object) -> OrchestrateOutcome:
        return fixed

    monkeypatch.setattr("agent_fleet_api.routes.collect_orchestration", _fake_collect)

    response = client.post("/orchestrate", json={"task": "build me an agent that reads logs"})
    assert response.status_code == 200
    body = response.json()
    assert body["output"] == "done"
    assert body["spec"] is None


def test_no_token_configured_serves_without_auth(client: TestClient) -> None:
    # the default app configures no token, so protected endpoints need no Authorization header
    assert client.get("/catalog").status_code == 200


def test_every_protected_endpoint_enforces_token(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent_fleet.engine.orchestrate import OrchestrateOutcome

    async def _fake_collect(task: object, router: object) -> OrchestrateOutcome:
        return OrchestrateOutcome(output="done", spec=None)

    monkeypatch.setattr("agent_fleet_api.routes.collect_orchestration", _fake_collect)
    auth = {"Authorization": f"Bearer {_TOKEN}"}
    gen_body = {"task": "read log files and summarize errors", "name": "log-summarizer"}
    orch_body = {"task": "read the logs now"}
    bad = {"Authorization": "Bearer wrong"}
    with TestClient(create_app(ApiSettings(api_token=_TOKEN))) as c:
        generated = c.post("/generate", json=gen_body, headers=auth)
        assert generated.status_code == 200  # fail clearly here, not with a later KeyError
        spec = generated.json()["spec"]
        # each protected route, as a header-parameterized call (no getattr dispatch)
        calls = [
            ("/catalog", lambda h: c.get("/catalog", headers=h)),
            ("/report", lambda h: c.get("/report", headers=h)),
            ("/generate", lambda h: c.post("/generate", json=gen_body, headers=h)),
            ("/render", lambda h: c.post("/render", json=spec, headers=h)),
            ("/orchestrate", lambda h: c.post("/orchestrate", json=orch_body, headers=h)),
        ]
        for path, call in calls:
            assert call({}).status_code == 401, f"{path} served without auth"
            assert call(bad).status_code == 401, f"{path} took a bad token"
            assert call(auth).status_code == 200, f"{path} rejected a valid token"


def test_healthz_stays_open_when_token_configured() -> None:
    with TestClient(create_app(ApiSettings(api_token=_TOKEN))) as c:
        assert c.get("/healthz").status_code == 200


@pytest.fixture
def pool_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("AGENT_FLEET_POOL_DB", str(tmp_path / "pool.db"))
    with TestClient(create_app(ApiSettings(home_dir=tmp_path))) as started:
        yield started


_SAVE_BODY = {"task": "read log files and summarize errors", "name": "log-summarizer"}


def test_pool_save_get_list_find_delete(pool_client: TestClient) -> None:
    saved = pool_client.post("/pool/INC-1", json=_SAVE_BODY)
    assert saved.status_code == 200
    assert saved.json()["agent_key"] == "INC-1"

    got = pool_client.get("/pool/INC-1")
    assert got.status_code == 200
    assert got.json()["name"] == "log-summarizer"

    assert pool_client.get("/pool/UNKNOWN").status_code == 404

    listed = pool_client.get("/pool").json()
    assert [e["agent_key"] for e in listed] == ["INC-1"]

    found = pool_client.get("/pool/find", params={"query": "summarize log file errors"})
    assert found.status_code == 200
    assert any(e["agent_key"] == "INC-1" for e in found.json())

    assert pool_client.delete("/pool/INC-1").status_code == 204
    assert pool_client.delete("/pool/INC-1").status_code == 404
    assert pool_client.get("/pool/INC-1").status_code == 404


def test_pool_runs_endpoint_lists_recorded_runs(pool_client: TestClient) -> None:
    pool_client.post("/pool/INC-2", json=_SAVE_BODY)
    assert pool_client.get("/pool/INC-2/runs").json() == []  # no runs yet


def test_pool_run_captures_and_returns_outcome(
    pool_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_fleet import RunOutcome, RunRecord

    pool_client.post("/pool/INC-3", json=_SAVE_BODY)

    async def _fake_capture(
        pool: object,
        agent_key: object,
        task: object,
        options: object,
        *,
        run: RunRecord | None = None,
        prompt: object = None,
    ) -> RunOutcome:
        finished = pool.finish_run(run.run_id)  # type: ignore[attr-defined,union-attr]
        return RunOutcome(output="captured", run=finished, agent_runs=[])

    monkeypatch.setattr("agent_fleet_api.routes.run_with_capture", _fake_capture)

    response = pool_client.post("/pool/INC-3/run", json={"task": "read the logs and report now"})
    assert response.status_code == 200
    body = response.json()
    assert body["output"] == "captured"
    assert body["run"]["finished_at"] is not None
    assert body["agent_runs"] == []

    runs = pool_client.get("/pool/INC-3/runs").json()
    assert len(runs) == 1  # the fake capture recorded one run through the real pool
    assert isinstance(RunRecord.model_validate(runs[0]), RunRecord)


def test_pool_run_404_for_unknown_entry(pool_client: TestClient) -> None:
    response = pool_client.post("/pool/NOPE/run", json={"task": "read the logs and report now"})
    assert response.status_code == 404


def test_pool_run_agents_endpoint_lists_captured_agents(
    pool_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent_fleet import AgentRunRecord, RunOutcome, RunRecord

    saved = pool_client.post("/pool/INC-4", json=_SAVE_BODY).json()

    async def _fake_capture(
        pool: object,
        agent_key: object,
        task: object,
        options: object,
        *,
        run: RunRecord | None = None,
        prompt: object = None,
    ) -> RunOutcome:
        pool.record_agent_run(run.run_id, saved["session_id"])  # type: ignore[attr-defined,union-attr]
        finished = pool.finish_run(run.run_id)  # type: ignore[attr-defined,union-attr]
        agent_runs = pool.list_agent_runs(run.run_id)  # type: ignore[attr-defined,union-attr]
        return RunOutcome(output="captured", run=finished, agent_runs=agent_runs)

    monkeypatch.setattr("agent_fleet_api.routes.run_with_capture", _fake_capture)

    response = pool_client.post("/pool/INC-4/run", json={"task": "read the logs and report now"})
    assert response.status_code == 200
    run_id = response.json()["run"]["run_id"]

    agents = pool_client.get(f"/pool/INC-4/runs/{run_id}/agents").json()
    assert len(agents) == 1
    assert isinstance(AgentRunRecord.model_validate(agents[0]), AgentRunRecord)
    assert agents[0]["session_id"] == saved["session_id"]


def test_pool_findings_endpoint_empty_for_no_findings(pool_client: TestClient) -> None:
    pool_client.post("/pool/INC-5", json=_SAVE_BODY)
    assert pool_client.get("/pool/INC-5/findings").json() == []
