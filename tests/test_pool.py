from __future__ import annotations

import asyncio
import inspect
import uuid
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from agent_fleet import (
    AgentPool,
    AgentSpec,
    AsyncAgentPool,
    Catalog,
    InMemoryCatalogSource,
    PoolEntry,
    create_agent,
    slugify_name,
)
from agent_fleet.engine.render import to_options
from agent_fleet.models.agent import AgentName, ProblemRequest

_PROMPT = "You are auditor. Audit the code for vulnerabilities and stop."
_AGENT_KEY = "PROJ-4821"
_TASK = "audit the codebase for security vulnerabilities now"


def _spec(
    system_prompt: str = _PROMPT,
    *,
    name: str = "auditor",
    description: str = "Audits code for vulnerabilities.",
    tags: tuple[str, ...] = (),
) -> AgentSpec:
    return AgentSpec(
        name=name,
        description=description,
        system_prompt=system_prompt,
        tools=("Read", "Grep"),
        tags=list(tags),
    )


def _pool(tmp_path: Path) -> AgentPool:
    return AgentPool(tmp_path / "pool.db")


def test_save_is_keyed_by_agent_key_not_name(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    first = pool.save("PROJ-1", _spec())
    second = pool.save("PROJ-2", _spec())  # same spec.name, different agent key

    assert first.agent_key == "PROJ-1"
    assert second.agent_key == "PROJ-2"
    assert first.name == second.name == "auditor"  # same display label, no collision
    assert {e.agent_key for e in pool.list()} == {"PROJ-1", "PROJ-2"}
    assert uuid.UUID(first.session_id)  # a real UUID, not the agent key


def test_resave_preserves_session_and_created_but_bumps_updated(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    first = pool.save(_AGENT_KEY, _spec())
    second = pool.save(
        _AGENT_KEY, _spec("You are auditor. Now audit dependencies as well and then stop.")
    )

    assert second.session_id == first.session_id  # default preserves the live session
    assert second.created_at == first.created_at
    assert second.updated_at >= first.updated_at
    assert second.spec.system_prompt != first.spec.system_prompt  # the spec updated


def test_reset_session_mints_a_new_uuid(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    first = pool.save(_AGENT_KEY, _spec())
    reset = pool.save(_AGENT_KEY, _spec(), reset_session=True)
    assert reset.session_id != first.session_id
    assert uuid.UUID(reset.session_id)


def test_get_by_key_returns_none_for_unknown_id(tmp_path: Path) -> None:
    assert _pool(tmp_path).get_by_key("nonexistent") is None


def test_get_by_key_returns_the_right_entry(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    saved = pool.save(_AGENT_KEY, _spec())
    assert pool.get_by_key(_AGENT_KEY) == saved


def test_list_orders_by_updated_descending(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    pool.save("PROJ-A", _spec())
    pool.save("PROJ-B", _spec(name="reviewer", description="Reviews pull requests."))
    pool.save("PROJ-A", _spec("You are auditor. Re-audit after the fix and then stop now here."))

    ids = [entry.agent_key for entry in pool.list()]
    assert ids == ["PROJ-A", "PROJ-B"]


def test_delete_removes_entry_and_reports_existence(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    pool.save(_AGENT_KEY, _spec())
    assert pool.delete(_AGENT_KEY) is True
    assert pool.get_by_key(_AGENT_KEY) is None
    assert pool.delete(_AGENT_KEY) is False


def test_find_ranks_the_closest_description_first(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    pool.save("P1", _spec(name="auditor", description="Audits code for security vulnerabilities."))
    pool.save("P2", _spec(name="curator", description="Summarizes git commits into changelog."))
    pool.save("P3", _spec(name="tester", description="Runs the pytest suite and reports failures."))

    results = pool.find("summarize git commit history into a changelog")
    assert results[0].agent_key == "P2"


def test_find_truncates_to_limit(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    for i in range(4):
        pool.save(f"P{i}", _spec(name=f"agent-{i}", description=f"Handles telemetry {i} now."))
    assert len(pool.find("telemetry", limit=2)) == 2


def test_find_is_deterministic(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    pool.save("P1", _spec(name="auditor", description="Audits code for security vulnerabilities."))
    pool.save("P2", _spec(name="curator", description="Summarizes git commits into a changelog."))

    query = "audit code security"
    first = [e.agent_key for e in pool.find(query)]
    second = [e.agent_key for e in pool.find(query)]
    assert first == second


def test_slugify_name_produces_a_valid_agent_name() -> None:
    adapter: TypeAdapter[AgentName] = TypeAdapter(AgentName)
    task = "Summarize the ENTIRE git commit history!! into grouped, human-readable changelog logs"
    slug = slugify_name(task)
    assert adapter.validate_python(slug) == slug  # satisfies the AgentName pattern
    assert len(slug) <= 64
    assert slug == "summarize-the-entire-git-commit-history"


def test_slugify_name_rejects_a_too_short_task() -> None:
    with pytest.raises(ValidationError):
        slugify_name("short")  # below TaskBrief's min_length


def test_agent_key_rejects_invalid_via_round_trip(tmp_path: Path) -> None:
    entry = _pool(tmp_path).save(_AGENT_KEY, _spec())
    payload = entry.model_dump_json()
    assert PoolEntry.model_validate_json(payload) == entry

    with pytest.raises(ValidationError):
        PoolEntry(
            agent_key="bad id with spaces",
            name=entry.name,
            spec=entry.spec,
            session_id=entry.session_id,
            cwd=entry.cwd,
            created_at=entry.created_at,
            updated_at=entry.updated_at,
        )


def test_session_id_rejects_non_uuid_via_round_trip(tmp_path: Path) -> None:
    entry = _pool(tmp_path).save(_AGENT_KEY, _spec())
    with pytest.raises(ValidationError):
        PoolEntry(
            agent_key=entry.agent_key,
            name=entry.name,
            spec=entry.spec,
            session_id="not-a-uuid",
            cwd=entry.cwd,
            created_at=entry.created_at,
            updated_at=entry.updated_at,
        )


def test_to_new_run_options_pins_session_and_matches_to_options(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    entry = pool.save(_AGENT_KEY, _spec())
    options = pool.to_new_run_options(entry)
    baseline = to_options(entry.spec)

    assert options.session_id == entry.session_id
    assert options.resume is None
    assert options.system_prompt == baseline.system_prompt  # built ON to_options
    assert options.allowed_tools == baseline.allowed_tools


def test_to_resume_options_sets_resume_and_leaves_session_none(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    entry = pool.save(_AGENT_KEY, _spec())
    options = pool.to_resume_options(entry)
    assert options.resume == entry.session_id
    assert options.session_id is None


def test_storage_persists_across_pool_instances(tmp_path: Path) -> None:
    db_path = tmp_path / "pool.db"
    saved = AgentPool(db_path).save(_AGENT_KEY, _spec())
    reopened = AgentPool(db_path).get_by_key(_AGENT_KEY)
    assert reopened == saved


def test_create_agent_auto_slugs_name_when_omitted(tmp_path: Path, catalog: Catalog) -> None:
    source = InMemoryCatalogSource(catalog)
    pool = _pool(tmp_path)
    request = ProblemRequest(
        task="Summarize the git commit history into grouped changelog entries",
    )
    entry = create_agent("INC-9", request, source, pool)

    assert entry.agent_key == "INC-9"
    assert entry.name == slugify_name(request.task)
    assert entry.name  # a meaningful label, not empty or the hardcoded default
    assert uuid.UUID(entry.session_id)
    assert pool.get_by_key("INC-9") == entry


def test_create_agent_keeps_explicit_name(tmp_path: Path, catalog: Catalog) -> None:
    source = InMemoryCatalogSource(catalog)
    pool = _pool(tmp_path)
    request = ProblemRequest(
        task="Summarize the git commit history into grouped changelog entries",
        name="changelog-curator",
    )
    entry = create_agent("INC-10", request, source, pool)
    assert entry.name == "changelog-curator"


def test_async_save_round_trips_and_matches_sync_save(tmp_path: Path) -> None:
    async_pool = AsyncAgentPool(_pool(tmp_path))
    sync_pool = AgentPool(tmp_path / "sync.db")

    saved = asyncio.run(async_pool.save(_AGENT_KEY, _spec()))
    baseline = sync_pool.save(_AGENT_KEY, _spec())

    assert saved.agent_key == baseline.agent_key
    assert saved.name == baseline.name
    assert saved.spec == baseline.spec
    assert asyncio.run(async_pool.get_by_key(_AGENT_KEY)) == saved


def test_async_save_reset_session_mints_new_uuid(tmp_path: Path) -> None:
    async_pool = AsyncAgentPool(_pool(tmp_path))
    first = asyncio.run(async_pool.save(_AGENT_KEY, _spec()))
    reset = asyncio.run(async_pool.save(_AGENT_KEY, _spec(), reset_session=True))
    assert reset.session_id != first.session_id


def test_async_get_by_key_returns_none_for_unknown_id(tmp_path: Path) -> None:
    async_pool = AsyncAgentPool(_pool(tmp_path))
    assert asyncio.run(async_pool.get_by_key("nonexistent")) is None


def test_async_list_and_delete_through_wrapper(tmp_path: Path) -> None:
    async_pool = AsyncAgentPool(_pool(tmp_path))
    asyncio.run(async_pool.save("PROJ-A", _spec()))
    asyncio.run(async_pool.save("PROJ-B", _spec(name="reviewer", description="Reviews PRs.")))

    ids = {entry.agent_key for entry in asyncio.run(async_pool.list())}
    assert ids == {"PROJ-A", "PROJ-B"}

    assert asyncio.run(async_pool.delete("PROJ-A")) is True
    assert asyncio.run(async_pool.delete("PROJ-A")) is False
    assert {e.agent_key for e in asyncio.run(async_pool.list())} == {"PROJ-B"}


def test_async_find_ranks_the_closest_description_first(tmp_path: Path) -> None:
    async_pool = AsyncAgentPool(_pool(tmp_path))
    asyncio.run(async_pool.save("P1", _spec(description="Audits code for security holes.")))
    asyncio.run(async_pool.save("P2", _spec(description="Summarizes git commits into changelog.")))
    asyncio.run(async_pool.save("P3", _spec(description="Runs the pytest suite and reports.")))

    results = asyncio.run(async_pool.find("summarize git commit history into a changelog"))
    assert results[0].agent_key == "P2"


def test_async_option_builders_are_plain_sync_methods(tmp_path: Path) -> None:
    sync_pool = _pool(tmp_path)
    async_pool = AsyncAgentPool(sync_pool)
    entry = sync_pool.save(_AGENT_KEY, _spec())

    assert not inspect.iscoroutinefunction(async_pool.to_new_run_options)
    assert not inspect.iscoroutinefunction(async_pool.to_resume_options)
    assert inspect.iscoroutinefunction(async_pool.save)
    assert inspect.iscoroutinefunction(async_pool.get_by_key)
    assert inspect.iscoroutinefunction(async_pool.list)
    assert inspect.iscoroutinefunction(async_pool.delete)
    assert inspect.iscoroutinefunction(async_pool.find)

    new_opts = async_pool.to_new_run_options(entry)
    resume_opts = async_pool.to_resume_options(entry)
    assert new_opts == sync_pool.to_new_run_options(entry)
    assert resume_opts == sync_pool.to_resume_options(entry)


def test_save_defaults_cwd_to_process_cwd_on_first_insert(tmp_path: Path) -> None:
    entry = _pool(tmp_path).save(_AGENT_KEY, _spec())
    assert entry.cwd == Path.cwd()


def test_save_pins_explicit_cwd(tmp_path: Path) -> None:
    entry = _pool(tmp_path).save(_AGENT_KEY, _spec(), cwd=tmp_path)
    assert entry.cwd == tmp_path


def test_resave_preserves_cwd_unless_overridden(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    first = pool.save(_AGENT_KEY, _spec(), cwd=tmp_path / "a")
    preserved = pool.save(_AGENT_KEY, _spec())
    assert preserved.cwd == tmp_path / "a"  # overwrite keeps the stored cwd
    moved = pool.save(_AGENT_KEY, _spec(), cwd=tmp_path / "b")
    assert moved.cwd == tmp_path / "b"  # an explicit cwd moves the entry
    assert first.cwd == tmp_path / "a"


def test_option_builders_carry_entry_cwd(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    entry = pool.save(_AGENT_KEY, _spec(), cwd=tmp_path)
    assert pool.to_new_run_options(entry).cwd == tmp_path
    assert pool.to_resume_options(entry).cwd == tmp_path


def test_reconcile_session_updates_only_session_and_preserves_rest(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    original = pool.save(_AGENT_KEY, _spec(), cwd=tmp_path)
    observed = str(uuid.uuid4())
    reconciled = pool.reconcile_session(_AGENT_KEY, observed)

    assert reconciled.session_id == observed
    assert reconciled.session_id != original.session_id
    assert reconciled.spec == original.spec
    assert reconciled.cwd == original.cwd
    assert reconciled.created_at == original.created_at
    assert reconciled.updated_at >= original.updated_at
    assert pool.get_by_key(_AGENT_KEY) == reconciled


def test_reconcile_session_raises_for_unknown_agent_key(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        _pool(tmp_path).reconcile_session("never-saved", str(uuid.uuid4()))


def test_start_and_finish_run_stamps_finished_at(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    pool.save(_AGENT_KEY, _spec())
    run = pool.start_run(_AGENT_KEY, _TASK)
    assert run.finished_at is None
    assert uuid.UUID(run.run_id)

    finished = pool.finish_run(run.run_id)
    assert finished.run_id == run.run_id
    assert finished.finished_at is not None
    assert finished.finished_at >= run.started_at


def test_finish_run_raises_for_unknown_run(tmp_path: Path) -> None:
    with pytest.raises(KeyError):
        _pool(tmp_path).finish_run(str(uuid.uuid4()))


def test_record_agent_run_main_and_dispatched(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    pool.save(_AGENT_KEY, _spec())
    run = pool.start_run(_AGENT_KEY, _TASK)

    main = pool.record_agent_run(run.run_id, str(uuid.uuid4()))
    assert main.tool_use_id is None
    assert main.agent_name is None

    dispatched = pool.record_agent_run(
        run.run_id, str(uuid.uuid4()), tool_use_id="toolu_1", agent_name="reviewer"
    )
    assert dispatched.tool_use_id == "toolu_1"
    assert dispatched.agent_name == "reviewer"

    rows = pool.list_agent_runs(run.run_id)
    assert [r.tool_use_id for r in rows] == [None, "toolu_1"]  # main row first


def test_get_run_round_trips_and_returns_none_for_unknown(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    pool.save(_AGENT_KEY, _spec())
    run = pool.start_run(_AGENT_KEY, _TASK)
    assert pool.get_run(run.run_id) == run
    assert pool.get_run(str(uuid.uuid4())) is None


def test_async_get_run_round_trips_and_returns_none_for_unknown(tmp_path: Path) -> None:
    async_pool = AsyncAgentPool(_pool(tmp_path))

    async def scenario() -> None:
        await async_pool.save(_AGENT_KEY, _spec())
        run = await async_pool.start_run(_AGENT_KEY, _TASK)
        assert await async_pool.get_run(run.run_id) == run
        assert await async_pool.get_run(str(uuid.uuid4())) is None

    asyncio.run(scenario())


def test_list_runs_orders_most_recent_first(tmp_path: Path) -> None:
    pool = _pool(tmp_path)
    pool.save(_AGENT_KEY, _spec())
    first = pool.start_run(_AGENT_KEY, _TASK)
    second = pool.start_run(_AGENT_KEY, _TASK)
    runs = pool.list_runs(_AGENT_KEY)
    assert [r.run_id for r in runs] == [second.run_id, first.run_id]


def test_async_reconcile_and_run_tracking(tmp_path: Path) -> None:
    async_pool = AsyncAgentPool(_pool(tmp_path))

    async def scenario() -> None:
        await async_pool.save(_AGENT_KEY, _spec(), cwd=tmp_path)
        observed = str(uuid.uuid4())
        reconciled = await async_pool.reconcile_session(_AGENT_KEY, observed)
        assert reconciled.session_id == observed

        run = await async_pool.start_run(_AGENT_KEY, _TASK)
        await async_pool.record_agent_run(run.run_id, str(uuid.uuid4()))
        await async_pool.record_agent_run(
            run.run_id, str(uuid.uuid4()), tool_use_id="toolu_9", agent_name="reviewer"
        )
        finished = await async_pool.finish_run(run.run_id)
        assert finished.finished_at is not None

        assert [r.run_id for r in await async_pool.list_runs(_AGENT_KEY)] == [run.run_id]
        agent_runs = await async_pool.list_agent_runs(run.run_id)
        assert [r.tool_use_id for r in agent_runs] == [None, "toolu_9"]

    asyncio.run(scenario())


def test_async_save_does_not_block_the_event_loop(tmp_path: Path) -> None:
    async_pool = AsyncAgentPool(_pool(tmp_path))

    async def scenario() -> int:
        ticks = 0

        async def tick() -> None:
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0)

        ticker = asyncio.create_task(tick())
        for i in range(20):
            await async_pool.save(f"P{i}", _spec())
        ticker.cancel()
        return ticks

    ticks = asyncio.run(scenario())
    assert ticks > 0  # the ticker interleaved: save() yielded the loop via the thread
