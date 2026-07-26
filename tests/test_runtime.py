from __future__ import annotations

from agent_fleet.runtime import RuntimeAgentPool, RuntimeAgentSpec


def test_runtime_pool_pins_then_resumes_session(tmp_path) -> None:
    pool = RuntimeAgentPool(tmp_path / "fleet.db")
    try:
        spec = RuntimeAgentSpec(
            name="validator",
            description="Fixed validator",
            system_prompt="Validate the request.",
            permission_mode="dontAsk",
            max_turns=4,
        )
        entry = pool.save("fr-test-validator", spec, cwd=tmp_path, reset_session=True)

        first_run, first_options = pool.begin_run(entry, "first task")
        assert first_options.session_id == entry.session_id
        assert first_options.resume is None
        pool.finish_run(first_run.run_id)

        second_run, second_options = pool.begin_run(entry, "second task")
        assert second_options.session_id is None
        assert second_options.resume == entry.session_id
        pool.finish_run(second_run.run_id)
    finally:
        pool.close()
