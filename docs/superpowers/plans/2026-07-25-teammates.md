# Teammates Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Name-addressed, templated, background-run teammates as the primary MCP surface over the existing agent pool.

**Architecture:** A hardcoded roster of `TeammateTemplate`s (capability bundles via `Toolkit`s) resolves to deterministic pool keys (`teammate.{name}`), reusing `create_agent`/`prepare_run`/`run_with_capture` unchanged beneath. Background execution is an in-process `asyncio.Task` registry in the MCP server; run outcomes are persisted onto the `runs` table so results survive the call. Status is derived (`finished_at` + live registry), never stored.

**Tech Stack:** Python 3.12+, Pydantic v2, FastMCP, claude-agent-sdk, SQLite, pytest.

**Spec:** [2026-07-25-teammates-design.md](../specs/2026-07-25-teammates-design.md)

**Repo conventions that bite:**
- The git hook blocks `git commit -m`; write the message to a file and use `git commit -F <file>`.
- No `import json` — `TypeAdapter`/`model_dump_json` only.
- Every new domain value gets an `Annotated` alias in `src/agent_fleet/models/agent/types.py`.
- Gates: `uv run pytest`, `uv run ruff check`, `uv run mypy src` — run from the repo root.

---

### Task 1: Domain aliases (`ToolkitName`, `RunOutput`, `CostUsd`)

**Files:**
- Modify: `src/agent_fleet/models/agent/types.py` (append after `TeamSlug`, before the `DEFAULT_*` constants)
- Modify: `src/agent_fleet/models/agent/__init__.py` (add to the `.types` import block and `__all__`, keeping alphabetical order)
- Test: `tests/test_teammate_models.py` (new)

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from agent_fleet.models.agent import CostUsd, RunOutput, ToolkitName

_TOOLKIT_NAME = TypeAdapter(ToolkitName)
_COST = TypeAdapter(CostUsd)
_OUTPUT = TypeAdapter(RunOutput)


def test_toolkit_name_accepts_slug_and_rejects_uppercase() -> None:
    assert _TOOLKIT_NAME.validate_python("pydantic-review") == "pydantic-review"
    with pytest.raises(ValidationError):
        _TOOLKIT_NAME.validate_python("Pydantic Review")


def test_cost_rejects_negative() -> None:
    assert _COST.validate_python(0.42) == 0.42
    with pytest.raises(ValidationError):
        _COST.validate_python(-0.01)


def test_run_output_accepts_text() -> None:
    assert _OUTPUT.validate_python("done") == "done"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_teammate_models.py -v`
Expected: FAIL — `ImportError: cannot import name 'CostUsd'`

- [ ] **Step 3: Add the aliases to `types.py`**

```python
ToolkitName = Annotated[
    str,
    Field(
        pattern=r"^[a-z0-9][a-z0-9-]{0,63}$",
        title="Toolkit name",
        description="Name of a hardcoded capability bundle a teammate template pins.",
        examples=["pydantic-review"],
    ),
]
RunOutput = Annotated[
    str,
    Field(
        title="Run output",
        description="The collected assistant text of a finished run, persisted on its record.",
    ),
]
CostUsd = Annotated[
    float,
    Field(
        ge=0,
        title="Cost (USD)",
        description="Total cost of a run in US dollars, from the terminal result message.",
        examples=[0.42],
    ),
]
```

- [ ] **Step 4: Re-export from `models/agent/__init__.py`** — add `CostUsd`, `RunOutput`, `ToolkitName` to the `.types` import and to `__all__` (alphabetical).

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_teammate_models.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
cd ~/workplace/claude-sdk-agent-fleet
printf 'feat(models): add ToolkitName, RunOutput, CostUsd aliases\n' > /tmp/cmsg.txt
git add src/agent_fleet/models/agent/types.py src/agent_fleet/models/agent/__init__.py tests/test_teammate_models.py
git commit -F /tmp/cmsg.txt
```

---

### Task 2: Teammate models

**Files:**
- Create: `src/agent_fleet/models/agent/teammate.py`
- Modify: `src/agent_fleet/models/agent/__init__.py` (import + `__all__`: `RosterEntry`, `TEAMMATE_KEY_PREFIX`, `TeammateRunStatus`, `TeammateStatus`, `TeammateTemplate`, `Toolkit`, `teammate_key`)
- Test: `tests/test_teammate_models.py` (append)

- [ ] **Step 1: Append failing tests**

```python
from pydantic import TypeAdapter, ValidationError

from agent_fleet.models.agent import (
    AgentKey,
    TeammateRunStatus,
    TeammateTemplate,
    Toolkit,
    teammate_key,
)

_AGENT_KEY = TypeAdapter(AgentKey)


def test_teammate_key_is_a_valid_agent_key() -> None:
    key = teammate_key("reviewer")
    assert key == "teammate.reviewer"
    assert _AGENT_KEY.validate_python(key) == key


def test_template_defaults() -> None:
    template = TeammateTemplate(
        name="reviewer",
        brief="Review code changes for correctness and regressions.",
    )
    assert template.toolkits == []
    assert template.model.value == "inherit"


def test_toolkit_requires_valid_entries() -> None:
    kit = Toolkit(name="pydantic-review", entries=["skill-pydantic-type-discipline"])
    assert kit.entries == ["skill-pydantic-type-discipline"]


def test_status_enum_members() -> None:
    assert {s.value for s in TeammateRunStatus} == {
        "unspawned", "idle", "running", "finished", "stale",
    }
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_teammate_models.py -v`
Expected: FAIL — `ImportError: cannot import name 'TeammateTemplate'`

- [ ] **Step 3: Create `teammate.py`**

```python
from __future__ import annotations

from enum import StrEnum

from pydantic import JsonValue

from capdisc.base import FrozenModel
from capdisc.catalog import CatalogEntryId, Tag

from .types import (
    AgentKey,
    AgentName,
    CostUsd,
    ModelId,
    PromptBody,
    RunId,
    RunOutput,
    SessionId,
    TaskBrief,
    ToolkitName,
)


class TeammateRunStatus(StrEnum):
    """Derived state of a teammate's latest run — never stored, so it cannot lie after a crash:
    `stale` means an unfinished run unknown to the live registry (the session itself is intact)."""

    unspawned = "unspawned"
    idle = "idle"
    running = "running"
    finished = "finished"
    stale = "stale"


class Toolkit(FrozenModel):
    """A named capability bundle a template pins — the dedicated skills of a teammate."""

    name: ToolkitName
    entries: list[CatalogEntryId]


class TeammateTemplate(FrozenModel):
    """One hardcoded roster entry: the name a teammate is addressed by and what it is built from."""

    name: AgentName
    brief: TaskBrief
    toolkits: list[ToolkitName] = []
    tags: list[Tag] = []
    model: ModelId = ModelId.inherit
    system_prompt: PromptBody | None = None


class TeammateStatus(FrozenModel):
    """What `spawn_teammate`/`check_teammate`/`message_teammate` report: the derived status of the
    teammate's latest run plus its persisted outcome once finished."""

    name: AgentName
    agent_key: AgentKey
    status: TeammateRunStatus
    run_id: RunId | None = None
    session_id: SessionId | None = None
    output: RunOutput | None = None
    structured_output: JsonValue | None = None
    total_cost_usd: CostUsd | None = None


class RosterEntry(FrozenModel):
    """One `roster()` row: the template plus the live pool state of its teammate."""

    template: TeammateTemplate
    status: TeammateRunStatus
    session_id: SessionId | None = None


TEAMMATE_KEY_PREFIX = "teammate."


def teammate_key(name: AgentName) -> AgentKey:
    """The deterministic pool key for a teammate — name-based addressing is key derivation."""
    return f"{TEAMMATE_KEY_PREFIX}{name}"
```

- [ ] **Step 4: Wire the exports in `__init__.py`**, run `uv run pytest tests/test_teammate_models.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
printf 'feat(models): teammate templates, toolkits, status models\n' > /tmp/cmsg.txt
git add src/agent_fleet/models/agent/teammate.py src/agent_fleet/models/agent/__init__.py tests/test_teammate_models.py
git commit -F /tmp/cmsg.txt
```

---

### Task 3: Roster module and template resolution

**Files:**
- Create: `src/agent_fleet/engine/teammates.py`
- Test: `tests/test_teammates.py` (new)

- [ ] **Step 1: Write failing tests**

```python
from __future__ import annotations

import pytest

from agent_fleet.engine.teammates import ROSTER, resolve_template, template_request
from agent_fleet.models.agent import TeammateTemplate, Toolkit


def test_resolve_known_template() -> None:
    template = resolve_template(ROSTER[0].name)
    assert template is ROSTER[0]


def test_resolve_unknown_lists_roster() -> None:
    with pytest.raises(ValueError, match=ROSTER[0].name):
        resolve_template("nobody-here")


def test_template_request_maps_fields() -> None:
    template = ROSTER[0]
    request = template_request(template)
    assert request.task == template.brief
    assert request.name == template.name
    assert request.model == template.model


def test_template_request_flattens_toolkits() -> None:
    kits = (
        Toolkit(name="kit-a", entries=["skill-alpha", "tool-beta"]),
        Toolkit(name="kit-b", entries=["skill-gamma"]),
    )
    template = TeammateTemplate(
        name="specialist",
        brief="Exercise toolkit flattening in template resolution.",
        toolkits=["kit-a", "kit-b"],
    )
    request = template_request(template, toolkits=kits)
    assert request.pinned == ["skill-alpha", "tool-beta", "skill-gamma"]


def test_template_with_dangling_toolkit_raises() -> None:
    template = TeammateTemplate(
        name="specialist",
        brief="Exercise the dangling toolkit reference error.",
        toolkits=["kit-missing"],
    )
    with pytest.raises(ValueError, match="kit-missing"):
        template_request(template, toolkits=())
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_teammates.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent_fleet.engine.teammates'`

- [ ] **Step 3: Create `engine/teammates.py`**

The shipped `ROSTER` is the mechanism plus two starter teammates; the owner edits these tuples —
they are configuration, and toolkit entries are environment-specific `CatalogEntryId`s.

```python
from __future__ import annotations

from collections.abc import Sequence

from capdisc.catalog import CatalogEntryId

from ..models.agent import (
    AgentName,
    ProblemRequest,
    TeammateTemplate,
    Toolkit,
    ToolkitName,
)

TOOLKITS: tuple[Toolkit, ...] = ()

ROSTER: tuple[TeammateTemplate, ...] = (
    TeammateTemplate(
        name="researcher",
        brief=(
            "Research questions against the local code and documentation indexes and report "
            "cited findings."
        ),
    ),
    TeammateTemplate(
        name="reviewer",
        brief=(
            "Review code changes for correctness, regressions, and adherence to project "
            "conventions."
        ),
    ),
)

_TEMPLATE_BY_NAME: dict[AgentName, TeammateTemplate] = {t.name: t for t in ROSTER}
if len(_TEMPLATE_BY_NAME) != len(ROSTER):
    raise ValueError("duplicate teammate names in ROSTER")


def resolve_template(name: AgentName) -> TeammateTemplate:
    """The roster template addressed by `name`.

    Raises:
        ValueError: When `name` is not on the roster; the message lists the valid names.
    """
    template = _TEMPLATE_BY_NAME.get(name)
    if template is None:
        known = ", ".join(sorted(_TEMPLATE_BY_NAME))
        raise ValueError(f"unknown teammate {name!r} (roster: {known})")
    return template


def template_request(
    template: TeammateTemplate, toolkits: Sequence[Toolkit] = TOOLKITS
) -> ProblemRequest:
    """Translate a template into the `ProblemRequest` the existing create path consumes,
    flattening its toolkits into pinned capability ids.

    Raises:
        ValueError: When the template references a toolkit name absent from `toolkits`.
    """
    by_name: dict[ToolkitName, Toolkit] = {kit.name: kit for kit in toolkits}
    pinned: list[CatalogEntryId] = []
    for kit_name in template.toolkits:
        kit = by_name.get(kit_name)
        if kit is None:
            raise ValueError(f"template {template.name!r} references unknown toolkit {kit_name!r}")
        pinned.extend(kit.entries)
    return ProblemRequest(
        task=template.brief,
        name=template.name,
        tags=template.tags,
        model=template.model,
        pinned=pinned,
        system_prompt=template.system_prompt,
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_teammates.py tests/test_teammate_models.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
printf 'feat(engine): teammate roster and template resolution\n' > /tmp/cmsg.txt
git add src/agent_fleet/engine/teammates.py tests/test_teammates.py
git commit -F /tmp/cmsg.txt
```

---

### Task 4: Persist run outcomes in the pool

**Files:**
- Modify: `src/agent_fleet/models/agent/pool.py` (`RunRecord`)
- Modify: `src/agent_fleet/engine/pool.py` (schema migration, `finish_run`, `_row_to_run`, run SELECTs, `AsyncAgentPool.finish_run`)
- Test: `tests/test_pool.py` (append)

- [ ] **Step 1: Write failing tests** (append to `tests/test_pool.py`, matching its existing fixture style — it builds `AgentPool(tmp_path / "pool.db")` directly)

```python
def test_finish_run_persists_outcome(tmp_path: Path) -> None:
    pool = AgentPool(tmp_path / "pool.db")
    entry = pool.save("PROJ-OUT", _spec())  # reuse the module's existing spec helper
    run = pool.start_run(entry.agent_key, "record the outcome of this run for later reading")
    finished = pool.finish_run(
        run.run_id,
        output="all done",
        structured_output={"ok": True},
        total_cost_usd=0.07,
    )
    assert finished.output == "all done"
    assert finished.structured_output == {"ok": True}
    assert finished.total_cost_usd == 0.07
    assert pool.get_run(run.run_id) == finished


def test_finish_run_without_outcome_stays_none(tmp_path: Path) -> None:
    pool = AgentPool(tmp_path / "pool.db")
    entry = pool.save("PROJ-NONE", _spec())
    run = pool.start_run(entry.agent_key, "finish this run with no outcome payload at all")
    finished = pool.finish_run(run.run_id)
    assert finished.output is None
    assert finished.structured_output is None
    assert finished.total_cost_usd is None


def test_opening_a_pre_outcome_schema_migrates_it(tmp_path: Path) -> None:
    db = tmp_path / "pool.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE runs (run_id TEXT PRIMARY KEY, agent_key TEXT NOT NULL, "
        "task TEXT NOT NULL, started_at TEXT NOT NULL, finished_at TEXT)"
    )
    run_id = "7c9e6679-7425-40de-944b-e07fc1f90ae7"
    conn.execute(
        "INSERT INTO runs (run_id, agent_key, task, started_at) VALUES (?, ?, ?, ?)",
        (run_id, "PROJ-MIG", "a run recorded before the outcome columns existed", "2026-07-01T00:00:00+00:00"),
    )
    conn.commit()
    conn.close()

    pool = AgentPool(db)  # opening must ALTER the old table, preserving the row
    migrated = pool.get_run(run_id)
    assert migrated is not None
    assert migrated.output is None
    pool.finish_run(run_id, output="works after migration")
    reread = pool.get_run(run_id)
    assert reread is not None and reread.output == "works after migration"
```

Note: reuse the module's existing `_spec()` helper (defined near the top of `tests/test_pool.py`)
for the first two tests; `import sqlite3` for the migration test.

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_pool.py -v -k outcome`
Expected: FAIL — `TypeError: finish_run() got an unexpected keyword argument 'output'`

- [ ] **Step 3: Extend `RunRecord`** in `src/agent_fleet/models/agent/pool.py` — add after `finished_at`:

```python
    output: RunOutput | None = None
    structured_output: JsonValue | None = None
    total_cost_usd: CostUsd | None = None
```

Extend the module's `.types` import with `CostUsd, RunOutput` (JsonValue is already imported).
Update the class docstring's last sentence to mention the outcome fields staying None while in
flight or when never captured.

- [ ] **Step 4: Migrate the schema and readers in `src/agent_fleet/engine/pool.py`**

Add `from pydantic import JsonValue, TypeAdapter` to the imports, then near the other
`_CREATE_*` constants:

```python
_RUN_COLUMNS = (
    "run_id, agent_key, task, started_at, finished_at, "
    "output, structured_output_json, total_cost_usd"
)
_RUNS_OUTCOME_COLUMNS = (
    ("output", "TEXT"),
    ("structured_output_json", "TEXT"),
    ("total_cost_usd", "REAL"),
)
_JSON_ADAPTER: TypeAdapter[JsonValue] = TypeAdapter(JsonValue)
```

(`TypeAdapter` and `JsonValue` come from `pydantic`; extend the existing import.)

In `AgentPool.__init__`, after the four `_CREATE_*` executes and before `commit()`:

```python
        self._ensure_runs_columns()
```

Add the method:

```python
    def _ensure_runs_columns(self) -> None:
        """Add the run-outcome columns to a database created before they existed (idempotent)."""
        existing = {row["name"] for row in self._conn.execute("PRAGMA table_info(runs)")}
        for column, sql_type in _RUNS_OUTCOME_COLUMNS:
            if column not in existing:
                self._conn.execute(f"ALTER TABLE runs ADD COLUMN {column} {sql_type}")
```

Replace the hardcoded `"SELECT run_id, agent_key, task, started_at, finished_at "` column list in
`finish_run`, `get_run`, and `_list_runs_locked` with `f"SELECT {_RUN_COLUMNS} "`.

Replace `finish_run`'s signature and UPDATE:

```python
    def finish_run(
        self,
        run_id: RunId,
        *,
        output: RunOutput | None = None,
        structured_output: JsonValue | None = None,
        total_cost_usd: CostUsd | None = None,
    ) -> RunRecord:
        """Stamp `finished_at` (and any captured outcome) on the run, returning the updated record.

        Raises:
            KeyError: When `run_id` is unknown.
        """
        now = datetime.now(UTC)
        structured_json = (
            None
            if structured_output is None
            else _JSON_ADAPTER.dump_json(structured_output).decode()
        )
        with self._write_lock:
            cursor = self._conn.execute(
                "UPDATE runs SET finished_at = ?, output = ?, "
                "structured_output_json = ?, total_cost_usd = ? WHERE run_id = ?",
                (now.isoformat(), output, structured_json, total_cost_usd, run_id),
            )
            self._conn.commit()
            if cursor.rowcount == 0:
                raise KeyError(run_id)
            row = self._conn.execute(
                f"SELECT {_RUN_COLUMNS} FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return _row_to_run(row)
```

Extend `_row_to_run`:

```python
def _row_to_run(row: sqlite3.Row) -> RunRecord:
    """Rebuild a `RunRecord` from a database row; `finished_at` and the outcome fields stay None
    while a run is open (or when it finished without a captured outcome)."""
    finished_at = row["finished_at"]
    structured_json = row["structured_output_json"]
    return RunRecord(
        run_id=row["run_id"],
        agent_key=row["agent_key"],
        task=row["task"],
        started_at=datetime.fromisoformat(row["started_at"]),
        finished_at=datetime.fromisoformat(finished_at) if finished_at is not None else None,
        output=row["output"],
        structured_output=(
            _JSON_ADAPTER.validate_json(structured_json) if structured_json is not None else None
        ),
        total_cost_usd=row["total_cost_usd"],
    )
```

Update `AsyncAgentPool.finish_run` (currently `await asyncio.to_thread(self._pool.finish_run, run_id)`):

```python
    async def finish_run(
        self,
        run_id: RunId,
        *,
        output: RunOutput | None = None,
        structured_output: JsonValue | None = None,
        total_cost_usd: CostUsd | None = None,
    ) -> RunRecord:
        """Stamp `finished_at` (and any captured outcome) on the run.

        Delegates to `AgentPool.finish_run` via `asyncio.to_thread`.
        """
        return await asyncio.to_thread(
            self._pool.finish_run,
            run_id,
            output=output,
            structured_output=structured_output,
            total_cost_usd=total_cost_usd,
        )
```

Extend the engine module's model imports with `CostUsd, RunOutput` as needed.

- [ ] **Step 5: Run the pool tests**

Run: `uv run pytest tests/test_pool.py -v`
Expected: PASS (all — new and pre-existing)

- [ ] **Step 6: Commit**

```bash
printf 'feat(pool): persist run outcomes on the runs table\n' > /tmp/cmsg.txt
git add src/agent_fleet/models/agent/pool.py src/agent_fleet/engine/pool.py tests/test_pool.py
git commit -F /tmp/cmsg.txt
```

---

### Task 5: `run_with_capture` writes the outcome through `finish_run`

**Files:**
- Modify: `src/agent_fleet/engine/dispatch.py` (end of `run_with_capture`)
- Test: `tests/test_dispatch.py` (append)

- [ ] **Step 1: Write the failing test** (reuse the module's `_assistant`, `_pool`, `_spec`, `_fake_query` helpers)

```python
def test_finished_run_row_carries_the_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool = _pool(tmp_path)
    entry = pool.save("PROJ-PERSIST", _spec())
    messages = [_assistant("captured text", session_id=entry.session_id)]
    monkeypatch.setattr("agent_fleet.engine.dispatch.query", _fake_query(messages))

    outcome = asyncio.run(
        run_with_capture(pool, entry.agent_key, _TASK, pool.to_new_run_options(entry))
    )
    stored = pool.get_run(outcome.run.run_id)
    assert stored is not None
    assert stored.output == "captured text"
```

(Adapt the options-building call to whatever the surrounding tests in the file already use.)

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_dispatch.py -v -k carries`
Expected: FAIL — `stored.output` is None

- [ ] **Step 3: Change the tail of `run_with_capture`** — replace

```python
    finished = pool.finish_run(run.run_id)
```

with

```python
    finished = pool.finish_run(
        run.run_id,
        output="\n".join(parts),
        structured_output=structured_output,
        total_cost_usd=total_cost_usd,
    )
```

(The `RunOutcome` construction below it is unchanged.)

- [ ] **Step 4: Run dispatch + pool-server tests**

Run: `uv run pytest tests/test_dispatch.py tests/mcp/test_pool_server.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
printf 'feat(dispatch): persist captured outcome when finishing a run\n' > /tmp/cmsg.txt
git add src/agent_fleet/engine/dispatch.py tests/test_dispatch.py
git commit -F /tmp/cmsg.txt
```

---

### Task 6: `TeammateRunner` — the background-task registry

**Files:**
- Create: `src/agent_fleet_mcp/runner.py`
- Test: `tests/mcp/test_runner.py` (new)

- [ ] **Step 1: Write failing tests**

```python
from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from agent_fleet.models.agent import RunRecord, TeammateRunStatus
from agent_fleet_mcp.runner import TeammateRunner, run_status

_RUN_ID = "7c9e6679-7425-40de-944b-e07fc1f90ae7"


def _open_run() -> RunRecord:
    return RunRecord(
        run_id=_RUN_ID,
        agent_key="teammate.reviewer",
        task="an unfinished run for status derivation tests",
        started_at=datetime.now(UTC),
    )


def test_status_of_no_run_is_idle() -> None:
    assert run_status(None, TeammateRunner()) is TeammateRunStatus.idle


def test_status_of_finished_run() -> None:
    run = _open_run().model_copy(update={"finished_at": datetime.now(UTC)})
    assert run_status(run, TeammateRunner()) is TeammateRunStatus.finished


def test_status_running_while_registered_then_stale_after() -> None:
    async def scenario() -> None:
        runner = TeammateRunner()
        release = asyncio.Event()

        async def job() -> str:
            await release.wait()
            return "done"

        runner.spawn(_RUN_ID, job())
        assert run_status(_open_run(), runner) is TeammateRunStatus.running
        release.set()
        await runner.wait_all()
        # task completed and deregistered; the run row was never stamped -> stale
        assert run_status(_open_run(), runner) is TeammateRunStatus.stale

    asyncio.run(scenario())


def test_failing_task_deregisters_without_raising() -> None:
    async def scenario() -> None:
        runner = TeammateRunner()

        async def job() -> str:
            raise RuntimeError("boom")

        runner.spawn(_RUN_ID, job())
        await runner.wait_all()
        assert not runner.is_running(_RUN_ID)

    asyncio.run(scenario())
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/mcp/test_runner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent_fleet_mcp.runner'`

- [ ] **Step 3: Create `runner.py`**

```python
from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine

from agent_fleet.models.agent import RunId, RunRecord, TeammateRunStatus

logger = logging.getLogger(__name__)


class TeammateRunner:
    """In-process registry of live background runs, keyed by run id.

    Holds each spawned `asyncio.Task` until it completes; membership is what distinguishes a
    `running` run from a `stale` one (unfinished in the database, unknown here — the server died
    mid-run). Results are not read from the task: the run coroutine persists its own outcome via
    `finish_run`, so a completed task's value is already in the pool."""

    def __init__(self) -> None:
        self._tasks: dict[RunId, asyncio.Task[object]] = {}

    def spawn(self, run_id: RunId, coro: Coroutine[object, object, object]) -> None:
        """Start `coro` as a background task registered under `run_id`."""
        task = asyncio.get_running_loop().create_task(coro)
        self._tasks[run_id] = task
        task.add_done_callback(lambda finished: self._deregister(run_id, finished))

    def _deregister(self, run_id: RunId, task: asyncio.Task[object]) -> None:
        """Drop the finished task; a failure is logged, not raised (the run row stays unfinished,
        so `run_status` reports it `stale`)."""
        self._tasks.pop(run_id, None)
        if not task.cancelled() and task.exception() is not None:
            logger.error("teammate run %s failed", run_id, exc_info=task.exception())

    def is_running(self, run_id: RunId) -> bool:
        """Whether `run_id` is a currently-registered live task."""
        return run_id in self._tasks

    async def wait_all(self) -> None:
        """Wait for every registered task to complete (failures swallowed) — shutdown and tests."""
        await asyncio.gather(*list(self._tasks.values()), return_exceptions=True)


def run_status(run: RunRecord | None, runner: TeammateRunner) -> TeammateRunStatus:
    """Derive the status of a teammate's latest run; `None` (no runs yet) is `idle`."""
    if run is None:
        return TeammateRunStatus.idle
    if run.finished_at is not None:
        return TeammateRunStatus.finished
    return TeammateRunStatus.running if runner.is_running(run.run_id) else TeammateRunStatus.stale
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/mcp/test_runner.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
printf 'feat(mcp): TeammateRunner background-task registry with derived status\n' > /tmp/cmsg.txt
git add src/agent_fleet_mcp/runner.py tests/mcp/test_runner.py
git commit -F /tmp/cmsg.txt
```

---

### Task 7: Teammate MCP tools

**Files:**
- Modify: `src/agent_fleet_mcp/pool_server.py`
- Test: `tests/mcp/test_teammate_tools.py` (new)

- [ ] **Step 1: Write failing tests** (mirror the `pool` fixture in `tests/mcp/test_pool_server.py`, which monkeypatches `_pool`/`_source`/`_capability_router`; add `_runner`)

```python
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from claude_agent_sdk import Message

from agent_fleet import AgentPool
from agent_fleet.engine.source import InMemoryCatalogSource
from agent_fleet.engine.teammates import ROSTER
from agent_fleet.models.agent import TeammateRunStatus, teammate_key
from agent_fleet.router.capability import CapabilityRouter
from agent_fleet_mcp import pool_server
from agent_fleet_mcp.runner import TeammateRunner
from capdisc.catalog import Catalog
from test_dispatch import _assistant

_NAME = ROSTER[0].name
_TASK = "run the roster teammate against a synthetic task"


@pytest.fixture
def pool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AgentPool:
    built = AgentPool(tmp_path / "pool.db")
    runner = TeammateRunner()
    monkeypatch.setattr(pool_server, "_pool", lambda: built)
    monkeypatch.setattr(pool_server, "_source", lambda: InMemoryCatalogSource(Catalog(entries=[])))
    monkeypatch.setattr(
        pool_server, "_capability_router", lambda: CapabilityRouter([], [], [], [], {})
    )
    monkeypatch.setattr(pool_server, "_runner", lambda: runner)
    return built


def _fake_query(messages: list[Message]) -> object:
    async def _query(**kwargs: object):
        for message in messages:
            yield message

    return _query


def test_roster_lists_templates_unspawned(pool: AgentPool) -> None:
    entries = pool_server.roster()
    assert [e.template.name for e in entries] == [t.name for t in ROSTER]
    assert all(e.status is TeammateRunStatus.unspawned for e in entries)


def test_check_unspawned(pool: AgentPool) -> None:
    status = pool_server.check_teammate(_NAME)
    assert status.status is TeammateRunStatus.unspawned
    assert status.agent_key == teammate_key(_NAME)


def test_check_unknown_name_raises_with_roster(pool: AgentPool) -> None:
    with pytest.raises(ValueError, match=_NAME):
        pool_server.check_teammate("nobody-here")


def test_spawn_runs_in_background_and_persists(
    pool: AgentPool, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        release = asyncio.Event()

        async def gated(**kwargs: object):
            await release.wait()
            entry = pool.get_by_key(teammate_key(_NAME))
            assert entry is not None
            yield _assistant("background done", session_id=entry.session_id)

        monkeypatch.setattr("agent_fleet.engine.dispatch.query", lambda **kw: gated(**kw))

        status = await pool_server.spawn_teammate(_NAME, _TASK)
        assert status.status is TeammateRunStatus.running
        assert status.run_id is not None

        # spawning again while running returns the live run, not a second one
        again = await pool_server.spawn_teammate(_NAME, _TASK)
        assert again.run_id == status.run_id

        release.set()
        await pool_server._runner().wait_all()

        finished = pool_server.check_teammate(_NAME)
        assert finished.status is TeammateRunStatus.finished
        assert finished.output == "background done"

    asyncio.run(scenario())


def test_stale_after_registry_loss(pool: AgentPool) -> None:
    # create the entry through the template path, then open a run no registry knows about
    pool_server._ensure_teammate(_NAME)
    pool.start_run(teammate_key(_NAME), _TASK)
    status = pool_server.check_teammate(_NAME)
    assert status.status is TeammateRunStatus.stale


def test_message_wait_returns_finished_status(
    pool: AgentPool, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        pool_server._ensure_teammate(_NAME)
        entry = pool.get_by_key(teammate_key(_NAME))
        assert entry is not None
        monkeypatch.setattr(
            "agent_fleet.engine.dispatch.query",
            _fake_query([_assistant("sync reply", session_id=entry.session_id)]),
        )
        status = await pool_server.message_teammate(_NAME, _TASK, wait=True)
        assert status.status is TeammateRunStatus.finished
        assert status.output == "sync reply"
        # re-messaging revived the SAME standing session, not a fresh one
        revived = pool.get_by_key(teammate_key(_NAME))
        assert revived is not None and revived.session_id == entry.session_id

    asyncio.run(scenario())
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/mcp/test_teammate_tools.py -v`
Expected: FAIL — `AttributeError: module 'agent_fleet_mcp.pool_server' has no attribute 'roster'`

- [ ] **Step 3: Add the teammate surface to `pool_server.py`**

Extend the imports:

```python
from agent_fleet.engine.teammates import ROSTER, resolve_template, template_request
from agent_fleet.models.agent import (
    ...existing names...,
    RosterEntry,
    TeammateRunStatus,
    TeammateStatus,
    teammate_key,
)
from .runner import TeammateRunner, run_status
```

Add the cached runner and the shared internals:

```python
@cache
def _runner() -> TeammateRunner:
    """The process-wide registry of live background teammate runs, once, on first use."""
    return TeammateRunner()


def _ensure_teammate(name: AgentName, *, fresh_session: bool = False) -> PoolEntry:
    """The pool entry for teammate `name`, creating it from its roster template when absent.

    Raises:
        ValueError: When `name` is not on the roster.
    """
    template = resolve_template(name)
    key = teammate_key(name)
    entry = _pool().get_by_key(key)
    if entry is not None and not fresh_session:
        return entry
    return pool_create_agent(
        key, template_request(template), _source(), _pool(), reset_session=fresh_session
    )


def _teammate_status(name: AgentName, entry: PoolEntry | None) -> TeammateStatus:
    """The derived status of `name`'s latest run, with persisted outcome fields once finished."""
    key = teammate_key(name)
    if entry is None:
        return TeammateStatus(name=name, agent_key=key, status=TeammateRunStatus.unspawned)
    runs = _pool().list_runs(key)
    latest = runs[0] if runs else None
    return TeammateStatus(
        name=name,
        agent_key=key,
        status=run_status(latest, _runner()),
        run_id=latest.run_id if latest is not None else None,
        session_id=entry.session_id,
        output=latest.output if latest is not None else None,
        structured_output=latest.structured_output if latest is not None else None,
        total_cost_usd=latest.total_cost_usd if latest is not None else None,
    )
```

Add the four tools:

```python
@mcp.tool
def roster() -> list[RosterEntry]:
    """Every teammate template plus its live pool state — the teammate directory.

    Returns:
        One row per roster template, in roster order: the template, the derived status of its
        latest run (`unspawned` before first spawn), and its session id once it exists.
    """
    entries: list[RosterEntry] = []
    for template in ROSTER:
        entry = _pool().get_by_key(teammate_key(template.name))
        status = _teammate_status(template.name, entry)
        entries.append(
            RosterEntry(template=template, status=status.status, session_id=status.session_id)
        )
    return entries


@mcp.tool
def check_teammate(name: AgentName) -> TeammateStatus:
    """The teammate's latest-run status; persisted output/structured output/cost once finished.

    `stale` means an unfinished run this server process doesn't own (it died mid-run) — the
    session itself is intact, so `message_teammate` resumes the conversation.

    Raises:
        ValueError: When `name` is not on the roster.
    """
    resolve_template(name)
    return _teammate_status(name, _pool().get_by_key(teammate_key(name)))


@mcp.tool
async def spawn_teammate(
    name: AgentName, task: TaskBrief, fresh_session: bool = False
) -> TeammateStatus:
    """Stand the teammate up (creating its pool entry from the roster template when absent) and
    run `task` in the background, returning immediately.

    Spawning while the latest run is still live returns that run's status instead of opening a
    second concurrent run against the same session. `fresh_session=True` re-assembles the entry
    with a new session UUID first.

    Raises:
        ValueError: When `name` is not on the roster.
    """
    entry = _ensure_teammate(name, fresh_session=fresh_session)
    key = entry.agent_key
    runs = _pool().list_runs(key)
    if runs and run_status(runs[0], _runner()) is TeammateRunStatus.running:
        return _teammate_status(name, entry)
    run, options, prompt = prepare_run(_pool(), _capability_router(), key, task)
    _runner().spawn(
        run.run_id,
        run_with_capture(_pool(), key, task, options, run=run, prompt=prompt),
    )
    return _teammate_status(name, entry)


@mcp.tool
async def message_teammate(
    name: AgentName,
    task: TaskBrief,
    wait: bool = False,
    resume_agent_id: AgentId | None = None,
) -> TeammateStatus:
    """Revive the teammate's standing session with a new turn (creating the entry from its
    template when absent). Backgrounded by default; `wait=True` blocks until the run finishes and
    returns the outcome on the status. `resume_agent_id` continues one previously-dispatched
    subagent, as in `run_agent`.

    Raises:
        ValueError: When `name` is not on the roster.
    """
    entry = _ensure_teammate(name)
    key = entry.agent_key
    run, options, prompt = prepare_run(
        _pool(), _capability_router(), key, task, resume_agent_id=resume_agent_id
    )
    coro = run_with_capture(_pool(), key, task, options, run=run, prompt=prompt)
    if wait:
        await coro
        return _teammate_status(name, entry)
    _runner().spawn(run.run_id, coro)
    return _teammate_status(name, entry)
```

(Completion notification arrives in Task 8; this task's `prepare_run` calls take no extra hooks.)

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/mcp/ -v`
Expected: PASS (all teammate, runner, and pre-existing pool-server tests)

- [ ] **Step 5: Commit**

```bash
printf 'feat(mcp): teammate tools — roster, spawn, check, message\n' > /tmp/cmsg.txt
git add src/agent_fleet_mcp/pool_server.py tests/mcp/test_teammate_tools.py
git commit -F /tmp/cmsg.txt
```

---

### Task 8: Stop-hook completion notification

**Files:**
- Modify: `src/agent_fleet/settings.py` (`AgentFleetSettings`)
- Modify: `src/agent_fleet/engine/render.py` (`with_hooks`)
- Modify: `src/agent_fleet/engine/dispatch.py` (`prepare_run`)
- Modify: `src/agent_fleet_mcp/pool_server.py` (`_notify_hooks` real body; pass `extra_hooks=`)
- Test: `tests/mcp/test_teammate_tools.py` (append)

- [ ] **Step 1: Write the failing test**

```python
def test_notify_command_folds_a_stop_hook_into_settings(
    pool: AgentPool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENT_FLEET_NOTIFY_COMMAND", "notify-send teammate-done")

    async def scenario() -> None:
        pool_server._ensure_teammate(_NAME)
        entry = pool.get_by_key(teammate_key(_NAME))
        assert entry is not None
        captured: list[object] = []

        async def capturing(**kwargs: object):
            captured.append(kwargs["options"])
            yield _assistant("ok then", session_id=entry.session_id)

        monkeypatch.setattr("agent_fleet.engine.dispatch.query", lambda **kw: capturing(**kw))
        await pool_server.message_teammate(_NAME, _TASK, wait=True)

        options = captured[0]
        assert options.settings is not None
        text = Path(options.settings).read_text(encoding="utf-8")
        assert '"Stop"' in text
        assert "notify-send teammate-done" in text

    asyncio.run(scenario())
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/mcp/test_teammate_tools.py -v -k notify`
Expected: FAIL — `options.settings` is None

- [ ] **Step 3: Add the setting** to `AgentFleetSettings` in `src/agent_fleet/settings.py`, after `pool_db`:

```python
    notify_command: str | None = Field(
        default=None,
        description="Shell command a Stop hook runs when a teammate run finishes; None disables.",
        validation_alias="AGENT_FLEET_NOTIFY_COMMAND",
    )
```

- [ ] **Step 4: Extend `with_hooks`** in `render.py` — add a keyword parameter and fold it into the merge:

```python
def with_hooks(
    options: ClaudeAgentOptions,
    spec: AgentSpec,
    directory: Path,
    *,
    subagents: Mapping[AgentName, AgentSpec] | None = None,
    extra: HookConfig | None = None,
) -> ClaudeAgentOptions:
```

and replace the merge line:

```python
    specs = [spec, *(subagents.values() if subagents is not None else ())]
    configs = [s.hooks for s in specs if s.hooks is not None]
    if extra is not None:
        configs.append(extra)
    merged = _merge_hook_configs(configs)
```

Document `extra` in the docstring's Args: a caller-supplied hook config (e.g. the teammate
Stop-notification) folded into the same settings file as the specs' own hooks.
(`HookConfig` is already imported in the models; add `from capdisc.hooks import HookConfig` to
`render.py` if absent.)

- [ ] **Step 5: Thread it through `prepare_run`** in `dispatch.py` — add the parameter:

```python
def prepare_run(
    pool: AgentPool,
    capability_router: CapabilityRouter,
    agent_key: AgentKey,
    task: TaskBrief,
    *,
    subagent_agent_keys: dict[AgentName, AgentKey] | None = None,
    resume_agent_id: AgentId | None = None,
    extra_hooks: HookConfig | None = None,
) -> tuple[RunRecord, ClaudeAgentOptions, TaskBrief | None]:
```

and pass it on:

```python
    options = with_hooks(
        options, entry.spec, pool.db_path.parent, subagents=subagents, extra=extra_hooks
    )
```

with `from capdisc.hooks import HookConfig` added to the imports and `extra_hooks` documented in
the Args (hooks folded in beyond the specs' own — the teammate notification path).

- [ ] **Step 6: Add `_notify_hooks` to `pool_server.py`** (near the other `_`-helpers):

```python
def _notify_hooks() -> HookConfig | None:
    """The Stop-notification hook for teammate runs, from `AGENT_FLEET_NOTIFY_COMMAND`; None
    when unconfigured."""
    command = AgentFleetSettings().notify_command
    if command is None:
        return None
    return HookConfig(
        {HookEvent.stop: [MatcherGroup(hooks=[CommandHook(command=command)])]}
    )
```

with imports `from capdisc.hooks import CommandHook, HookConfig, HookEvent, MatcherGroup`
(all four are re-exported from `capdisc.hooks`). Then pass `extra_hooks=_notify_hooks()` in both
of Task 7's `prepare_run` calls — `spawn_teammate`:

```python
    run, options, prompt = prepare_run(
        _pool(), _capability_router(), key, task, extra_hooks=_notify_hooks()
    )
```

and `message_teammate`:

```python
    run, options, prompt = prepare_run(
        _pool(),
        _capability_router(),
        key,
        task,
        resume_agent_id=resume_agent_id,
        extra_hooks=_notify_hooks(),
    )
```

- [ ] **Step 7: Run the full suite**

Run: `uv run pytest -x -q`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
printf 'feat(hooks): optional Stop-hook notification for teammate runs\n' > /tmp/cmsg.txt
git add src/agent_fleet/settings.py src/agent_fleet/engine/render.py src/agent_fleet/engine/dispatch.py src/agent_fleet_mcp/pool_server.py tests/mcp/test_teammate_tools.py
git commit -F /tmp/cmsg.txt
```

---

### Task 9: Spec amendment, docs, full gates

**Files:**
- Modify: `docs/superpowers/specs/2026-07-25-teammates-design.md` (key scheme)
- Modify: `README.md` (Pool section)

- [ ] **Step 1: Amend the spec** — `AgentKey`'s pattern forbids `:`; replace both occurrences of
`` `teammate:{name}` `` with `` `teammate.{name}` `` in the spec.

- [ ] **Step 2: Add a Teammates paragraph to README.md** after the Pool section:

```markdown
## Teammates

The pool's primary MCP surface: a hardcoded roster of templated teammates
(`src/agent_fleet/engine/teammates.py`), addressed by name. `spawn_teammate` stands one up from
its template (toolkits pin its capabilities) and runs it in the background; `check_teammate`
reads the derived status and persisted outcome; `message_teammate` revives the standing session.
The pool key is `teammate.{name}`, so the same name always resumes the same conversation.
```

- [ ] **Step 3: Run every gate**

Run: `uv run pytest -q && uv run ruff check && uv run mypy src`
Expected: pytest reports all tests passing, ruff reports no violations, mypy reports no errors.
Report the actual output — if any gate fails, fix before committing.

- [ ] **Step 4: Commit**

```bash
printf 'docs: teammate surface in README; fix key scheme in spec\n' > /tmp/cmsg.txt
git add docs/superpowers/specs/2026-07-25-teammates-design.md README.md
git commit -F /tmp/cmsg.txt
```
