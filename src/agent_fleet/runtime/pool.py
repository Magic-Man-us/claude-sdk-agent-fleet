"""SQLite-backed fixed-agent sessions without discovery dependencies."""

from __future__ import annotations

import dataclasses
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions

from .models import (
    RuntimeAgentRunRecord,
    RuntimeAgentSpec,
    RuntimePoolEntry,
    RuntimeRunRecord,
)

_CREATE_AGENTS = """
CREATE TABLE IF NOT EXISTS runtime_agents (
    agent_key TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    spec_json TEXT NOT NULL,
    cwd TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""
_CREATE_RUNS = """
CREATE TABLE IF NOT EXISTS runtime_runs (
    run_id TEXT PRIMARY KEY,
    agent_key TEXT NOT NULL,
    task TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT
)
"""
_CREATE_AGENT_RUNS = """
CREATE TABLE IF NOT EXISTS runtime_agent_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    tool_use_id TEXT,
    agent_name TEXT,
    agent_id TEXT,
    recorded_at TEXT NOT NULL
)
"""


class RuntimeAgentPool:
    """Persistent fixed-agent sessions that do not require ``capdisc``."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute(_CREATE_AGENTS)
        self._conn.execute(_CREATE_RUNS)
        self._conn.execute(_CREATE_AGENT_RUNS)
        self._conn.commit()

    @property
    def db_path(self) -> Path:
        return self._db_path

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def save(
        self,
        agent_key: str,
        spec: RuntimeAgentSpec,
        *,
        cwd: Path,
        reset_session: bool = False,
    ) -> RuntimePoolEntry:
        resolved_cwd = cwd.resolve(strict=True)
        now = datetime.now(UTC)
        with self._lock:
            existing = self._get_locked(agent_key)
            session_id = (
                existing.session_id
                if existing is not None and not reset_session
                else str(uuid.uuid4())
            )
            created_at = existing.created_at if existing is not None else now
            entry = RuntimePoolEntry(
                agent_key=agent_key,
                spec=spec,
                session_id=session_id,
                cwd=resolved_cwd,
                created_at=created_at,
                updated_at=now,
            )
            self._conn.execute(
                """
                INSERT INTO runtime_agents (
                    agent_key, session_id, spec_json, cwd, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(agent_key) DO UPDATE SET
                    session_id=excluded.session_id,
                    spec_json=excluded.spec_json,
                    cwd=excluded.cwd,
                    updated_at=excluded.updated_at
                """,
                (
                    entry.agent_key,
                    entry.session_id,
                    entry.spec.model_dump_json(),
                    str(entry.cwd),
                    entry.created_at.isoformat(),
                    entry.updated_at.isoformat(),
                ),
            )
            self._conn.commit()
        return entry

    def get_by_key(self, agent_key: str) -> RuntimePoolEntry | None:
        with self._lock:
            return self._get_locked(agent_key)

    def _get_locked(self, agent_key: str) -> RuntimePoolEntry | None:
        row = self._conn.execute(
            "SELECT * FROM runtime_agents WHERE agent_key = ?", (agent_key,)
        ).fetchone()
        return None if row is None else _entry_from_row(row)

    def start_run(self, agent_key: str, task: str) -> RuntimeRunRecord:
        _validate_task(task)
        run = RuntimeRunRecord(
            run_id=str(uuid.uuid4()),
            agent_key=agent_key,
            task=task,
            started_at=datetime.now(UTC),
        )
        with self._lock:
            if self._get_locked(agent_key) is None:
                raise KeyError(agent_key)
            self._insert_run_locked(run)
        return run

    def begin_run(
        self, entry: RuntimePoolEntry, task: str
    ) -> tuple[RuntimeRunRecord, ClaudeAgentOptions]:
        """Atomically choose first-session versus resume options and open a run."""
        _validate_task(task)
        with self._lock:
            stored = self._get_locked(entry.agent_key)
            if stored is None:
                raise KeyError(entry.agent_key)
            prior = self._conn.execute(
                "SELECT 1 FROM runtime_runs WHERE agent_key = ? LIMIT 1",
                (entry.agent_key,),
            ).fetchone()
            options = _base_options(stored.spec, stored.cwd)
            options = dataclasses.replace(
                options,
                resume=stored.session_id if prior is not None else None,
                session_id=None if prior is not None else stored.session_id,
            )
            run = RuntimeRunRecord(
                run_id=str(uuid.uuid4()),
                agent_key=entry.agent_key,
                task=task,
                started_at=datetime.now(UTC),
            )
            self._insert_run_locked(run)
        return run, options

    def finish_run(self, run_id: str) -> RuntimeRunRecord:
        now = datetime.now(UTC)
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE runtime_runs SET finished_at = ? WHERE run_id = ?",
                (now.isoformat(), run_id),
            )
            self._conn.commit()
            if cursor.rowcount == 0:
                raise KeyError(run_id)
            row = self._conn.execute(
                "SELECT * FROM runtime_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:  # pragma: no cover - guarded by rowcount
            raise KeyError(run_id)
        return _run_from_row(row)

    def reconcile_session(self, agent_key: str, session_id: str) -> RuntimePoolEntry:
        now = datetime.now(UTC)
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE runtime_agents SET session_id = ?, updated_at = ? WHERE agent_key = ?",
                (session_id, now.isoformat(), agent_key),
            )
            self._conn.commit()
            if cursor.rowcount == 0:
                raise KeyError(agent_key)
            row = self._conn.execute(
                "SELECT * FROM runtime_agents WHERE agent_key = ?", (agent_key,)
            ).fetchone()
        if row is None:  # pragma: no cover - guarded by rowcount
            raise KeyError(agent_key)
        return _entry_from_row(row)

    def record_agent_run(
        self,
        run_id: str,
        session_id: str,
        *,
        tool_use_id: str | None = None,
        agent_name: str | None = None,
        agent_id: str | None = None,
    ) -> RuntimeAgentRunRecord:
        record = RuntimeAgentRunRecord(
            run_id=run_id,
            session_id=session_id,
            recorded_at=datetime.now(UTC),
            tool_use_id=tool_use_id,
            agent_name=agent_name,
            agent_id=agent_id,
        )
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO runtime_agent_runs (
                    run_id, session_id, tool_use_id, agent_name, agent_id, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.run_id,
                    record.session_id,
                    record.tool_use_id,
                    record.agent_name,
                    record.agent_id,
                    record.recorded_at.isoformat(),
                ),
            )
            self._conn.commit()
        return record

    def list_agent_runs(self, run_id: str) -> list[RuntimeAgentRunRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM runtime_agent_runs WHERE run_id = ? ORDER BY recorded_at, id",
                (run_id,),
            ).fetchall()
        return [_agent_run_from_row(row) for row in rows]

    def _insert_run_locked(self, run: RuntimeRunRecord) -> None:
        self._conn.execute(
            """
            INSERT INTO runtime_runs (run_id, agent_key, task, started_at, finished_at)
            VALUES (?, ?, ?, ?, NULL)
            """,
            (run.run_id, run.agent_key, run.task, run.started_at.isoformat()),
        )
        self._conn.commit()


def _validate_task(task: str) -> None:
    if not task or len(task) > 8_000:
        raise ValueError("runtime task must contain 1-8000 characters")


def _base_options(spec: RuntimeAgentSpec, cwd: Path) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        system_prompt=spec.system_prompt,
        model=None if spec.model == "inherit" else spec.model,
        max_turns=spec.max_turns,
        permission_mode=spec.permission_mode,
        cwd=cwd,
    )


def _entry_from_row(row: sqlite3.Row) -> RuntimePoolEntry:
    return RuntimePoolEntry(
        agent_key=row["agent_key"],
        spec=RuntimeAgentSpec.model_validate_json(row["spec_json"]),
        session_id=row["session_id"],
        cwd=Path(row["cwd"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def _run_from_row(row: sqlite3.Row) -> RuntimeRunRecord:
    return RuntimeRunRecord(
        run_id=row["run_id"],
        agent_key=row["agent_key"],
        task=row["task"],
        started_at=datetime.fromisoformat(row["started_at"]),
        finished_at=(
            datetime.fromisoformat(row["finished_at"])
            if row["finished_at"] is not None
            else None
        ),
    )


def _agent_run_from_row(row: sqlite3.Row) -> RuntimeAgentRunRecord:
    return RuntimeAgentRunRecord(
        run_id=row["run_id"],
        session_id=row["session_id"],
        tool_use_id=row["tool_use_id"],
        agent_name=row["agent_name"],
        agent_id=row["agent_id"],
        recorded_at=datetime.fromisoformat(row["recorded_at"]),
    )
