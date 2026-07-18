from __future__ import annotations

import asyncio
import dataclasses
import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

from claude_agent_sdk import ClaudeAgentOptions

from capabilities_discovery.catalog import DEFAULT_RECALL_LIMIT, RecallLimit

from ..models.agent import (
    AgentId,
    AgentKey,
    AgentName,
    AgentRunRecord,
    AgentSpec,
    Finding,
    FindingContent,
    PoolEntry,
    ProblemRequest,
    RunId,
    RunRecord,
    SessionId,
    TaskBrief,
)
from .pipeline import assemble
from .render import to_options
from .source import CatalogSource, bm25_normalized, token_list

_CREATE_TABLE = (
    "CREATE TABLE IF NOT EXISTS agents ("
    "agent_key TEXT PRIMARY KEY, "
    "name TEXT NOT NULL, "
    "session_id TEXT NOT NULL, "
    "spec_json TEXT NOT NULL, "
    "cwd TEXT NOT NULL, "
    "created_at TEXT NOT NULL, "
    "updated_at TEXT NOT NULL)"
)
_CREATE_RUNS_TABLE = (
    "CREATE TABLE IF NOT EXISTS runs ("
    "run_id TEXT PRIMARY KEY, "
    "agent_key TEXT NOT NULL, "
    "task TEXT NOT NULL, "
    "started_at TEXT NOT NULL, "
    "finished_at TEXT)"
)
_CREATE_AGENT_RUNS_TABLE = (
    "CREATE TABLE IF NOT EXISTS agent_runs ("
    "id INTEGER PRIMARY KEY AUTOINCREMENT, "
    "run_id TEXT NOT NULL, "
    "tool_use_id TEXT, "
    "agent_name TEXT, "
    "agent_id TEXT, "
    "session_id TEXT NOT NULL, "
    "recorded_at TEXT NOT NULL)"
)

_CREATE_FINDINGS_TABLE = (
    "CREATE TABLE IF NOT EXISTS findings ("
    "id INTEGER PRIMARY KEY AUTOINCREMENT, "
    "agent_key TEXT NOT NULL, "
    "run_id TEXT NOT NULL, "
    "agent_name TEXT, "
    "session_id TEXT NOT NULL, "
    "content TEXT NOT NULL, "
    "recorded_at TEXT NOT NULL)"
)

type PoolEntryList = list[PoolEntry]  # `AgentPool.list` shadows builtin `list` in annotations
type RunRecordList = list[RunRecord]
type AgentRunRecordList = list[AgentRunRecord]
type FindingList = list[Finding]


def _new_session_id() -> SessionId:
    """Mint a fresh session UUID for a pool entry — the internal id an agent key maps to."""
    return str(uuid.uuid4())


def _new_run_id() -> RunId:
    """Mint a fresh run UUID — the id one invocation of a pooled agent is tracked under."""
    return str(uuid.uuid4())


def _entry_text(entry: PoolEntry) -> str:
    """The text a pool entry is scored on — its spec description plus tags, mirroring
    `source._entry_text`'s description-and-tags shape for catalog entries."""
    return " ".join([entry.spec.description, *entry.spec.tags])


class AgentPool:
    """A SQLite-backed pool of key-addressed, resumable agent sessions.

    Each entry maps a stable `AgentKey` to a display `name`, a generated session UUID, and the
    `AgentSpec` that built it, so a pooled agent can be retrieved by its key (or rediscovered
    fuzzily via `find`) and resumed against the same live Claude Agent SDK conversation. Specs are
    stored and loaded through `AgentSpec.model_dump_json` / `AgentSpec.model_validate_json`;
    timestamps are ISO-8601 UTC strings.

    A single connection is held for the pool's lifetime (`check_same_thread=False`), since the
    schema is one small table and per-call reconnection buys nothing here.
    """

    def __init__(self, db_path: Path) -> None:
        """Open (creating parents) the SQLite database at `db_path` and ensure the schema exists.

        Args:
            db_path: The SQLite file backing the pool; its parent directory is created if missing.
        """
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        # cross-thread self._conn (AsyncAgentPool runs it via asyncio.to_thread) needs this lock.
        self._write_lock = threading.Lock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        # busy_timeout is per-connection; other processes sharing db_path must set their own.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_CREATE_TABLE)
        self._conn.execute(_CREATE_RUNS_TABLE)
        self._conn.execute(_CREATE_AGENT_RUNS_TABLE)
        self._conn.execute(_CREATE_FINDINGS_TABLE)
        self._conn.commit()

    @property
    def db_path(self) -> Path:
        """The SQLite file backing the pool; its parent is the pool's writable state directory."""
        return self._db_path

    def save(
        self,
        agent_key: AgentKey,
        spec: AgentSpec,
        *,
        cwd: Path | None = None,
        reset_session: bool = False,
    ) -> PoolEntry:
        """Insert or overwrite the entry keyed by `agent_key`, returning the stored entry.

        On overwrite the existing session UUID, `cwd`, and `created_at` are preserved by default, so
        a resume continues the same conversation from the same working directory; pass
        `reset_session=True` to mint a new UUID, or an explicit `cwd` to move the entry's working
        directory. On first insert `cwd` defaults to `Path.cwd()` when not given. The display `name`
        is taken from `spec.name` on every save.

        Args:
            agent_key: The agent key that identifies the entry.
            spec: The agent spec to store; its `name` supplies the display label.
            cwd: The working directory to run the entry's session from; on first insert defaults to
                the process cwd, on overwrite preserves the stored value unless given.
            reset_session: When True, generate a fresh session UUID even if the id already exists.

        Returns:
            The persisted pool entry.
        """
        with self._write_lock:
            existing = self._get_by_key_locked(agent_key)
            now = datetime.now(UTC)
            if existing is not None and not reset_session:
                session_id = existing.session_id
            else:
                session_id = _new_session_id()
            created_at = existing.created_at if existing is not None else now
            if cwd is not None:
                resolved_cwd = cwd
            elif existing is not None:
                resolved_cwd = existing.cwd
            else:
                resolved_cwd = Path.cwd()
            entry = PoolEntry(
                agent_key=agent_key,
                name=spec.name,
                spec=spec,
                session_id=session_id,
                cwd=resolved_cwd,
                created_at=created_at,
                updated_at=now,
            )
            self._conn.execute(
                "INSERT OR REPLACE INTO agents "
                "(agent_key, name, session_id, spec_json, cwd, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    entry.agent_key,
                    entry.name,
                    entry.session_id,
                    spec.model_dump_json(),
                    str(entry.cwd),
                    entry.created_at.isoformat(),
                    entry.updated_at.isoformat(),
                ),
            )
            self._conn.commit()
        return entry

    def reconcile_session(self, agent_key: AgentKey, session_id: SessionId) -> PoolEntry:
        """Point an existing entry's session at `session_id`, updating only it and `updated_at`.

        The self-healing step for the cwd-mismatch failure mode: when a live run's observed
        top-level session id doesn't match what the entry expected (a resume that returned a fresh
        session instead of history), this repins the entry to the session that is actually
        continuable going forward, preserving `spec`, `cwd`, and `created_at`.

        Args:
            agent_key: The existing entry to reconcile.
            session_id: The observed session id to adopt.

        Returns:
            The updated pool entry.

        Raises:
            KeyError: When `agent_key` is not in the pool — this reconciles an existing entry, it
                does not create one.
        """
        now = datetime.now(UTC)
        with self._write_lock:
            existing = self._get_by_key_locked(agent_key)
            if existing is None:
                raise KeyError(agent_key)
            cursor = self._conn.execute(
                "UPDATE agents SET session_id = ?, updated_at = ? WHERE agent_key = ?",
                (session_id, now.isoformat(), agent_key),
            )
            self._conn.commit()
            if cursor.rowcount == 0:
                raise KeyError(agent_key)
        return existing.model_copy(update={"session_id": session_id, "updated_at": now})

    def _get_by_key_locked(self, agent_key: AgentKey) -> PoolEntry | None:
        """Read the entry under `agent_key`, assuming the caller already holds `_write_lock`."""
        row = self._conn.execute(
            "SELECT agent_key, name, session_id, spec_json, cwd, created_at, updated_at "
            "FROM agents WHERE agent_key = ?",
            (agent_key,),
        ).fetchone()
        return None if row is None else _row_to_entry(row)

    def get_by_key(self, agent_key: AgentKey) -> PoolEntry | None:
        """Return the entry stored under `agent_key`, or None if the pool holds no such entry."""
        with self._write_lock:
            return self._get_by_key_locked(agent_key)

    def list(self) -> PoolEntryList:
        """Return every entry, most-recently-updated first."""
        with self._write_lock:
            rows = self._conn.execute(
                "SELECT agent_key, name, session_id, spec_json, cwd, created_at, updated_at "
                "FROM agents ORDER BY updated_at DESC"
            ).fetchall()
        return [_row_to_entry(row) for row in rows]

    def _start_run_locked(self, agent_key: AgentKey, task: TaskBrief) -> RunRecord:
        """Open a run, minting a fresh run id, assuming the caller already holds `_write_lock`."""
        now = datetime.now(UTC)
        record = RunRecord(run_id=_new_run_id(), agent_key=agent_key, task=task, started_at=now)
        self._conn.execute(
            "INSERT INTO runs (run_id, agent_key, task, started_at, finished_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (record.run_id, record.agent_key, record.task, record.started_at.isoformat(), None),
        )
        self._conn.commit()
        return record

    def start_run(self, agent_key: AgentKey, task: TaskBrief) -> RunRecord:
        """Open a run for `agent_key`, minting a fresh run id and stamping `started_at`.

        Args:
            agent_key: The pooled agent the run belongs to.
            task: The task the run was launched with.

        Returns:
            The opened run record, with `finished_at` left None.
        """
        with self._write_lock:
            return self._start_run_locked(agent_key, task)

    def begin_run(self, entry: PoolEntry, task: TaskBrief) -> tuple[RunRecord, ClaudeAgentOptions]:
        """Atomically decide resume-vs-fresh from run history and open the run, under one lock.

        The resume-vs-fresh decision (does this entry have prior runs?) and the `start_run` that
        acts on it must not interleave with a concurrent call for the same entry: two callers that
        each separately observed an empty run history would both build fresh-session options pinned
        to the entry's single session UUID, racing two live SDK sessions against the same id.
        Reading the prior runs and opening the new run under one `_write_lock` acquisition closes
        that window — the second caller sees the first's just-started run and resumes instead. The
        option builders are pure in-memory computation and take no lock, so calling them here does
        not re-enter it.

        Args:
            entry: The pool entry to run; its stored session UUID is what fresh options pin and
                resume options continue.
            task: The task the opened run is launched with and recorded under.

        Returns:
            The opened run record and its live options — resume options when the entry already had
            runs, fresh-session options for its first.
        """
        with self._write_lock:
            options = (
                self.to_resume_options(entry)
                if self._list_runs_locked(entry.agent_key)
                else self.to_new_run_options(entry)
            )
            run = self._start_run_locked(entry.agent_key, task)
        return run, options

    def finish_run(self, run_id: RunId) -> RunRecord:
        """Stamp `finished_at` on the run and return the updated record.

        Args:
            run_id: The run to close.

        Returns:
            The updated run record.

        Raises:
            KeyError: When `run_id` is unknown.
        """
        now = datetime.now(UTC)
        with self._write_lock:
            cursor = self._conn.execute(
                "UPDATE runs SET finished_at = ? WHERE run_id = ?", (now.isoformat(), run_id)
            )
            self._conn.commit()
            if cursor.rowcount == 0:
                raise KeyError(run_id)
            row = self._conn.execute(
                "SELECT run_id, agent_key, task, started_at, finished_at "
                "FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return _row_to_run(row)

    def record_agent_run(
        self,
        run_id: RunId,
        session_id: SessionId,
        *,
        tool_use_id: str | None = None,
        agent_name: AgentName | None = None,
        agent_id: AgentId | None = None,
    ) -> AgentRunRecord:
        """Append one agent that ran within `run_id`.

        Args:
            run_id: The run the agent ran within.
            session_id: The run's session id — the same for the main agent and every subagent it
                dispatches (a dispatched subagent has no separate resumable session).
            tool_use_id: The id of the dispatching tool-use block, or None for the main agent.
            agent_name: The dispatched agent's `subagent_type`, or None for the main agent.
            agent_id: The dispatched subagent's harness id (`TaskStartedMessage.task_id`) — the
                handle to resume that one subagent via `SendMessage`, or None for the main agent.

        Returns:
            The appended agent-run record.
        """
        now = datetime.now(UTC)
        record = AgentRunRecord(
            run_id=run_id,
            tool_use_id=tool_use_id,
            agent_name=agent_name,
            agent_id=agent_id,
            session_id=session_id,
            recorded_at=now,
        )
        with self._write_lock:
            self._conn.execute(
                "INSERT INTO agent_runs "
                "(run_id, tool_use_id, agent_name, agent_id, session_id, recorded_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    record.run_id,
                    record.tool_use_id,
                    record.agent_name,
                    record.agent_id,
                    record.session_id,
                    record.recorded_at.isoformat(),
                ),
            )
            self._conn.commit()
        return record

    def get_run(self, run_id: RunId) -> RunRecord | None:
        """Return the run stored under `run_id`, or None if the pool holds no such run."""
        with self._write_lock:
            row = self._conn.execute(
                "SELECT run_id, agent_key, task, started_at, finished_at "
                "FROM runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return None if row is None else _row_to_run(row)

    def _list_runs_locked(self, agent_key: AgentKey) -> RunRecordList:
        """Read every run for `agent_key`, assuming the caller already holds `_write_lock`."""
        rows = self._conn.execute(
            "SELECT run_id, agent_key, task, started_at, finished_at "
            "FROM runs WHERE agent_key = ? ORDER BY started_at DESC",
            (agent_key,),
        ).fetchall()
        return [_row_to_run(row) for row in rows]

    def list_runs(self, agent_key: AgentKey) -> RunRecordList:
        """Return every run for `agent_key`, most-recently-started first."""
        with self._write_lock:
            return self._list_runs_locked(agent_key)

    def list_agent_runs(self, run_id: RunId) -> AgentRunRecordList:
        """Return every agent within `run_id`, in `recorded_at` order (main row first)."""
        with self._write_lock:
            rows = self._conn.execute(
                "SELECT run_id, tool_use_id, agent_name, agent_id, session_id, recorded_at "
                "FROM agent_runs WHERE run_id = ? ORDER BY recorded_at, id",
                (run_id,),
            ).fetchall()
        return [_row_to_agent_run(row) for row in rows]

    def record_finding(
        self,
        agent_key: AgentKey,
        run_id: RunId,
        session_id: SessionId,
        content: FindingContent,
        *,
        agent_name: AgentName | None = None,
    ) -> Finding:
        """Append one finding to the pooled agent's shared, append-only findings document.

        A single atomic INSERT stamped `recorded_at=now`; SQLite serializes row writes, so
        concurrent lenses calling this can't corrupt each other's rows. Nothing is ever updated in
        place — every call adds a row.

        Args:
            agent_key: The pooled agent the finding belongs to.
            run_id: The run the finding was produced within.
            session_id: The writing agent's session id.
            content: The finding text to preserve.
            agent_name: The lens that wrote it, or None for the main/supervisor agent.

        Returns:
            The appended finding.
        """
        now = datetime.now(UTC)
        finding = Finding(
            agent_key=agent_key,
            run_id=run_id,
            agent_name=agent_name,
            session_id=session_id,
            content=content,
            recorded_at=now,
        )
        with self._write_lock:
            self._conn.execute(
                "INSERT INTO findings "
                "(agent_key, run_id, agent_name, session_id, content, recorded_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    finding.agent_key,
                    finding.run_id,
                    finding.agent_name,
                    finding.session_id,
                    finding.content,
                    finding.recorded_at.isoformat(),
                ),
            )
            self._conn.commit()
        return finding

    def list_findings(self, agent_key: AgentKey, *, run_id: RunId | None = None) -> FindingList:
        """Return the pooled agent's findings oldest-first — the assembled-document reading order.

        Args:
            agent_key: The pooled agent whose findings to read.
            run_id: When given, return only the findings recorded within that run.

        Returns:
            Every finding for `agent_key` (optionally filtered to `run_id`), oldest-first.
        """
        with self._write_lock:
            if run_id is None:
                rows = self._conn.execute(
                    "SELECT agent_key, run_id, agent_name, session_id, content, recorded_at "
                    "FROM findings WHERE agent_key = ? ORDER BY recorded_at, id",
                    (agent_key,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT agent_key, run_id, agent_name, session_id, content, recorded_at "
                    "FROM findings WHERE agent_key = ? AND run_id = ? ORDER BY recorded_at, id",
                    (agent_key, run_id),
                ).fetchall()
        return [_row_to_finding(row) for row in rows]

    def delete(self, agent_key: AgentKey) -> bool:
        """Remove the entry under `agent_key`, cascading to its runs, agent-runs, and findings.

        All four tables are cleared under one `_write_lock` before a single commit, so the cascade
        is atomic. `agent_runs` is keyed only by `run_id` (no `agent_key` column), so its rows are
        deleted via the agent's `runs` before those `runs` rows are removed; `findings` and `runs`
        each carry their own `agent_key` column and are deleted directly. Existence is reported from
        the `agents` delete.

        Returns:
            True if an entry existed and was removed, False if the id was absent.
        """
        with self._write_lock:
            self._conn.execute(
                "DELETE FROM agent_runs WHERE run_id IN "
                "(SELECT run_id FROM runs WHERE agent_key = ?)",
                (agent_key,),
            )
            self._conn.execute("DELETE FROM findings WHERE agent_key = ?", (agent_key,))
            self._conn.execute("DELETE FROM runs WHERE agent_key = ?", (agent_key,))
            cursor = self._conn.execute("DELETE FROM agents WHERE agent_key = ?", (agent_key,))
            self._conn.commit()
        return cursor.rowcount > 0

    def find(self, query: TaskBrief, limit: RecallLimit = DEFAULT_RECALL_LIMIT) -> PoolEntryList:
        """Rediscover pooled entries by re-describing the problem, ranked most-relevant first.

        Reuses the catalog recall machinery — `token_list` and `bm25_normalized` from
        `engine.source`, exactly as `BM25Ranker.rank` scores catalog entries — against each entry's
        spec description and tags. Ties break by `agent_key` for a deterministic order.

        Args:
            query: The re-described problem text to rank entries against.
            limit: The most entries to return.

        Returns:
            Up to `limit` entries, highest BM25 score first.
        """
        entries = self.list()
        terms = sorted(set(token_list(query)))  # sorted → sum order is hash-seed-stable
        docs = [token_list(_entry_text(entry)) for entry in entries]
        scores = bm25_normalized(terms, docs)
        ranked = sorted(
            zip(entries, scores, strict=True),
            key=lambda pair: (-pair[1], pair[0].agent_key),
        )
        return [entry for entry, _ in ranked[:limit]]

    def to_new_run_options(self, entry: PoolEntry) -> ClaudeAgentOptions:
        """Build the live options for an entry's FIRST run, pinning the session to its UUID.

        Wraps `render.to_options(entry.spec)` and sets `session_id` so the SDK creates the session
        under the pool's chosen UUID instead of an auto-generated one.

        Args:
            entry: The pool entry to run.

        Returns:
            The options with `session_id` set to the entry's UUID and `resume` left None.
        """
        return dataclasses.replace(
            to_options(entry.spec), session_id=entry.session_id, cwd=entry.cwd
        )

    def to_resume_options(self, entry: PoolEntry) -> ClaudeAgentOptions:
        """Build the live options that RESUME an entry's session, loading its conversation history.

        Wraps `render.to_options(entry.spec)` and sets `resume` to the entry's UUID.

        Args:
            entry: The pool entry to resume.

        Returns:
            The options with `resume` set to the entry's UUID and `session_id` left None.
        """
        return dataclasses.replace(to_options(entry.spec), resume=entry.session_id, cwd=entry.cwd)


class AsyncAgentPool:
    """A thin async facade over a sync `AgentPool` for consumers running an asyncio loop.

    `AgentPool` is backed by stdlib `sqlite3`, whose calls genuinely block; the rest of this
    package's live-SDK surface (`run_agent`, `Orchestrator.spawn`) is `async def`. A consumer that
    creates or resumes pool entries and then `await`s a run would otherwise have to wrap every
    `AgentPool` call in `asyncio.to_thread` by hand to keep the event loop free. This wraps that
    once: the SQLite-touching methods (`save`, `create_agent`, `reconcile_session`, `get_by_key`,
    `list`, `delete`, `find`, `start_run`, `finish_run`, `record_agent_run`, `get_run`, `list_runs`,
    `list_agent_runs`, `record_finding`, `list_findings`) become `async def` delegations dispatched
    off the loop via
    `asyncio.to_thread`, while the pure
    in-memory option builders (`to_new_run_options`, `to_resume_options`) stay plain synchronous
    delegations — there is no I/O to offload.

    Composition, not reimplementation: this holds an existing `AgentPool` and every method body
    delegates to it. It opens no connection and owns no schema of its own.
    """

    def __init__(self, pool: AgentPool) -> None:
        """Wrap an existing `AgentPool`; all state lives in the wrapped pool.

        Args:
            pool: The sync pool whose blocking calls are dispatched to a thread.
        """
        self._pool = pool

    @property
    def pool(self) -> AgentPool:
        """The wrapped sync pool, for callers that must invoke its sync methods directly (e.g.
        `run_with_capture`, whose own live-SDK loop already runs on the event loop)."""
        return self._pool

    async def save(
        self,
        agent_key: AgentKey,
        spec: AgentSpec,
        *,
        cwd: Path | None = None,
        reset_session: bool = False,
    ) -> PoolEntry:
        """Insert or overwrite the entry keyed by `agent_key`, off the event loop.

        Delegates to `AgentPool.save` via `asyncio.to_thread`. See that method for the
        session/cwd/timestamp preservation semantics.

        Args:
            agent_key: The agent key that identifies the entry.
            spec: The agent spec to store; its `name` supplies the display label.
            cwd: The working directory to run the entry's session from; see `AgentPool.save`.
            reset_session: When True, generate a fresh session UUID even if the id already exists.

        Returns:
            The persisted pool entry.
        """
        return await asyncio.to_thread(
            self._pool.save, agent_key, spec, cwd=cwd, reset_session=reset_session
        )

    async def create_agent(
        self,
        agent_key: AgentKey,
        request: ProblemRequest,
        source: CatalogSource,
        *,
        cwd: Path | None = None,
        reset_session: bool = False,
    ) -> PoolEntry:
        """Assemble an agent from a request and store it in the pool, off the event loop.

        Delegates to the module-level `create_agent` via `asyncio.to_thread`.

        Args:
            agent_key: The agent key the stored entry is saved under.
            request: The problem request driving assembly; its `name` is auto-slugged when omitted.
            source: The catalog source to recall candidates from.
            cwd: The working directory to pin the entry's session to; see `AgentPool.save`.
            reset_session: When True, mint a fresh session UUID even if the id already exists.

        Returns:
            The stored pool entry.
        """
        return await asyncio.to_thread(
            create_agent,
            agent_key,
            request,
            source,
            self._pool,
            cwd=cwd,
            reset_session=reset_session,
        )

    async def reconcile_session(self, agent_key: AgentKey, session_id: SessionId) -> PoolEntry:
        """Repin an existing entry's session to `session_id`, off the event loop.

        Delegates to `AgentPool.reconcile_session` via `asyncio.to_thread`.
        """
        return await asyncio.to_thread(self._pool.reconcile_session, agent_key, session_id)

    async def get_by_key(self, agent_key: AgentKey) -> PoolEntry | None:
        """Return the entry stored under `agent_key`, or None if absent — off the event loop.

        Delegates to `AgentPool.get_by_key` via `asyncio.to_thread`.
        """
        return await asyncio.to_thread(self._pool.get_by_key, agent_key)

    async def list(self) -> PoolEntryList:
        """Return every entry, most-recently-updated first — off the event loop.

        Delegates to `AgentPool.list` via `asyncio.to_thread`.
        """
        return await asyncio.to_thread(self._pool.list)

    async def delete(self, agent_key: AgentKey) -> bool:
        """Remove the entry stored under `agent_key`, off the event loop.

        Delegates to `AgentPool.delete` via `asyncio.to_thread`, cascading to the agent's runs,
        agent-runs, and findings atomically.

        Returns:
            True if an entry existed and was removed, False if the id was absent.
        """
        return await asyncio.to_thread(self._pool.delete, agent_key)

    async def find(
        self, query: TaskBrief, limit: RecallLimit = DEFAULT_RECALL_LIMIT
    ) -> PoolEntryList:
        """Rediscover pooled entries by re-describing the problem — off the event loop.

        Delegates to `AgentPool.find` via `asyncio.to_thread`.

        Args:
            query: The re-described problem text to rank entries against.
            limit: The most entries to return.

        Returns:
            Up to `limit` entries, highest BM25 score first.
        """
        return await asyncio.to_thread(self._pool.find, query, limit)

    async def start_run(self, agent_key: AgentKey, task: TaskBrief) -> RunRecord:
        """Open a run for `agent_key`, off the event loop.

        Delegates to `AgentPool.start_run` via `asyncio.to_thread`.
        """
        return await asyncio.to_thread(self._pool.start_run, agent_key, task)

    async def finish_run(self, run_id: RunId) -> RunRecord:
        """Stamp `finished_at` on the run, off the event loop.

        Delegates to `AgentPool.finish_run` via `asyncio.to_thread`.
        """
        return await asyncio.to_thread(self._pool.finish_run, run_id)

    async def record_agent_run(
        self,
        run_id: RunId,
        session_id: SessionId,
        *,
        tool_use_id: str | None = None,
        agent_name: AgentName | None = None,
        agent_id: AgentId | None = None,
    ) -> AgentRunRecord:
        """Append one agent that ran within `run_id`, off the event loop.

        Delegates to `AgentPool.record_agent_run` via `asyncio.to_thread`.
        """
        return await asyncio.to_thread(
            self._pool.record_agent_run,
            run_id,
            session_id,
            tool_use_id=tool_use_id,
            agent_name=agent_name,
            agent_id=agent_id,
        )

    async def get_run(self, run_id: RunId) -> RunRecord | None:
        """Return the run stored under `run_id`, or None if absent — off the event loop.

        Delegates to `AgentPool.get_run` via `asyncio.to_thread`.
        """
        return await asyncio.to_thread(self._pool.get_run, run_id)

    async def list_runs(self, agent_key: AgentKey) -> RunRecordList:
        """Return every run for `agent_key`, most-recently-started first — off the event loop.

        Delegates to `AgentPool.list_runs` via `asyncio.to_thread`.
        """
        return await asyncio.to_thread(self._pool.list_runs, agent_key)

    async def list_agent_runs(self, run_id: RunId) -> AgentRunRecordList:
        """Return every agent dispatched within `run_id`, in `recorded_at` order — off the loop.

        Delegates to `AgentPool.list_agent_runs` via `asyncio.to_thread`.
        """
        return await asyncio.to_thread(self._pool.list_agent_runs, run_id)

    async def record_finding(
        self,
        agent_key: AgentKey,
        run_id: RunId,
        session_id: SessionId,
        content: FindingContent,
        *,
        agent_name: AgentName | None = None,
    ) -> Finding:
        """Append one finding to the pooled agent's shared findings document — off the event loop.

        Delegates to `AgentPool.record_finding` via `asyncio.to_thread`.
        """
        return await asyncio.to_thread(
            self._pool.record_finding,
            agent_key,
            run_id,
            session_id,
            content,
            agent_name=agent_name,
        )

    async def list_findings(
        self, agent_key: AgentKey, *, run_id: RunId | None = None
    ) -> FindingList:
        """Return the pooled agent's findings oldest-first — off the event loop.

        Delegates to `AgentPool.list_findings` via `asyncio.to_thread`.
        """
        return await asyncio.to_thread(self._pool.list_findings, agent_key, run_id=run_id)

    def to_new_run_options(self, entry: PoolEntry) -> ClaudeAgentOptions:
        """Build the live options for an entry's FIRST run — synchronous by design.

        Plain `def`, not `async def`: `AgentPool.to_new_run_options` is pure in-memory computation
        (`dataclasses.replace` over `render.to_options`) with no I/O, so there is nothing to offload
        to a thread — dispatching it would be pure overhead. Delegates directly.

        Args:
            entry: The pool entry to run.

        Returns:
            The options with `session_id` set to the entry's UUID and `resume` left None.
        """
        return self._pool.to_new_run_options(entry)

    def to_resume_options(self, entry: PoolEntry) -> ClaudeAgentOptions:
        """Build the live options that RESUME an entry's session — synchronous by design.

        Plain `def`, not `async def`: `AgentPool.to_resume_options` is pure in-memory computation
        (`dataclasses.replace` over `render.to_options`) with no I/O, so there is nothing to offload
        to a thread — dispatching it would be pure overhead. Delegates directly.

        Args:
            entry: The pool entry to resume.

        Returns:
            The options with `resume` set to the entry's UUID and `session_id` left None.
        """
        return self._pool.to_resume_options(entry)


def _row_to_entry(row: sqlite3.Row) -> PoolEntry:
    """Rebuild a `PoolEntry` from a database row, decoding the spec via its JSON validator."""
    return PoolEntry(
        agent_key=row["agent_key"],
        name=row["name"],
        spec=AgentSpec.model_validate_json(row["spec_json"]),
        session_id=row["session_id"],
        cwd=Path(row["cwd"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


def _row_to_run(row: sqlite3.Row) -> RunRecord:
    """Rebuild a `RunRecord` from a database row; `finished_at` stays None while a run is open."""
    finished_at = row["finished_at"]
    return RunRecord(
        run_id=row["run_id"],
        agent_key=row["agent_key"],
        task=row["task"],
        started_at=datetime.fromisoformat(row["started_at"]),
        finished_at=datetime.fromisoformat(finished_at) if finished_at is not None else None,
    )


def _row_to_agent_run(row: sqlite3.Row) -> AgentRunRecord:
    """Rebuild an `AgentRunRecord` from a database row."""
    return AgentRunRecord(
        run_id=row["run_id"],
        tool_use_id=row["tool_use_id"],
        agent_name=row["agent_name"],
        agent_id=row["agent_id"],
        session_id=row["session_id"],
        recorded_at=datetime.fromisoformat(row["recorded_at"]),
    )


def _row_to_finding(row: sqlite3.Row) -> Finding:
    """Rebuild a `Finding` from a database row."""
    return Finding(
        agent_key=row["agent_key"],
        run_id=row["run_id"],
        agent_name=row["agent_name"],
        session_id=row["session_id"],
        content=row["content"],
        recorded_at=datetime.fromisoformat(row["recorded_at"]),
    )


def create_agent(
    agent_key: AgentKey,
    request: ProblemRequest,
    source: CatalogSource,
    pool: AgentPool,
    *,
    cwd: Path | None = None,
    reset_session: bool = False,
) -> PoolEntry:
    """Assemble an agent from an incoming request and store it in the pool under `agent_key`.

    Ties the assembly pipeline (recall → select → compose → score) to a pool save. A request that
    carries no display name has one auto-slugged from its task text by `compose`, so the stored
    entry's `name` is always a meaningful label.

    Args:
        agent_key: The agent key the stored entry is saved under.
        request: The problem request driving assembly; its `name` is auto-slugged when omitted.
        source: The catalog source to recall candidates from.
        pool: The pool to store the assembled agent in.
        cwd: The working directory to pin the entry's session to; defaults to the process cwd on
            first insert, preserved on overwrite unless given.
        reset_session: When True, mint a fresh session UUID even if the id already exists.

    Returns:
        The stored pool entry.
    """
    result = assemble(request, source)
    return pool.save(agent_key, result.spec, cwd=cwd, reset_session=reset_session)
