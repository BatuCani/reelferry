"""SQLite storage in WAL mode, safe for several local processes at once.

Every thread opens its own connection. Schema changes are numbered migrations
applied in order; later building blocks (posting, calendar) append new ones.
"""

from __future__ import annotations

import json
import secrets
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MIGRATIONS: list[str] = [
    # 1: engine foundation
    """
    CREATE TABLE sources (
        id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        uri TEXT NOT NULL,
        title TEXT NOT NULL,
        path TEXT,
        status TEXT NOT NULL,
        duration REAL,
        width INTEGER,
        height INTEGER,
        fps REAL,
        has_audio INTEGER,
        language TEXT,
        error TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    CREATE TABLE transcripts (
        source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
        version INTEGER NOT NULL,
        language TEXT,
        engine TEXT NOT NULL,
        model TEXT NOT NULL,
        alignment TEXT NOT NULL,
        words TEXT NOT NULL,
        duration REAL NOT NULL,
        note TEXT,
        created_at TEXT NOT NULL,
        PRIMARY KEY (source_id, version)
    );
    CREATE TABLE candidates (
        id TEXT PRIMARY KEY,
        source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
        job_id TEXT,
        start REAL NOT NULL,
        "end" REAL NOT NULL,
        title TEXT NOT NULL,
        score INTEGER NOT NULL,
        reason TEXT NOT NULL,
        emphasis TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE INDEX candidates_source ON candidates(source_id);
    CREATE TABLE clips (
        id TEXT PRIMARY KEY,
        source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
        candidate_id TEXT,
        title TEXT NOT NULL,
        start REAL NOT NULL,
        "end" REAL NOT NULL,
        format TEXT NOT NULL,
        framing TEXT NOT NULL,
        crop_x REAL NOT NULL,
        caption_style TEXT NOT NULL,
        burn_captions INTEGER NOT NULL,
        captions_enabled INTEGER NOT NULL,
        status TEXT NOT NULL,
        score INTEGER,
        reason TEXT,
        emphasis TEXT NOT NULL,
        revision INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    CREATE INDEX clips_source ON clips(source_id);
    CREATE TABLE renders (
        id TEXT PRIMARY KEY,
        clip_id TEXT NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
        job_id TEXT,
        clip_revision INTEGER NOT NULL,
        video_path TEXT NOT NULL,
        captions TEXT NOT NULL,
        format TEXT NOT NULL,
        width INTEGER NOT NULL,
        height INTEGER NOT NULL,
        duration REAL NOT NULL,
        size_bytes INTEGER NOT NULL,
        caption_style TEXT NOT NULL,
        burned_captions INTEGER NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE INDEX renders_clip ON renders(clip_id);
    CREATE TABLE jobs (
        id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        status TEXT NOT NULL,
        input TEXT NOT NULL,
        steps TEXT NOT NULL DEFAULT '{}',
        result TEXT,
        error TEXT,
        progress REAL NOT NULL DEFAULT 0,
        message TEXT,
        cost_usd REAL NOT NULL DEFAULT 0,
        cancel_requested INTEGER NOT NULL DEFAULT 0,
        lease_owner TEXT,
        lease_expires_at REAL,
        attempts INTEGER NOT NULL DEFAULT 0,
        source_id TEXT,
        clip_id TEXT,
        created_at TEXT NOT NULL,
        started_at TEXT,
        finished_at TEXT,
        updated_at TEXT NOT NULL
    );
    CREATE INDEX jobs_status ON jobs(status, created_at);
    CREATE TABLE workers (
        id TEXT PRIMARY KEY,
        pid INTEGER NOT NULL,
        started_at TEXT NOT NULL,
        heartbeat_at REAL NOT NULL
    );
    """,
]


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def new_id(prefix: str) -> str:
    alphabet = "0123456789abcdefghijkmnopqrstuvwxyz"
    return prefix + "_" + "".join(secrets.choice(alphabet) for _ in range(10))


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class Database:
    """Thread-aware access to the SQLite file. One connection per thread."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._local = threading.local()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    @property
    def conn(self) -> sqlite3.Connection:
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._connect()
            self._local.conn = conn
        return conn

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        """A write transaction that takes the lock up front (no upgrade deadlocks)."""
        conn = self.conn
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")

    def one(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
        row: sqlite3.Row | None = self.conn.execute(sql, params).fetchone()
        return row

    def all(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        return list(self.conn.execute(sql, params).fetchall())

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> int:
        return self.conn.execute(sql, params).rowcount

    def close(self) -> None:
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None

    def _migrate(self) -> None:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
            row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
            current = int(row[0] or 0)
            for number, script in enumerate(MIGRATIONS, start=1):
                if number <= current:
                    continue
                for statement in script.split(";"):
                    if statement.strip():
                        conn.execute(statement)
                conn.execute("INSERT INTO schema_version (version) VALUES (?)", (number,))
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()
