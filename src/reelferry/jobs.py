"""Durable, resumable jobs.

A job is a row in SQLite. A worker claims it with a lease and keeps the lease
alive with a heartbeat. Work is split into named steps whose results are stored
as they finish, so a job interrupted by a crash or restart resumes at the last
finished step when any worker picks up its expired lease.

Any process may run a worker: ``reelferry serve`` and ``reelferry mcp`` run one
in-process, and the CLI starts a detached one on demand.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from reelferry.db import Database, dumps, new_id, now_iso
from reelferry.engine.media import own_process_group
from reelferry.errors import ErrorCode, JobCanceled, JobInterrupted, ReelferryError, not_found
from reelferry.models import Job, JobKind, JobStatus, JobStep

if TYPE_CHECKING:
    from reelferry.app import App

log = logging.getLogger(__name__)

LEASE_SECONDS = 30.0
HEARTBEAT_SECONDS = 5.0
WORKER_ALIVE_SECONDS = 20.0
MAX_ATTEMPTS = 3

JobHandler = Callable[["App", "JobContext"], dict[str, Any]]
HANDLERS: dict[JobKind, JobHandler] = {}


def job_handler(kind: JobKind) -> Callable[[JobHandler], JobHandler]:
    def register(fn: JobHandler) -> JobHandler:
        HANDLERS[kind] = fn
        return fn

    return register


def _loads(value: str | None) -> Any:
    return None if value is None else json.loads(value)


class JobStore:
    def __init__(self, db: Database) -> None:
        self.db = db

    def _job(self, row: Any) -> Job:
        steps: dict[str, dict[str, Any]] = json.loads(row["steps"])
        return Job(
            id=row["id"],
            kind=JobKind(row["kind"]),
            status=JobStatus(row["status"]),
            progress=row["progress"],
            message=row["message"],
            source_id=row["source_id"],
            clip_id=row["clip_id"],
            steps=[JobStep(name=name, status=step["status"]) for name, step in steps.items()],
            result=_loads(row["result"]),
            error=_loads(row["error"]),
            cost_usd=round(row["cost_usd"], 6),
            attempts=row["attempts"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
        )

    def enqueue(
        self,
        kind: JobKind,
        payload: dict[str, Any],
        *,
        source_id: str | None = None,
        clip_id: str | None = None,
    ) -> Job:
        job_id = new_id("job")
        ts = now_iso()
        self.db.execute(
            "INSERT INTO jobs (id, kind, status, input, source_id, clip_id, created_at, updated_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                job_id,
                kind.value,
                JobStatus.QUEUED.value,
                dumps(payload),
                source_id,
                clip_id,
                ts,
                ts,
            ),
        )
        return self.get(job_id)

    def get(self, job_id: str) -> Job:
        row = self.db.one("SELECT * FROM jobs WHERE id = ?", (job_id,))
        if row is None:
            raise not_found("job", job_id)
        return self._job(row)

    def list(self, *, status: JobStatus | None, limit: int) -> list[Job]:
        if status:
            rows = self.db.all(
                "SELECT * FROM jobs WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status.value, limit),
            )
        else:
            rows = self.db.all("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,))
        return [self._job(row) for row in rows]

    def active_for(
        self, kind: JobKind, *, source_id: str | None, clip_id: str | None
    ) -> Job | None:
        row = self.db.one(
            "SELECT * FROM jobs WHERE kind = ? AND status IN ('queued', 'running')"
            " AND source_id IS ? AND clip_id IS ? ORDER BY created_at DESC LIMIT 1",
            (kind.value, source_id, clip_id),
        )
        return self._job(row) if row else None

    def request_cancel(self, job_id: str) -> Job:
        with self.db.tx() as conn:
            row = conn.execute("SELECT status FROM jobs WHERE id = ?", (job_id,)).fetchone()
            if row is None:
                raise not_found("job", job_id)
            status = JobStatus(row["status"])
            if status is JobStatus.QUEUED:
                ts = now_iso()
                conn.execute(
                    "UPDATE jobs SET status = ?, cancel_requested = 1, finished_at = ?,"
                    " updated_at = ?, message = 'canceled before start' WHERE id = ?",
                    (JobStatus.CANCELED.value, ts, ts, job_id),
                )
            elif status is JobStatus.RUNNING:
                ts = now_iso()
                conn.execute(
                    "UPDATE jobs SET cancel_requested = 1, updated_at = ? WHERE id = ?",
                    (ts, job_id),
                )
                # Released by a stopped worker, or left by a crashed one: no one will reach
                # a checkpoint, so end it here.
                conn.execute(
                    "UPDATE jobs SET status = ?, finished_at = ?, message = 'canceled',"
                    " lease_owner = NULL, lease_expires_at = NULL WHERE id = ?"
                    " AND (lease_owner IS NULL OR lease_expires_at < ?)",
                    (JobStatus.CANCELED.value, ts, job_id, time.time()),
                )
        return self.get(job_id)

    def cancel_requested(self, job_id: str) -> bool:
        row = self.db.one("SELECT cancel_requested FROM jobs WHERE id = ?", (job_id,))
        return bool(row and row["cancel_requested"])

    def claim(self, owner: str, *, only_job_id: str | None = None) -> str | None:
        """Atomically take the oldest runnable job (queued, or running with a dead lease)."""
        now = time.time()
        with self.db.tx() as conn:
            sql = (
                "SELECT id FROM jobs WHERE"
                " (status = 'queued' OR (status = 'running' AND lease_expires_at < ?))"
            )
            params: list[Any] = [now]
            if only_job_id:
                sql += " AND id = ?"
                params.append(only_job_id)
            row = conn.execute(sql + " ORDER BY created_at LIMIT 1", params).fetchone()
            if row is None:
                return None
            job_id: str = row["id"]
            ts = now_iso()
            conn.execute(
                "UPDATE jobs SET status = 'running', lease_owner = ?, lease_expires_at = ?,"
                " attempts = attempts + 1, started_at = COALESCE(started_at, ?), updated_at = ?"
                " WHERE id = ?",
                (owner, now + LEASE_SECONDS, ts, ts, job_id),
            )
        return job_id

    def renew(self, owner: str) -> None:
        self.db.execute(
            "UPDATE jobs SET lease_expires_at = ? WHERE lease_owner = ? AND status = 'running'",
            (time.time() + LEASE_SECONDS, owner),
        )

    def release(self, owner: str) -> None:
        """Hand the running jobs of a stopping worker back so any worker resumes them now.
        A clean hand-back does not count toward the attempts that end in giving up."""
        self.db.execute(
            "UPDATE jobs SET lease_owner = NULL, lease_expires_at = 0,"
            " attempts = MAX(attempts - 1, 0) WHERE lease_owner = ? AND status = 'running'",
            (owner,),
        )

    def set_progress(self, job_id: str, progress: float, message: str | None) -> None:
        self.db.execute(
            "UPDATE jobs SET progress = ?, message = COALESCE(?, message), updated_at = ?"
            " WHERE id = ?",
            (max(0.0, min(1.0, progress)), message, now_iso(), job_id),
        )

    def add_cost(self, job_id: str, usd: float) -> None:
        self.db.execute("UPDATE jobs SET cost_usd = cost_usd + ? WHERE id = ?", (usd, job_id))

    def steps(self, job_id: str) -> dict[str, dict[str, Any]]:
        row = self.db.one("SELECT steps FROM jobs WHERE id = ?", (job_id,))
        return {} if row is None else dict(json.loads(row["steps"]))

    def save_step(self, job_id: str, name: str, status: str, result: Any = None) -> None:
        with self.db.tx() as conn:
            row = conn.execute("SELECT steps FROM jobs WHERE id = ?", (job_id,)).fetchone()
            steps = json.loads(row["steps"])
            steps[name] = {"status": status, "result": result}
            conn.execute(
                "UPDATE jobs SET steps = ?, updated_at = ? WHERE id = ?",
                (dumps(steps), now_iso(), job_id),
            )

    def finish(
        self,
        job_id: str,
        status: JobStatus,
        *,
        owner: str,
        result: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
        message: str | None = None,
    ) -> bool:
        """Record the final state. False if ``owner`` no longer holds the job's lease."""
        ts = now_iso()
        updated = self.db.execute(
            "UPDATE jobs SET status = ?, result = ?, error = ?, message = COALESCE(?, message),"
            " progress = CASE WHEN ? = 'succeeded' THEN 1 ELSE progress END,"
            " finished_at = ?, updated_at = ?, lease_owner = NULL, lease_expires_at = NULL"
            " WHERE id = ? AND lease_owner = ?",
            (
                status.value,
                None if result is None else dumps(result),
                None if error is None else dumps(error),
                message,
                status.value,
                ts,
                ts,
                job_id,
                owner,
            ),
        )
        return updated == 1

    def wait(self, job_id: str, timeout: float, *, poll: float = 0.5) -> Job:
        deadline = time.monotonic() + timeout
        while True:
            job = self.get(job_id)
            if job.status.terminal or time.monotonic() >= deadline:
                return job
            time.sleep(min(poll, max(0.0, deadline - time.monotonic())))

    # Worker registry

    def register_worker(self, worker_id: str) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO workers (id, pid, started_at, heartbeat_at)"
            " VALUES (?, ?, ?, ?)",
            (worker_id, os.getpid(), now_iso(), time.time()),
        )

    def heartbeat_worker(self, worker_id: str) -> None:
        self.db.execute(
            "UPDATE workers SET heartbeat_at = ? WHERE id = ?", (time.time(), worker_id)
        )

    def unregister_worker(self, worker_id: str) -> None:
        self.db.execute("DELETE FROM workers WHERE id = ?", (worker_id,))

    def live_workers(self) -> int:
        cutoff = time.time() - WORKER_ALIVE_SECONDS
        self.db.execute("DELETE FROM workers WHERE heartbeat_at < ?", (cutoff,))
        row = self.db.one("SELECT COUNT(*) FROM workers")
        return int(row[0]) if row else 0

    def pending_count(self) -> int:
        """Jobs that need a worker: queued, or running for a worker that is gone."""
        now = time.time()
        row = self.db.one(
            "SELECT COUNT(*) FROM jobs WHERE status = 'queued' OR (status = 'running' AND"
            " (lease_expires_at < ? OR lease_owner NOT IN"
            " (SELECT id FROM workers WHERE heartbeat_at >= ?)))",
            (now, now - WORKER_ALIVE_SECONDS),
        )
        return int(row[0]) if row else 0


class StopSignal:
    """A worker's stop flag. Setting it also kills the job subprocesses tracked under it."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.RLock()
        self._procs: set[subprocess.Popen[str]] = set()

    def is_set(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: float) -> bool:
        return self._event.wait(timeout)

    def set(self) -> None:
        with self._lock:
            self._event.set()
            for proc in self._procs:
                proc.kill()

    @contextmanager
    def track(self, proc: subprocess.Popen[str]) -> Iterator[None]:
        with self._lock:
            if self._event.is_set():
                proc.kill()
            self._procs.add(proc)
        try:
            yield
        finally:
            with self._lock:
                self._procs.discard(proc)


class JobContext:
    """What a job handler sees: its input, step memory, progress, cancellation."""

    def __init__(
        self, app: App, job_id: str, owner: str, stopping: StopSignal | None = None
    ) -> None:
        self.app = app
        self.job_id = job_id
        self.owner = owner
        self.stopping = stopping or StopSignal()
        row = app.db.one("SELECT * FROM jobs WHERE id = ?", (job_id,))
        assert row is not None
        self.kind = JobKind(row["kind"])
        self.attempts: int = row["attempts"]
        self.input: dict[str, Any] = json.loads(row["input"])
        self.source_id: str | None = row["source_id"]
        self.clip_id: str | None = row["clip_id"]
        self._steps = app.jobs.steps(job_id)
        self._last_progress_write = 0.0
        self._last_cancel_check = 0.0
        self._progress_base = 0.0
        self._progress_span = 1.0

    def step[T](
        self, name: str, fn: Callable[[], T], *, valid: Callable[[T], bool] | None = None
    ) -> T:
        """Run ``fn`` once. If an earlier attempt finished this step, reuse its result
        (unless ``valid`` says the stored result is unusable, e.g. a deleted file)."""
        stored = self._steps.get(name)
        if stored and stored.get("status") == "done":
            result = cast(T, stored.get("result"))
            if valid is None or valid(result):
                log.info("job %s: reusing finished step %s", self.job_id, name)
                return result
        self.check_cancel()
        self.app.jobs.save_step(self.job_id, name, "running")
        result = fn()
        self.app.jobs.save_step(self.job_id, name, "done", result)
        self._steps[name] = {"status": "done", "result": result}
        return result

    def phase(self, start: float, end: float) -> None:
        """Map following ``progress`` calls (0..1) into the overall range ``start..end``."""
        self._progress_base = start
        self._progress_span = end - start
        self.progress(0.0)

    def progress(self, fraction: float, message: str | None = None) -> None:
        now = time.monotonic()
        if message is None and now - self._last_progress_write < 0.5 and fraction < 1:
            return
        self._last_progress_write = now
        overall = self._progress_base + self._progress_span * max(0.0, min(1.0, fraction))
        self.app.jobs.set_progress(self.job_id, overall, message)
        self.check_cancel()

    def check_cancel(self) -> None:
        if self.stopping.is_set():
            raise JobInterrupted
        now = time.monotonic()
        if now - self._last_cancel_check < 0.25:
            return
        self._last_cancel_check = now
        if self.app.jobs.cancel_requested(self.job_id):
            raise JobCanceled

    def add_cost(self, usd: float) -> None:
        if usd:
            self.app.jobs.add_cost(self.job_id, usd)

    def artifact(self, name: str) -> Path:
        directory = self.app.paths.artifacts_dir / self.job_id
        directory.mkdir(parents=True, exist_ok=True)
        return directory / name

    def run_process(
        self,
        cmd: Sequence[str],
        *,
        on_line: Callable[[str], None] | None = None,
        cwd: Path | None = None,
    ) -> str:
        """Run a subprocess, stream stdout lines to ``on_line``, kill it on cancel or stop."""
        proc = subprocess.Popen(
            list(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            **own_process_group(),
        )
        stderr_chunks: list[str] = []

        def drain_stderr() -> None:
            assert proc.stderr is not None
            stderr_chunks.extend(proc.stderr)

        drainer = threading.Thread(target=drain_stderr, daemon=True)
        drainer.start()
        try:
            with self.stopping.track(proc):
                assert proc.stdout is not None
                for line in proc.stdout:
                    if on_line:
                        on_line(line.rstrip("\n"))
                    self.check_cancel()
                proc.wait()
        except BaseException:
            proc.kill()
            proc.wait()
            raise
        finally:
            drainer.join(timeout=5)
        stderr = "".join(stderr_chunks)
        if proc.returncode != 0:
            raise ReelferryError(
                ErrorCode.MEDIA_ERROR,
                f"{Path(cmd[0]).name} exited with code {proc.returncode}",
                details={"stderr": stderr[-2000:]},
            )
        return stderr


class Worker:
    """Claims and runs jobs on background threads until stopped."""

    def __init__(
        self,
        app: App,
        *,
        worker_id: str | None = None,
        concurrency: int | None = None,
        only_job_id: str | None = None,
        exit_when_idle: bool = False,
        idle_grace: float = 5.0,
    ) -> None:
        self.app = app
        self.id = worker_id or new_id("wrk")
        self.concurrency = concurrency or app.settings.max_parallel_jobs
        self.only_job_id = only_job_id
        self.exit_when_idle = exit_when_idle
        self.idle_grace = idle_grace
        self._stop = StopSignal()
        self._threads: list[threading.Thread] = []
        self._busy = 0
        self._busy_lock = threading.Lock()
        self._idle_since: float | None = None

    def start(self) -> None:
        _HOSTED.add(self)
        self.app.jobs.register_worker(self.id)
        heart = threading.Thread(target=self._heartbeat, name="reelferry-heartbeat", daemon=True)
        heart.start()
        self._threads.append(heart)
        for index in range(self.concurrency):
            thread = threading.Thread(
                target=self._loop, name=f"reelferry-worker-{index}", daemon=True
            )
            thread.start()
            self._threads.append(thread)

    def interrupt(self) -> None:
        """Start stopping at once: kill job children and stop claiming new jobs."""
        self._stop.set()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        _HOSTED.discard(self)
        for thread in self._threads:
            thread.join(timeout=timeout)
        try:
            self.app.jobs.release(self.id)
            self.app.jobs.unregister_worker(self.id)
        finally:
            self.app.db.close()

    def run_until_idle(self) -> None:
        """Run in the foreground until there is nothing left to do."""
        self.start()
        try:
            while not self._stop.wait(0.5):
                if not self.exit_when_idle:
                    continue
                if self._busy == 0 and self.app.jobs.pending_count() == 0:
                    if self._idle_since is None:
                        self._idle_since = time.monotonic()
                    elif time.monotonic() - self._idle_since > self.idle_grace:
                        break
                else:
                    self._idle_since = None
        finally:
            self.stop()

    def _heartbeat(self) -> None:
        while not self._stop.wait(HEARTBEAT_SECONDS):
            try:
                self.app.jobs.heartbeat_worker(self.id)
                self.app.jobs.renew(self.id)
            except Exception:
                log.exception("heartbeat failed")
        self.app.db.close()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                job_id = self.app.jobs.claim(self.id, only_job_id=self.only_job_id)
            except Exception:
                log.exception("claim failed")
                job_id = None
            if job_id is None:
                if self.only_job_id and self.app.jobs.get(self.only_job_id).status.terminal:
                    self._stop.set()
                    break
                self._stop.wait(0.5)
                continue
            with self._busy_lock:
                self._busy += 1
            try:
                run_job(self.app, job_id, self.id, self._stop)
            finally:
                with self._busy_lock:
                    self._busy -= 1
        self.app.db.close()


def run_job(app: App, job_id: str, owner: str, stopping: StopSignal | None = None) -> None:
    ctx = JobContext(app, job_id, owner, stopping)
    handler = HANDLERS.get(ctx.kind)
    try:
        if ctx.attempts > MAX_ATTEMPTS:
            raise ReelferryError(
                ErrorCode.JOB_FAILED, f"gave up after {MAX_ATTEMPTS} interrupted attempts"
            )
        if handler is None:
            raise ReelferryError(ErrorCode.INTERNAL, f"no handler for job kind {ctx.kind.value}")
        result = handler(app, ctx)
    except JobCanceled:
        ended = app.jobs.finish(job_id, JobStatus.CANCELED, owner=owner, message="canceled")
    except Exception as exc:
        if isinstance(exc, JobInterrupted) or ctx.stopping.is_set():
            log.info("job %s interrupted by worker stop; it resumes later", job_id)
            return
        if isinstance(exc, ReelferryError):
            error, message = exc.to_dict(), exc.message
        else:
            log.exception("job %s crashed", job_id)
            error = {"code": ErrorCode.INTERNAL.value, "message": f"{type(exc).__name__}: {exc}"}
            message = str(exc)
        ended = app.jobs.finish(job_id, JobStatus.FAILED, owner=owner, error=error, message=message)
    else:
        ended = app.jobs.finish(
            job_id, JobStatus.SUCCEEDED, owner=owner, result=result, message="done"
        )
    if ended:
        notify_job_end(app, app.jobs.get(job_id))


_HOSTED: set[Worker] = set()


@contextmanager
def stop_on_signals() -> Iterator[None]:
    """For a process that hosts workers: Ctrl-C, SIGTERM and SIGHUP (terminal closed)
    interrupt the workers at once, killing their job children, and then raise
    KeyboardInterrupt so the host runs its normal ``Worker.stop`` path."""

    def handle(signum: int, _frame: Any) -> None:
        for worker in list(_HOSTED):
            worker.interrupt()
        if signum == getattr(signal, "SIGHUP", None):
            signal.raise_signal(signal.SIGTERM)
            return
        raise KeyboardInterrupt

    names = ("SIGINT", "SIGTERM", "SIGHUP")
    signums = [getattr(signal, name) for name in names if hasattr(signal, name)]
    previous = {signum: signal.signal(signum, handle) for signum in signums}
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def cancel(app: App, job_id: str) -> Job:
    """Cancel a job. A queued job, or a running one no worker holds, ends at once (and its
    end hook runs here); a held one stops at its worker's next checkpoint."""
    before = app.jobs.get(job_id)
    job = app.jobs.request_cancel(job_id)
    if not before.status.terminal and job.status is JobStatus.CANCELED:
        notify_job_end(app, job)
    elif job.status is JobStatus.RUNNING:
        ensure_worker(app)
    return job


EndHook = Callable[["App", Job], None]
END_HOOKS: dict[JobKind, EndHook] = {}


def job_end_hook(kind: JobKind) -> Callable[[EndHook], EndHook]:
    """Register a callback that runs once a job of this kind ends, in any final state."""

    def register(fn: EndHook) -> EndHook:
        END_HOOKS[kind] = fn
        return fn

    return register


def notify_job_end(app: App, job: Job) -> None:
    hook = END_HOOKS.get(job.kind)
    if hook is not None:
        try:
            hook(app, job)
        except Exception:
            log.exception("job end hook failed for %s", job.id)


def ensure_worker(app: App) -> None:
    """Make sure some process will pick up queued jobs; start a detached worker if needed.

    The new worker is registered before it starts, so callers right after this see it."""
    if os.environ.get("REELFERRY_NO_AUTOWORKER") or app.jobs.live_workers() > 0:
        return
    worker_id = new_id("wrk")
    app.jobs.register_worker(worker_id)
    app.paths.logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = app.paths.logs_dir / "worker.log"
    with log_path.open("ab") as log_file:
        kwargs: dict[str, Any] = {
            "stdin": subprocess.DEVNULL,
            "stdout": log_file,
            "stderr": log_file,
            "close_fds": True,
            "env": {
                **os.environ,
                "REELFERRY_DATA_DIR": str(app.paths.data_dir),
                "REELFERRY_MEDIA_DIR": str(app.paths.media_dir),
            },
        }
        kwargs.update(own_process_group())
        if sys.platform == "win32":
            kwargs["creationflags"] |= 0x00000008
        try:
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "reelferry",
                    "worker",
                    "--exit-when-idle",
                    "--id",
                    worker_id,
                ],
                **kwargs,
            )
        except BaseException:
            app.jobs.unregister_worker(worker_id)
            raise
