from __future__ import annotations

import os
import signal
import sys
import threading
import time
from typing import Any, cast

import pytest

from reelferry import jobs
from reelferry.app import App
from reelferry.engine.media import FFmpeg, probe
from reelferry.errors import ErrorCode, ReelferryError
from reelferry.jobs import JobContext, Worker, run_job
from reelferry.models import JobKind, JobStatus, SourceStatus
from reelferry.registry import load_operations
from reelferry.store import get_source, insert_source

from .conftest import run_jobs

CALLS: list[str] = []


@pytest.fixture
def test_handler(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    CALLS.clear()

    def handler(app: App, ctx: JobContext) -> dict[str, Any]:
        def step(name: str, value: str) -> str:
            CALLS.append(name)
            return value

        first = ctx.step("first", lambda: step("first", "one"))
        if ctx.input.get("crash") and not ctx.input.get("crashed_once"):
            # Simulate a process dying after the first step finished.
            raise SystemExit("worker died")
        second = ctx.step("second", lambda: step("second", "two"))
        return {"first": first, "second": second}

    monkeypatch.setitem(jobs.HANDLERS, JobKind.SUGGEST_CLIPS, handler)
    return CALLS


def test_job_runs_and_records_result(app: App, test_handler: list[str]) -> None:
    job = app.jobs.enqueue(JobKind.SUGGEST_CLIPS, {})
    run_jobs(app)
    done = app.jobs.get(job.id)
    assert done.status is JobStatus.SUCCEEDED
    assert done.result == {"first": "one", "second": "two"}
    assert done.progress == 1
    assert [s.name for s in done.steps] == ["first", "second"]


def test_interrupted_job_resumes_after_last_finished_step(
    app: App, test_handler: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    job = app.jobs.enqueue(JobKind.SUGGEST_CLIPS, {"crash": True})
    owner = "wrk_dead"
    assert app.jobs.claim(owner) == job.id
    with pytest.raises(SystemExit):
        run_job(app, job.id, owner)
    assert app.jobs.get(job.id).status is JobStatus.RUNNING
    assert test_handler == ["first"]

    # The lease of the dead worker expires; another worker resumes the job.
    app.db.execute("UPDATE jobs SET lease_expires_at = ? WHERE id = ?", (time.time() - 1, job.id))
    app.db.execute(
        "UPDATE jobs SET input = json_set(input, '$.crashed_once', json('true')) WHERE id = ?",
        (job.id,),
    )
    run_jobs(app)
    done = app.jobs.get(job.id)
    assert done.status is JobStatus.SUCCEEDED
    assert test_handler == ["first", "second"]  # the first step was not repeated
    assert done.attempts == 2


def test_live_lease_is_not_stolen(app: App, test_handler: list[str]) -> None:
    job = app.jobs.enqueue(JobKind.SUGGEST_CLIPS, {})
    assert app.jobs.claim("wrk_a") == job.id
    assert app.jobs.claim("wrk_b") is None


def test_cancel_queued_job(app: App) -> None:
    job = app.jobs.enqueue(JobKind.SUGGEST_CLIPS, {})
    canceled = app.jobs.request_cancel(job.id)
    assert canceled.status is JobStatus.CANCELED
    assert app.jobs.claim("wrk") is None


def test_cancel_running_job(app: App, monkeypatch: pytest.MonkeyPatch) -> None:
    def slow(app: App, ctx: JobContext) -> dict[str, Any]:
        app.jobs.request_cancel(ctx.job_id)
        time.sleep(0.3)
        ctx.check_cancel()
        raise AssertionError("should have been canceled")

    monkeypatch.setitem(jobs.HANDLERS, JobKind.SUGGEST_CLIPS, slow)
    job = app.jobs.enqueue(JobKind.SUGGEST_CLIPS, {})
    run_jobs(app)
    assert app.jobs.get(job.id).status is JobStatus.CANCELED


def test_failed_job_keeps_error(app: App, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(app: App, ctx: JobContext) -> dict[str, Any]:
        raise ValueError("boom")

    monkeypatch.setitem(jobs.HANDLERS, JobKind.SUGGEST_CLIPS, broken)
    job = app.jobs.enqueue(JobKind.SUGGEST_CLIPS, {})
    run_jobs(app)
    failed = app.jobs.get(job.id)
    assert failed.status is JobStatus.FAILED
    assert failed.error is not None
    assert "boom" in failed.error["message"]


def test_gives_up_after_repeated_crashes(app: App) -> None:
    job = app.jobs.enqueue(JobKind.SUGGEST_CLIPS, {})
    app.db.execute(
        "UPDATE jobs SET status = 'running', attempts = 3, lease_expires_at = 0 WHERE id = ?",
        (job.id,),
    )
    run_jobs(app)
    failed = app.jobs.get(job.id)
    assert failed.status is JobStatus.FAILED
    assert failed.error is not None
    assert "gave up" in failed.error["message"]


def test_giving_up_runs_the_end_hook(app: App) -> None:
    load_operations()
    source = insert_source(
        app.db, kind="url", uri="https://example.com/v", title="v", status=SourceStatus.DOWNLOADING
    )
    job = app.jobs.enqueue(JobKind.DOWNLOAD_SOURCE, {"url": source.uri}, source_id=source.id)
    app.db.execute(
        "UPDATE jobs SET status = 'running', attempts = 3, lease_expires_at = 0 WHERE id = ?",
        (job.id,),
    )
    run_jobs(app)
    assert app.jobs.get(job.id).status is JobStatus.FAILED
    assert get_source(app.db, source.id).status is SourceStatus.FAILED


def test_stopped_worker_hands_its_job_to_the_next_worker(
    app: App, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = threading.Event()
    release = threading.Event()

    def blocking(app: App, ctx: JobContext) -> dict[str, Any]:
        if not ctx.input.get("resumed"):
            started.set()
            release.wait(5)
        return {"attempt": ctx.attempts}

    monkeypatch.setitem(jobs.HANDLERS, JobKind.SUGGEST_CLIPS, blocking)
    job = app.jobs.enqueue(JobKind.SUGGEST_CLIPS, {})
    first = Worker(app, concurrency=1)
    first.start()
    assert started.wait(5)
    first.stop(timeout=0.1)
    app.db.execute(
        "UPDATE jobs SET input = json_set(input, '$.resumed', json('true')) WHERE id = ?",
        (job.id,),
    )
    assert app.jobs.claim("wrk_next") == job.id
    run_job(app, job.id, "wrk_next")
    release.set()
    done = app.jobs.get(job.id)
    assert done.status is JobStatus.SUCCEEDED
    assert done.result == {"attempt": 1}


def test_idle_worker_waits_for_a_job_held_by_a_dead_worker(
    app: App, test_handler: list[str]
) -> None:
    job = app.jobs.enqueue(JobKind.SUGGEST_CLIPS, {})
    assert app.jobs.claim("wrk_dead") == job.id

    worker = Worker(App(app.paths), exit_when_idle=True, idle_grace=0.0)
    runner = threading.Thread(target=worker.run_until_idle)
    runner.start()
    time.sleep(1.5)
    assert runner.is_alive()

    app.db.execute("UPDATE jobs SET lease_expires_at = 0 WHERE id = ?", (job.id,))
    runner.join(10)
    assert not runner.is_alive()
    assert app.jobs.get(job.id).status is JobStatus.SUCCEEDED


CHILD_SLEEPS = "import os, time; print(os.getpid(), flush=True); time.sleep(60)"
CHILD_TICKS = (
    "import os, time\nprint(os.getpid(), flush=True)\n"
    "while True:\n    print('tick', flush=True)\n    time.sleep(0.05)"
)


def _child_job(monkeypatch: pytest.MonkeyPatch, script: str) -> tuple[list[int], threading.Event]:
    pids: list[int] = []
    started = threading.Event()

    def handler(app: App, ctx: JobContext) -> dict[str, Any]:
        def on_line(line: str) -> None:
            if not pids:
                pids.append(int(line))
                started.set()

        ctx.run_process([sys.executable, "-c", script], on_line=on_line)
        return {}

    monkeypatch.setitem(jobs.HANDLERS, JobKind.SUGGEST_CLIPS, handler)
    return pids, started


def _gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    return False


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_stopping_a_worker_kills_its_child_and_keeps_the_job_resumable(
    app: App, monkeypatch: pytest.MonkeyPatch
) -> None:
    pids, started = _child_job(monkeypatch, CHILD_SLEEPS)
    job = app.jobs.enqueue(JobKind.SUGGEST_CLIPS, {})
    worker = Worker(app, concurrency=1)
    worker.start()
    assert started.wait(10)
    child = pids[0]
    # A terminal Ctrl-C goes to the foreground process group; the child is not in it.
    assert os.getpgid(child) != os.getpgid(0)

    worker.stop(timeout=5)
    assert _gone(child)
    stopped = app.jobs.get(job.id)
    assert stopped.status is JobStatus.RUNNING
    assert stopped.error is None
    assert app.jobs.claim("wrk_next") == job.id


def test_canceling_a_job_kills_its_child(app: App, monkeypatch: pytest.MonkeyPatch) -> None:
    pids, started = _child_job(monkeypatch, CHILD_TICKS)
    job = app.jobs.enqueue(JobKind.SUGGEST_CLIPS, {})
    worker = Worker(App(app.paths), concurrency=1)
    worker.start()
    try:
        assert started.wait(10)
        jobs.cancel(app, job.id)
        assert app.jobs.wait(job.id, 10).status is JobStatus.CANCELED
        deadline = time.monotonic() + 5
        while not _gone(pids[0]) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert _gone(pids[0])
    finally:
        worker.stop()


def _wait_gone(pid: int) -> bool:
    deadline = time.monotonic() + 5
    while not _gone(pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    return _gone(pid)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
@pytest.mark.parametrize("signame", ["SIGTERM", "SIGHUP"])
def test_termination_signals_kill_job_children_before_the_host_exits(
    app: App, monkeypatch: pytest.MonkeyPatch, signame: str
) -> None:
    pids, started = _child_job(monkeypatch, CHILD_SLEEPS)
    job = app.jobs.enqueue(JobKind.SUGGEST_CLIPS, {})
    worker = Worker(app, concurrency=1)
    with pytest.raises(KeyboardInterrupt), jobs.stop_on_signals():
        worker.start()
        assert started.wait(10)
        os.kill(os.getpid(), getattr(signal, signame))
        time.sleep(10)
    assert _wait_gone(pids[0])

    worker.stop()
    assert app.jobs.get(job.id).status is JobStatus.RUNNING
    assert app.jobs.claim("wrk_next") == job.id
    assert app.jobs.get(job.id).attempts == 1


def test_a_step_failing_while_its_worker_stops_is_an_interruption(
    app: App, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = threading.Event()

    def handler(app: App, ctx: JobContext) -> dict[str, Any]:
        started.set()
        while not ctx.stopping.is_set():
            time.sleep(0.01)
        raise ReelferryError(ErrorCode.UPSTREAM_ERROR, "the merge child was killed")

    monkeypatch.setitem(jobs.HANDLERS, JobKind.SUGGEST_CLIPS, handler)
    job = app.jobs.enqueue(JobKind.SUGGEST_CLIPS, {})
    worker = Worker(app, concurrency=1)
    worker.start()
    assert started.wait(10)
    worker.stop()
    stopped = app.jobs.get(job.id)
    assert stopped.status is JobStatus.RUNNING
    assert stopped.error is None
    assert app.jobs.claim("wrk_next") == job.id


def test_graceful_stops_never_use_up_the_attempts(
    app: App, monkeypatch: pytest.MonkeyPatch
) -> None:
    started = threading.Event()

    def handler(app: App, ctx: JobContext) -> dict[str, Any]:
        if ctx.input.get("finish"):
            return {"attempt": ctx.attempts}
        started.set()
        while True:
            ctx.check_cancel()
            time.sleep(0.01)

    monkeypatch.setitem(jobs.HANDLERS, JobKind.SUGGEST_CLIPS, handler)
    job = app.jobs.enqueue(JobKind.SUGGEST_CLIPS, {})
    for _ in range(jobs.MAX_ATTEMPTS + 1):
        started.clear()
        worker = Worker(app, concurrency=1)
        worker.start()
        assert started.wait(10)
        worker.stop()
    app.db.execute(
        "UPDATE jobs SET input = json_set(input, '$.finish', json('true')) WHERE id = ?",
        (job.id,),
    )
    run_jobs(app)
    done = app.jobs.get(job.id)
    assert done.status is JobStatus.SUCCEEDED
    assert done.result == {"attempt": 1}


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
def test_probe_runs_outside_the_terminal_process_group(tmp_path: Any) -> None:
    fake = tmp_path / "ffprobe"
    fake.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "open(sys.argv[-1] + '.pgid', 'w').write(str(os.getpgid(0)))\n"
        "print(json.dumps({'streams': [{'codec_type': 'audio'}], 'format': {'duration': '1'}}))\n"
    )
    fake.chmod(0o755)
    media = tmp_path / "talk.wav"
    media.write_bytes(b"")
    info = probe(FFmpeg("ffmpeg", str(fake), "test", frozenset(), frozenset()), media)
    assert info["duration"] == 1
    assert int((tmp_path / "talk.wav.pgid").read_text()) != os.getpgid(0)


def test_a_second_interrupt_while_stopping_does_not_deadlock() -> None:
    stopping = jobs.StopSignal()
    errors: list[BaseException] = []

    class SignalDuringKill:
        signaled = False

        def kill(self) -> None:
            if not self.signaled:
                self.signaled = True
                stopping.set()

    def stop() -> None:
        try:
            with stopping.track(SignalDuringKill()):  # type: ignore[arg-type]
                stopping.set()
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=stop, daemon=True)
    thread.start()
    thread.join(5)
    assert not thread.is_alive()
    assert errors == []
    assert stopping.is_set()


@pytest.mark.timeout(60)
def test_wait_job_starts_a_worker_when_the_one_holding_the_job_dies(
    app: App, test_handler: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    from reelferry.operations import jobs as job_ops

    started: list[Worker] = []

    def ensure_worker(app: App) -> None:
        # Stands in for the detached worker the real ensure_worker spawns.
        if app.jobs.live_workers() == 0:
            worker = Worker(App(app.paths), concurrency=1)
            worker.start()
            started.append(worker)

    monkeypatch.setattr(job_ops, "ensure_worker", ensure_worker)
    job = app.jobs.enqueue(JobKind.SUGGEST_CLIPS, {})
    app.jobs.register_worker("wrk_doomed")
    assert app.jobs.claim("wrk_doomed") == job.id

    def worker_dies() -> None:
        time.sleep(1.5)
        app.db.execute("UPDATE workers SET heartbeat_at = 0 WHERE id = 'wrk_doomed'")
        app.db.execute("UPDATE jobs SET lease_expires_at = 0 WHERE id = ?", (job.id,))
        app.db.close()

    killer = threading.Thread(target=worker_dies)
    killer.start()
    try:
        out = cast(
            job_ops.WaitJobOutput,
            load_operations()["wait_job"].run(App(app.paths), {"job_id": job.id, "timeout": 20}),
        )
    finally:
        killer.join()
        for worker in started:
            worker.stop()
    assert out.done
    assert out.job.status is JobStatus.SUCCEEDED


def _source_job(app: App) -> tuple[Any, Any]:
    load_operations()
    source = insert_source(
        app.db, kind="url", uri="https://example.com/v", title="v", status=SourceStatus.DOWNLOADING
    )
    job = app.jobs.enqueue(JobKind.DOWNLOAD_SOURCE, {"url": source.uri}, source_id=source.id)
    return source, job


def test_canceling_a_released_job_ends_it_at_once(app: App) -> None:
    source, job = _source_job(app)
    assert app.jobs.claim("wrk_stopped") == job.id
    app.jobs.release("wrk_stopped")

    canceled = jobs.cancel(app, job.id)
    assert canceled.status is JobStatus.CANCELED
    assert canceled.finished_at is not None
    assert get_source(app.db, source.id).status is SourceStatus.FAILED
    assert app.jobs.claim("wrk_next") is None


def test_canceling_a_job_of_a_crashed_worker_ends_it_at_once(app: App) -> None:
    _, job = _source_job(app)
    assert app.jobs.claim("wrk_crashed") == job.id
    app.db.execute("UPDATE jobs SET lease_expires_at = 0 WHERE id = ?", (job.id,))
    assert jobs.cancel(app, job.id).status is JobStatus.CANCELED


def test_canceling_a_job_held_by_a_live_worker_leaves_it_to_that_worker(app: App) -> None:
    _, job = _source_job(app)
    assert app.jobs.claim("wrk_live") == job.id
    held = jobs.cancel(app, job.id)
    assert held.status is JobStatus.RUNNING
    assert app.jobs.cancel_requested(job.id)
