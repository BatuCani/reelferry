from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from reelferry import jobs
from reelferry.app import App
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
    assert done.result == {"attempt": 2}


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
