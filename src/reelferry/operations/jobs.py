"""Jobs: long-running work (downloads, transcription, suggestions, renders)."""

from __future__ import annotations

from pydantic import Field

from reelferry.app import App
from reelferry.errors import ErrorCode
from reelferry.jobs import cancel, ensure_worker
from reelferry.models import Job, JobStatus, Model
from reelferry.operations.transcripts import JobOutput
from reelferry.registry import operation


class JobIdInput(Model):
    job_id: str


class WaitJobInput(Model):
    job_id: str
    timeout: float = Field(
        default=60, ge=0, le=3600, description="Seconds to wait before returning anyway."
    )


class WaitJobOutput(Model):
    job: Job
    done: bool = Field(description="True when the job reached a final state.")


class ListJobsInput(Model):
    status: JobStatus | None = None
    limit: int = Field(default=20, ge=1, le=500)


class JobList(Model):
    jobs: list[Job]


@operation(
    name="get_job",
    area="job",
    verb="get",
    summary="Status, progress, result and cost of a job",
    input=JobIdInput,
    output=JobOutput,
    method="GET",
    path="/v1/jobs/{job_id}",
    errors=(ErrorCode.NOT_FOUND,),
    positional=("job_id",),
    read_only=True,
)
def get_job(app: App, inp: JobIdInput) -> JobOutput:
    """Current state of a job. Finished jobs carry `result` or `error`."""
    return JobOutput(job=app.jobs.get(inp.job_id))


@operation(
    name="wait_job",
    area="job",
    verb="wait",
    summary="Wait until a job finishes or the timeout passes",
    input=WaitJobInput,
    output=WaitJobOutput,
    method="POST",
    path="/v1/jobs/{job_id}/wait",
    errors=(ErrorCode.NOT_FOUND,),
    positional=("job_id",),
    read_only=True,
)
def wait_job(app: App, inp: WaitJobInput) -> WaitJobOutput:
    """Block until the job succeeds, fails or is canceled, or until `timeout` seconds pass.
    Call again if done is false."""
    if not app.jobs.get(inp.job_id).status.terminal:
        ensure_worker(app)
    job = app.jobs.wait(inp.job_id, inp.timeout)
    return WaitJobOutput(job=job, done=job.status.terminal)


@operation(
    name="cancel_job",
    area="job",
    verb="cancel",
    summary="Cancel a queued or running job",
    input=JobIdInput,
    output=JobOutput,
    method="POST",
    path="/v1/jobs/{job_id}/cancel",
    errors=(ErrorCode.NOT_FOUND,),
    positional=("job_id",),
    destructive=True,
)
def cancel_job(app: App, inp: JobIdInput) -> JobOutput:
    """Queued jobs stop at once; running jobs stop at their next checkpoint (seconds)."""
    return JobOutput(job=cancel(app, inp.job_id))


@operation(
    name="list_jobs",
    area="job",
    verb="list",
    summary="Recent jobs, newest first",
    input=ListJobsInput,
    output=JobList,
    method="GET",
    path="/v1/jobs",
    read_only=True,
)
def list_jobs(app: App, inp: ListJobsInput) -> JobList:
    """Recent jobs, optionally only those in one status."""
    return JobList(jobs=app.jobs.list(status=inp.status, limit=inp.limit))
