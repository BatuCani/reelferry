"""Sources: the long videos clips are cut from."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pydantic import Field

from reelferry.app import App
from reelferry.engine.media import probe
from reelferry.errors import ErrorCode, ReelferryError
from reelferry.jobs import JobContext, ensure_worker, job_end_hook, job_handler
from reelferry.models import Job, JobKind, JobStatus, Model, Source, SourceStatus
from reelferry.registry import operation
from reelferry.store import get_source, insert_source, list_sources, update_source

VIDEO_SUFFIXES = {".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".flv", ".mp3", ".wav", ".m4a"}


class AddSourceInput(Model):
    uri: str = Field(description="Path to a local video file, or a public http(s) link.")
    title: str | None = Field(
        default=None, description="Display title. Default: file or page title."
    )


class SourceWithJob(Model):
    source: Source
    job: Job | None = Field(description="Download job for links; null for local files.")


class SourceIdInput(Model):
    source_id: str


class SourceOutput(Model):
    source: Source


class ListSourcesInput(Model):
    limit: int = Field(default=50, ge=1, le=500)


class SourceList(Model):
    sources: list[Source]


def _is_url(uri: str) -> bool:
    return bool(re.match(r"^https?://", uri.strip(), re.IGNORECASE))


@operation(
    name="add_source",
    area="source",
    verb="add",
    summary="Add a long video from a local file or a public link",
    input=AddSourceInput,
    output=SourceWithJob,
    method="POST",
    path="/v1/sources",
    errors=(ErrorCode.NOT_FOUND, ErrorCode.MEDIA_ERROR, ErrorCode.MISSING_DEPENDENCY),
    positional=("uri",),
    returns_job=True,
)
def add_source(app: App, inp: AddSourceInput) -> SourceWithJob:
    """Register a source video. Local files are probed right away and referenced in place
    (never copied). Links (YouTube, Twitch VODs, anything yt-dlp supports) are downloaded
    by a background job; wait for it before transcribing."""
    uri = inp.uri.strip()
    if _is_url(uri):
        source = insert_source(
            app.db,
            kind="url",
            uri=uri,
            title=inp.title or uri,
            status=SourceStatus.DOWNLOADING,
        )
        job = app.jobs.enqueue(JobKind.DOWNLOAD_SOURCE, {"url": uri}, source_id=source.id)
        ensure_worker(app)
        return SourceWithJob(source=source, job=job)

    path = Path(uri).expanduser()
    if not path.is_file():
        raise ReelferryError(ErrorCode.NOT_FOUND, f"file not found: {uri}")
    path = path.resolve()
    if path.suffix.lower() not in VIDEO_SUFFIXES:
        raise ReelferryError(
            ErrorCode.INVALID_INPUT,
            f"unsupported file type {path.suffix!r}",
            hint=f"Supported: {', '.join(sorted(VIDEO_SUFFIXES))}",
        )
    info = probe(app.ffmpeg(), path)
    source = insert_source(
        app.db,
        kind="file",
        uri=str(path),
        title=inp.title or path.stem,
        status=SourceStatus.READY,
        path=str(path),
        probe=info,
    )
    return SourceWithJob(source=source, job=None)


@operation(
    name="get_source",
    area="source",
    verb="get",
    summary="Get one source",
    input=SourceIdInput,
    output=SourceOutput,
    method="GET",
    path="/v1/sources/{source_id}",
    errors=(ErrorCode.NOT_FOUND,),
    positional=("source_id",),
    read_only=True,
)
def get_source_op(app: App, inp: SourceIdInput) -> SourceOutput:
    """Status, duration, size and transcript version of a source."""
    return SourceOutput(source=get_source(app.db, inp.source_id))


@operation(
    name="list_sources",
    area="source",
    verb="list",
    summary="List sources, newest first",
    input=ListSourcesInput,
    output=SourceList,
    method="GET",
    path="/v1/sources",
    read_only=True,
)
def list_sources_op(app: App, inp: ListSourcesInput) -> SourceList:
    """All registered source videos, newest first."""
    return SourceList(sources=list_sources(app.db, inp.limit))


@job_handler(JobKind.DOWNLOAD_SOURCE)
def download_source(app: App, ctx: JobContext) -> dict[str, Any]:
    import yt_dlp

    assert ctx.source_id is not None
    source_id = ctx.source_id
    url = str(ctx.input["url"])
    target_dir = app.paths.sources_dir / source_id
    target_dir.mkdir(parents=True, exist_ok=True)
    ff = app.ffmpeg()

    def hook(status: dict[str, Any]) -> None:
        if status.get("status") == "downloading":
            total = status.get("total_bytes") or status.get("total_bytes_estimate") or 0
            done = status.get("downloaded_bytes") or 0
            ctx.progress(done / total if total else 0.0, "downloading")

    options: dict[str, Any] = {
        "format": "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/bv*[height<=1080]+ba/b[height<=1080]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(target_dir / "source.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "progress_hooks": [hook],
        "ffmpeg_location": str(Path(ff.ffmpeg).parent),
        "continuedl": True,
    }

    def fetch() -> dict[str, Any]:
        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                info = ydl.extract_info(url, download=True)
                if info is None:
                    raise ReelferryError(ErrorCode.MEDIA_ERROR, f"nothing to download at {url}")
                filename = ydl.prepare_filename(info)
        except yt_dlp.utils.DownloadError as exc:
            raise ReelferryError(ErrorCode.UPSTREAM_ERROR, f"download failed: {exc}") from exc
        path = Path(filename)
        if not path.exists():
            candidates = sorted(target_dir.glob("source.*"), key=lambda p: p.stat().st_size)
            if not candidates:
                raise ReelferryError(ErrorCode.MEDIA_ERROR, "download produced no file")
            path = candidates[-1]
        return {"path": str(path), "title": info.get("title") or url}

    ctx.phase(0.0, 0.95)
    fetched = ctx.step("download", fetch)
    path = Path(fetched["path"])
    info = probe(ff, path)
    current = get_source(app.db, source_id)
    title = current.title if current.title != url else str(fetched["title"])
    update_source(
        app.db,
        source_id,
        path=str(path),
        title=title,
        status=SourceStatus.READY.value,
        duration=info["duration"],
        width=info["width"],
        height=info["height"],
        fps=info["fps"],
        has_audio=info["has_audio"],
        error=None,
    )
    return {"source_id": source_id, "path": str(path), "duration": info["duration"]}


@job_end_hook(JobKind.DOWNLOAD_SOURCE)
def _download_ended(app: App, job: Job) -> None:
    if job.source_id and job.status in (JobStatus.FAILED, JobStatus.CANCELED):
        message = (job.error or {}).get("message") or job.status.value
        update_source(app.db, job.source_id, status=SourceStatus.FAILED.value, error=message)
