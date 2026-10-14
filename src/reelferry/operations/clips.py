"""Clips: suggest moments, create and adjust clips, render and preview them."""

from __future__ import annotations

import contextlib
import re
from pathlib import Path
from typing import Any

from pydantic import Field

from reelferry.app import App
from reelferry.captions.build import build_pages, write_captions
from reelferry.captions.styles import get_style
from reelferry.config import resolve_model
from reelferry.engine.media import probe, progress_seconds
from reelferry.engine.render import OUTPUT_SIZES, RenderPlan, contact_sheet_cmd, render_cmd
from reelferry.engine.suggest import SuggestRequest, suggest
from reelferry.errors import ErrorCode, ReelferryError
from reelferry.jobs import JobContext, ensure_worker, job_end_hook, job_handler
from reelferry.models import (
    AspectFormat,
    Clip,
    ClipCandidate,
    ClipStatus,
    Framing,
    Job,
    JobKind,
    JobStatus,
    Model,
)
from reelferry.operations.transcripts import JobOutput
from reelferry.registry import operation
from reelferry.store import (
    get_candidate,
    get_clip,
    get_source,
    insert_candidate,
    insert_clip,
    insert_render,
    list_candidates,
    list_clips,
    load_transcript,
    ready_source,
    update_clip,
)

MIN_CLIP_SECONDS = 1.0


class SuggestClipsInput(Model):
    source_id: str
    goal: str = Field(
        default="the most engaging, self-contained moments with a strong hook",
        description="What to look for, e.g. 'the 5 funniest moments' or 'tips about pricing'.",
    )
    count: int = Field(default=5, ge=1, le=50)
    min_seconds: float = Field(default=15, ge=3, le=1800)
    max_seconds: float = Field(default=60, ge=5, le=1800)
    model: str | None = Field(
        default=None,
        description="'quality', 'budget' or an OpenRouter model id. Default: the setting.",
    )
    language: str | None = Field(
        default=None, description="Language for titles. Default: the transcript's language."
    )


class ListCandidatesInput(Model):
    source_id: str
    job_id: str | None = Field(default=None, description="Only candidates from this suggest job.")


class CandidateList(Model):
    candidates: list[ClipCandidate]


class CreateClipInput(Model):
    candidate_id: str | None = Field(default=None, description="Create from a suggestion.")
    source_id: str | None = Field(default=None, description="Or cut by hand: source plus times.")
    start: float | None = Field(default=None, ge=0, description="Seconds.")
    end: float | None = Field(default=None, gt=0, description="Seconds.")
    title: str | None = None
    format: AspectFormat | None = Field(default=None, description="Default: the setting (9:16).")
    framing: Framing = Field(
        default="crop",
        description="'crop' fills the frame; 'fit_blur' shows the whole picture on a blurred "
        "background.",
    )
    crop_x: float = Field(default=0.5, ge=0, le=1, description="Crop anchor: 0 left, 1 right.")
    caption_style: str | None = Field(default=None, description="See list_caption_styles.")
    burn_captions: bool = Field(default=True, description="Burn captions into the picture.")
    captions_enabled: bool = Field(default=True, description="False renders without captions.")
    emphasis: list[str] | None = Field(default=None, description="Words to highlight.")


class ClipOutput(Model):
    clip: Clip


class ClipIdInput(Model):
    clip_id: str


class UpdateClipInput(Model):
    clip_id: str
    title: str | None = None
    start: float | None = Field(default=None, ge=0)
    end: float | None = Field(default=None, gt=0)
    format: AspectFormat | None = None
    framing: Framing | None = None
    crop_x: float | None = Field(default=None, ge=0, le=1)
    caption_style: str | None = None
    burn_captions: bool | None = None
    captions_enabled: bool | None = None
    emphasis: list[str] | None = None


class RenderClipInput(Model):
    clip_id: str
    caption_style: str | None = Field(default=None, description="Override and save the style.")
    format: AspectFormat | None = Field(default=None, description="Override and save the format.")
    burn_captions: bool | None = Field(default=None, description="Override and save burn-in.")


class RenderOutput(Model):
    clip: Clip
    job: Job


class PreviewClipInput(Model):
    clip_id: str
    frames: int = Field(default=9, ge=1, le=36)
    columns: int = Field(default=3, ge=1, le=6)


class PreviewOutput(Model):
    clip_id: str
    sheet_path: str = Field(description="JPEG contact sheet of evenly spaced frames.")
    frames: int
    from_render: bool = Field(description="True if taken from the rendered video.")


class ListClipsInput(Model):
    source_id: str | None = None
    status: ClipStatus | None = None
    query: str | None = Field(default=None, description="Search titles and reasons.")
    min_score: int | None = Field(default=None, ge=0, le=100)
    limit: int = Field(default=50, ge=1, le=500)


class ClipList(Model):
    clips: list[Clip]


@operation(
    name="suggest_clips",
    area="clip",
    verb="suggest",
    summary="Let AI find the best moments (background job)",
    input=SuggestClipsInput,
    output=JobOutput,
    method="POST",
    path="/v1/sources/{source_id}/suggestions",
    errors=(ErrorCode.NOT_FOUND, ErrorCode.TRANSCRIPT_MISSING, ErrorCode.MISSING_API_KEY),
    positional=("source_id",),
    returns_job=True,
)
def suggest_clips(app: App, inp: SuggestClipsInput) -> JobOutput:
    """Find clip candidates with an LLM over OpenRouter (default Claude Opus 5.5; 'budget'
    uses Gemini 3.8 Flash). Each candidate has start/end snapped to word edges, a hook
    title, a 0-100 score and a reason. The finished job's result lists the candidates;
    turn one into a clip with create_clip."""
    if inp.min_seconds > inp.max_seconds:
        raise ReelferryError(ErrorCode.INVALID_INPUT, "min_seconds is larger than max_seconds")
    load_transcript(app.db, inp.source_id)
    app.openrouter_key()
    job = app.jobs.enqueue(
        JobKind.SUGGEST_CLIPS, inp.model_dump(mode="json"), source_id=inp.source_id
    )
    ensure_worker(app)
    return JobOutput(job=job)


@job_handler(JobKind.SUGGEST_CLIPS)
def run_suggest(app: App, ctx: JobContext) -> dict[str, Any]:
    inp = SuggestClipsInput.model_validate(ctx.input)
    source = get_source(app.db, inp.source_id)
    info, words = load_transcript(app.db, inp.source_id)
    model = resolve_model(inp.model or app.settings.suggest_model)

    def ask() -> dict[str, Any]:
        ctx.progress(0.1, f"asking {model}")
        picked, cost, used = suggest(
            app.chat_client(),
            model=model,
            words=words,
            req=SuggestRequest(
                goal=inp.goal,
                count=inp.count,
                min_seconds=inp.min_seconds,
                max_seconds=inp.max_seconds,
                language=inp.language or info.language,
                title=source.title,
            ),
            duration=source.duration or info.duration,
        )
        ctx.add_cost(cost)
        stored = [
            insert_candidate(app.db, source_id=source.id, job_id=ctx.job_id, **c.model_dump())
            for c in picked
        ]
        return {
            "model": used,
            "candidates": [c.model_dump(mode="json") for c in stored],
        }

    result = ctx.step("suggest", ask)
    if not result["candidates"]:
        ctx.progress(1.0, "no moments matched the request")
    return {"source_id": source.id, "transcript_version": info.version, **result}


@operation(
    name="list_candidates",
    area="clip",
    verb="candidates",
    summary="List clip suggestions for a source",
    input=ListCandidatesInput,
    output=CandidateList,
    method="GET",
    path="/v1/sources/{source_id}/suggestions",
    errors=(ErrorCode.NOT_FOUND,),
    positional=("source_id",),
    read_only=True,
)
def list_candidates_op(app: App, inp: ListCandidatesInput) -> CandidateList:
    """Suggestions from earlier suggest_clips runs, best first."""
    return CandidateList(candidates=list_candidates(app.db, inp.source_id, inp.job_id))


def _check_range(app: App, source_id: str, start: float, end: float) -> None:
    source = get_source(app.db, source_id)
    if end - start < MIN_CLIP_SECONDS:
        raise ReelferryError(
            ErrorCode.INVALID_INPUT, f"clip must be at least {MIN_CLIP_SECONDS:.0f} s long"
        )
    if source.duration is not None and start >= source.duration:
        raise ReelferryError(
            ErrorCode.INVALID_INPUT,
            f"start {start:.2f} s is past the end of the source ({source.duration:.2f} s)",
        )


@operation(
    name="create_clip",
    area="clip",
    verb="create",
    summary="Create a clip from a suggestion or from start and end times",
    input=CreateClipInput,
    output=ClipOutput,
    method="POST",
    path="/v1/clips",
    errors=(ErrorCode.NOT_FOUND,),
)
def create_clip(app: App, inp: CreateClipInput) -> ClipOutput:
    """Create a clip. Pass candidate_id, or source_id with start and end in seconds.
    Nothing is rendered yet; call render_clip when the clip looks right."""
    settings = app.settings
    style = get_style(inp.caption_style or settings.default_caption_style)
    if inp.candidate_id:
        cand = get_candidate(app.db, inp.candidate_id)
        source_id = cand.source_id
        start = cand.start if inp.start is None else inp.start
        end = cand.end if inp.end is None else inp.end
        title = inp.title or cand.title
        score: int | None = cand.score
        reason: str | None = cand.reason
        emphasis = cand.emphasis if inp.emphasis is None else inp.emphasis
    else:
        if not inp.source_id or inp.start is None or inp.end is None:
            raise ReelferryError(
                ErrorCode.INVALID_INPUT, "give candidate_id, or source_id with start and end"
            )
        source_id, start, end = inp.source_id, inp.start, inp.end
        title = inp.title or "Untitled clip"
        score = reason = None
        emphasis = inp.emphasis or []
    ready_source(app.db, source_id)
    _check_range(app, source_id, start, end)
    clip = insert_clip(
        app.db,
        source_id=source_id,
        candidate_id=inp.candidate_id,
        title=title,
        start=round(start, 3),
        end=round(end, 3),
        format=inp.format or settings.default_format,
        framing=inp.framing,
        crop_x=inp.crop_x,
        caption_style=style.id,
        burn_captions=int(inp.burn_captions),
        captions_enabled=int(inp.captions_enabled),
        score=score,
        reason=reason,
        emphasis=emphasis,
    )
    return ClipOutput(clip=clip)


def _apply_update(app: App, clip: Clip, changes: dict[str, Any]) -> Clip:
    if not changes:
        return clip
    if "caption_style" in changes:
        get_style(changes["caption_style"])
    start = changes.get("start", clip.start)
    end = changes.get("end", clip.end)
    if "start" in changes or "end" in changes:
        _check_range(app, clip.source_id, start, end)
    for flag in ("burn_captions", "captions_enabled"):
        if flag in changes:
            changes[flag] = int(changes[flag])
    status = (
        ClipStatus.DRAFT.value if clip.status is not ClipStatus.RENDERING else clip.status.value
    )
    return update_clip(app.db, clip.id, bump_revision=True, status=status, **changes)


@operation(
    name="update_clip",
    area="clip",
    verb="update",
    summary="Change bounds, title, format, framing or caption style",
    input=UpdateClipInput,
    output=ClipOutput,
    method="PATCH",
    path="/v1/clips/{clip_id}",
    errors=(ErrorCode.NOT_FOUND, ErrorCode.CONFLICT),
    positional=("clip_id",),
)
def update_clip_op(app: App, inp: UpdateClipInput) -> ClipOutput:
    """Edit a clip. Only the given fields change. A rendered clip goes back to draft
    and needs render_clip again."""
    clip = get_clip(app.db, inp.clip_id)
    if clip.status is ClipStatus.RENDERING:
        raise ReelferryError(
            ErrorCode.CONFLICT, "clip is rendering", hint="Cancel the render job or wait for it."
        )
    changes = inp.model_dump(exclude={"clip_id"}, exclude_none=True)
    return ClipOutput(clip=_apply_update(app, clip, changes))


@operation(
    name="render_clip",
    area="clip",
    verb="render",
    summary="Render the clip to MP4 with captions (background job)",
    input=RenderClipInput,
    output=RenderOutput,
    method="POST",
    path="/v1/clips/{clip_id}/render",
    errors=(
        ErrorCode.NOT_FOUND,
        ErrorCode.CONFLICT,
        ErrorCode.TRANSCRIPT_MISSING,
        ErrorCode.MISSING_DEPENDENCY,
    ),
    positional=("clip_id",),
    returns_job=True,
)
def render_clip(app: App, inp: RenderClipInput) -> RenderOutput:
    """Render to an H.264 MP4 (1080x1920 for 9:16, 1920x1080 for 16:9). Captions come
    from the word-level transcript and are always written next to the video as ASS, SRT,
    VTT and JSON. With burn_captions they are burned into the picture (needs ffmpeg with
    libass); otherwise they are added as a soft subtitle track."""
    clip = get_clip(app.db, inp.clip_id)
    if clip.status is ClipStatus.RENDERING:
        running = app.jobs.active_for(JobKind.RENDER_CLIP, source_id=None, clip_id=clip.id)
        if running:
            ensure_worker(app)
            return RenderOutput(clip=clip, job=running)
    changes = inp.model_dump(exclude={"clip_id"}, exclude_none=True)
    changes = {k: v for k, v in changes.items() if getattr(clip, k) != v}
    clip = _apply_update(app, clip, changes)
    source = ready_source(app.db, clip.source_id)
    if source.width is None:
        raise ReelferryError(ErrorCode.MEDIA_ERROR, "source has no video stream to render")
    if clip.captions_enabled:
        load_transcript(app.db, clip.source_id)
        if clip.burn_captions:
            app.ffmpeg().require_libass()
    else:
        app.ffmpeg()
    job = app.jobs.enqueue(
        JobKind.RENDER_CLIP, {"clip_id": clip.id, "revision": clip.revision}, clip_id=clip.id
    )
    clip = update_clip(app.db, clip.id, bump_revision=False, status=ClipStatus.RENDERING.value)
    ensure_worker(app)
    return RenderOutput(clip=clip, job=job)


def _slug(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:48] or "clip"


@job_handler(JobKind.RENDER_CLIP)
def run_render(app: App, ctx: JobContext) -> dict[str, Any]:
    assert ctx.clip_id is not None
    clip = get_clip(app.db, ctx.clip_id)
    source = ready_source(app.db, clip.source_id)
    settings = app.settings
    ff = app.ffmpeg()
    style = get_style(clip.caption_style)
    size = OUTPUT_SIZES[clip.format]
    out_dir = app.paths.clips_dir / clip.id
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{_slug(clip.title)}-{clip.format.replace(':', 'x')}-r{clip.revision}"
    captions: dict[str, str] = {}
    if clip.captions_enabled:
        _, words = load_transcript(app.db, clip.source_id)
        pages, picked = build_pages(
            words, start=clip.start, end=clip.end, style=style, emphasis=clip.emphasis
        )
        captions = write_captions(out_dir, stem, pages=pages, words=picked, style=style, size=size)
    burn = clip.captions_enabled and clip.burn_captions
    if burn:
        ff.require_libass()
    output = out_dir / f"{stem}.mp4"
    # Each worker writes its own partial: a crashed worker's ffmpeg can outlive it and
    # must never write into the file this attempt publishes.
    partial = out_dir / f".{stem}.{ctx.owner}.partial.mp4"
    plan = RenderPlan(
        source=Path(str(source.path)),
        start=clip.start,
        end=clip.end,
        format=clip.format,
        framing=clip.framing,
        crop_x=clip.crop_x,
        source_has_audio=bool(source.has_audio),
        ass_file=Path(captions["ass"]).name if burn else None,
        soft_subtitles=captions.get("srt") if captions and not burn else None,
        fonts_dir=settings.fonts_dir,
        output=partial,
    )
    duration = clip.end - clip.start

    def on_line(line: str) -> None:
        done = progress_seconds(line)
        if done is not None:
            ctx.progress(done / duration, "rendering")

    ctx.progress(0.0, "rendering")
    ctx.run_process(render_cmd(ff, plan), on_line=on_line, cwd=out_dir)
    partial.replace(output)
    for stale in out_dir.glob(f".{stem}.*.partial.mp4"):
        with contextlib.suppress(OSError):
            stale.unlink()
    info = probe(ff, output)
    render = insert_render(
        app.db,
        clip_id=clip.id,
        job_id=ctx.job_id,
        clip_revision=clip.revision,
        video_path=str(output),
        captions=captions,
        format=clip.format,
        width=info["width"] or size[0],
        height=info["height"] or size[1],
        duration=info["duration"] or duration,
        size_bytes=output.stat().st_size,
        caption_style=style.id,
        burned_captions=int(burn),
    )
    update_clip(app.db, clip.id, bump_revision=False, status=ClipStatus.RENDERED.value)
    return {"clip_id": clip.id, "render": render.model_dump(mode="json")}


@job_end_hook(JobKind.RENDER_CLIP)
def _render_ended(app: App, job: Job) -> None:
    if job.clip_id is None or job.status is JobStatus.SUCCEEDED:
        return
    for leftover in (app.paths.clips_dir / job.clip_id).glob(".*.partial.mp4"):
        leftover.unlink(missing_ok=True)
    clip = get_clip(app.db, job.clip_id)
    if clip.status is ClipStatus.RENDERING:
        new_status = ClipStatus.FAILED if job.status is JobStatus.FAILED else ClipStatus.DRAFT
        update_clip(app.db, clip.id, bump_revision=False, status=new_status.value)


@operation(
    name="preview_clip",
    area="clip",
    verb="preview",
    summary="Contact sheet of still frames so an agent can see the clip",
    input=PreviewClipInput,
    output=PreviewOutput,
    method="POST",
    path="/v1/clips/{clip_id}/preview",
    errors=(ErrorCode.NOT_FOUND, ErrorCode.MEDIA_ERROR),
    positional=("clip_id",),
    image_field="sheet_path",
)
def preview_clip(app: App, inp: PreviewClipInput) -> PreviewOutput:
    """A JPEG grid of evenly spaced frames, framed like the render (format, framing,
    crop_x). Uses the latest render when it matches the clip's current revision (so burned
    captions are visible), else the source range."""
    import subprocess

    clip = get_clip(app.db, inp.clip_id)
    ff = app.ffmpeg()
    render = clip.latest_render
    from_render = bool(
        render and render.clip_revision == clip.revision and Path(render.video_path).exists()
    )
    if from_render and render is not None:
        media, start, duration = Path(render.video_path), 0.0, render.duration
    else:
        media = Path(str(ready_source(app.db, clip.source_id).path))
        start, duration = clip.start, clip.end - clip.start
    out_dir = app.paths.clips_dir / clip.id
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet = out_dir / f"preview-r{clip.revision}{'-render' if from_render else ''}.jpg"
    cmd = contact_sheet_cmd(
        ff,
        media=media,
        start=start,
        duration=duration,
        fmt=clip.format,
        framing=clip.framing,
        crop_x=clip.crop_x,
        frames=inp.frames,
        columns=min(inp.columns, inp.frames),
        output=sheet,
    )
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not sheet.exists():
        raise ReelferryError(
            ErrorCode.MEDIA_ERROR,
            "could not build the preview",
            details={"stderr": proc.stderr[-1000:]},
        )
    return PreviewOutput(
        clip_id=clip.id, sheet_path=str(sheet), frames=inp.frames, from_render=from_render
    )


@operation(
    name="list_clips",
    area="clip",
    verb="list",
    summary="Browse the clip library",
    input=ListClipsInput,
    output=ClipList,
    method="GET",
    path="/v1/clips",
    read_only=True,
)
def list_clips_op(app: App, inp: ListClipsInput) -> ClipList:
    """Clips with score, status and file paths, newest first."""
    return ClipList(
        clips=list_clips(
            app.db,
            source_id=inp.source_id,
            status=inp.status.value if inp.status else None,
            query=inp.query,
            min_score=inp.min_score,
            limit=inp.limit,
        )
    )


@operation(
    name="get_clip",
    area="clip",
    verb="get",
    summary="Everything about one clip, including its files",
    input=ClipIdInput,
    output=ClipOutput,
    method="GET",
    path="/v1/clips/{clip_id}",
    errors=(ErrorCode.NOT_FOUND,),
    positional=("clip_id",),
    read_only=True,
)
def get_clip_op(app: App, inp: ClipIdInput) -> ClipOutput:
    """A clip with its latest render (video path, caption files, size)."""
    return ClipOutput(clip=get_clip(app.db, inp.clip_id))
