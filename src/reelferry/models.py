"""Domain models returned by every interface (MCP, CLI, API)."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

AspectFormat = Literal["9:16", "16:9"]
Framing = Literal["crop", "fit_blur"]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceStatus(StrEnum):
    DOWNLOADING = "downloading"
    READY = "ready"
    FAILED = "failed"


class Source(Model):
    id: str
    kind: Literal["file", "url"]
    uri: str = Field(description="Original file path or URL.")
    title: str
    path: str | None = Field(description="Local media file, once available.")
    status: SourceStatus
    duration: float | None = Field(description="Seconds.")
    width: int | None
    height: int | None
    fps: float | None
    has_audio: bool | None
    language: str | None = Field(description="Spoken language, known after transcription.")
    transcript_version: int | None = Field(description="Latest transcript version, if any.")
    error: str | None
    created_at: str


class Word(Model):
    text: str
    start: float = Field(description="Seconds from the start of the source.")
    end: float
    probability: float | None = None


class IndexedWord(Word):
    index: int = Field(description="Position in the transcript; use it with edit_transcript.")


class TranscriptSegment(Model):
    start: float
    end: float
    text: str
    first_word: int
    last_word: int


class TranscriptInfo(Model):
    source_id: str
    version: int
    language: str | None
    engine: str = Field(description="'local' or 'openrouter'.")
    model: str
    alignment: str = Field(description="'ctc' (forced alignment), 'whisper', or 'mixed'.")
    word_count: int
    duration: float
    created_at: str
    note: str | None


class CandidateBase(Model):
    start: float
    end: float
    title: str = Field(description="Hook title for the clip.")
    score: int = Field(ge=0, le=100, description="How strong the moment is, 0-100.")
    reason: str
    emphasis: list[str] = Field(
        default_factory=list, description="Words to emphasize in captions (names, numbers)."
    )


class ClipCandidate(CandidateBase):
    id: str
    source_id: str
    job_id: str | None
    duration: float
    created_at: str


class ClipStatus(StrEnum):
    DRAFT = "draft"
    RENDERING = "rendering"
    RENDERED = "rendered"
    FAILED = "failed"


class Render(Model):
    id: str
    clip_id: str
    job_id: str | None
    clip_revision: int
    video_path: str
    captions: dict[str, str] = Field(description="Caption sidecar files by format.")
    format: AspectFormat
    width: int
    height: int
    duration: float
    size_bytes: int
    caption_style: str
    burned_captions: bool
    created_at: str


class Clip(Model):
    id: str
    source_id: str
    candidate_id: str | None
    title: str
    start: float
    end: float
    duration: float
    format: AspectFormat
    framing: Framing
    crop_x: float = Field(description="Horizontal crop anchor, 0 = left, 0.5 = center, 1 = right.")
    caption_style: str
    burn_captions: bool
    captions_enabled: bool
    status: ClipStatus
    score: int | None
    reason: str | None
    emphasis: list[str]
    revision: int = Field(description="Increments on every edit; renders record the revision.")
    latest_render: Render | None
    created_at: str
    updated_at: str


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"

    @property
    def terminal(self) -> bool:
        return self in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELED)


class JobKind(StrEnum):
    DOWNLOAD_SOURCE = "download_source"
    TRANSCRIBE = "transcribe"
    SUGGEST_CLIPS = "suggest_clips"
    RENDER_CLIP = "render_clip"


class JobStep(Model):
    name: str
    status: Literal["done", "running"]


class Job(Model):
    id: str
    kind: JobKind
    status: JobStatus
    progress: float = Field(ge=0, le=1)
    message: str | None
    source_id: str | None
    clip_id: str | None
    steps: list[JobStep]
    result: dict[str, Any] | None
    error: dict[str, Any] | None
    cost_usd: float = Field(description="Paid API cost of this job so far.")
    attempts: int
    created_at: str
    started_at: str | None
    finished_at: str | None


class CaptionStyleInfo(Model):
    id: str
    name: str
    description: str
    font: str
    uppercase: bool
    animation: str
    position: str
    words_per_page: int
    lines: int
