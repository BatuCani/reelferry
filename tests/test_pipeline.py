"""End to end through the operation registry with real ffmpeg and synthetic media."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from reelferry.app import App
from reelferry.engine.media import FFmpeg, detect_ffmpeg, probe
from reelferry.errors import ErrorCode, ReelferryError
from reelferry.models import ClipStatus, JobStatus
from reelferry.registry import load_operations

from .conftest import FakeChat, finished

pytestmark = pytest.mark.ffmpeg

SUGGESTIONS = [
    {
        "start": 2.0,
        "end": 10.0,
        "title": "Three habits that changed my life",
        "score": 91,
        "reason": "Clear promise and list",
        "emphasis": ["three", "habits"],
    },
    {
        "start": 11.0,
        "end": 19.0,
        "title": "Say no to meetings",
        "score": 77,
        "reason": "Surprising payoff",
        "emphasis": ["ten"],
    },
]


def call(app: App, name: str, **payload: Any) -> Any:
    return load_operations()[name].run(app, payload)


@pytest.fixture
def ready(app: App, talk_video: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    chat = FakeChat(SUGGESTIONS)
    app.chat_client_factory = lambda app: chat
    source = call(app, "add_source", uri=str(talk_video), title="Talk").source
    job = finished(app, call(app, "transcribe", source_id=source.id, language="en").job)
    assert job.status is JobStatus.SUCCEEDED, job.error
    return {"source": source, "chat": chat}


def test_source_and_transcript(app: App, ready: dict[str, Any]) -> None:
    source = ready["source"]
    assert source.status.value == "ready"
    assert source.width == 1280 and source.height == 720
    assert source.duration == pytest.approx(20, abs=0.2)
    got = call(app, "get_source", source_id=source.id).source
    assert got.transcript_version == 1
    assert got.language == "en"

    seg = call(app, "get_transcript", source_id=source.id)
    assert seg.segments[0].text == "Welcome back to the show."
    words = call(app, "get_transcript", source_id=source.id, format="words", start=4, end=6).words
    assert words and all(w.end > 4 and w.start < 6 for w in words)
    text = call(app, "get_transcript", source_id=source.id, format="text").text
    assert text.startswith("Welcome back")


def test_edit_transcript_creates_new_version(app: App, ready: dict[str, Any]) -> None:
    source = ready["source"]
    words = call(app, "get_transcript", source_id=source.id, format="words").words
    out = call(
        app,
        "edit_transcript",
        source_id=source.id,
        edits=[{"index": 0, "text": "Hello"}],
        replacements=[{"find": "twenty", "replace": "20"}],
        note="fix",
    )
    assert out.changed_words == 2
    assert out.info.version == 2
    updated = call(app, "get_transcript", source_id=source.id, format="words").words
    assert updated[0].text == "Hello"
    assert (updated[0].start, updated[0].end) == (words[0].start, words[0].end)
    old = call(app, "get_transcript", source_id=source.id, format="words", version=1).words
    assert old[0].text == "Welcome"
    with pytest.raises(ReelferryError) as err:
        call(app, "edit_transcript", source_id=source.id, edits=[{"index": 9999, "text": "x"}])
    assert err.value.code is ErrorCode.INVALID_INPUT


def test_suggest_create_render_preview_export(app: App, ready: dict[str, Any]) -> None:
    source = ready["source"]
    job = finished(
        app,
        call(
            app, "suggest_clips", source_id=source.id, goal="habits", min_seconds=8, max_seconds=20
        ).job,
    )
    assert job.status is JobStatus.SUCCEEDED, job.error
    assert job.cost_usd == pytest.approx(0.0021)
    assert job.result is not None
    candidates = job.result["candidates"]
    assert [c["title"] for c in candidates] == [s["title"] for s in SUGGESTIONS]
    listed = call(app, "list_candidates", source_id=source.id).candidates
    assert {c.id for c in listed} == {c["id"] for c in candidates}

    ff = detect_ffmpeg()
    burn = ff.has_libass
    clip = call(
        app,
        "create_clip",
        candidate_id=candidates[0]["id"],
        caption_style="pop",
        burn_captions=burn,
    ).clip
    assert clip.status is ClipStatus.DRAFT
    assert clip.score == 91 and clip.emphasis == ["three", "habits"]

    out = call(app, "render_clip", clip_id=clip.id)
    assert out.clip.status is ClipStatus.RENDERING
    job = finished(app, out.job)
    assert job.status is JobStatus.SUCCEEDED, job.error
    rendered = call(app, "get_clip", clip_id=clip.id).clip
    assert rendered.status is ClipStatus.RENDERED
    render = rendered.latest_render
    assert render is not None
    assert render.burned_captions is burn
    video = Path(render.video_path)
    info = probe(ff, video)
    assert (info["width"], info["height"]) == (1080, 1920)
    assert info["duration"] == pytest.approx(clip.duration, abs=0.15)
    assert set(render.captions) == {"ass", "srt", "vtt", "json"}
    assert all(Path(p).exists() for p in render.captions.values())

    preview = call(app, "preview_clip", clip_id=clip.id, frames=4, columns=2)
    assert preview.from_render
    assert Path(preview.sheet_path).stat().st_size > 1000

    exported = call(app, "export_captions", clip_id=clip.id, format="vtt", include_content=True)
    assert exported.content.startswith("WEBVTT")
    assert Path(exported.path).exists()

    updated = call(app, "update_clip", clip_id=clip.id, start=clip.start + 1, format="16:9").clip
    assert updated.status is ClipStatus.DRAFT
    assert updated.revision == clip.revision + 1
    assert not call(app, "preview_clip", clip_id=clip.id, frames=2).from_render

    hand = call(
        app,
        "create_clip",
        source_id=source.id,
        start=1.0,
        end=6.0,
        format="16:9",
        framing="fit_blur",
        captions_enabled=False,
    ).clip
    job = finished(app, call(app, "render_clip", clip_id=hand.id).job)
    assert job.status is JobStatus.SUCCEEDED, job.error
    info = probe(ff, Path(call(app, "get_clip", clip_id=hand.id).clip.latest_render.video_path))
    assert (info["width"], info["height"]) == (1920, 1080)

    clips = call(app, "list_clips", source_id=source.id, min_score=80).clips
    assert [c.id for c in clips] == [clip.id]


def test_soft_subtitles_without_burn_in(app: App, ready: dict[str, Any]) -> None:
    source = ready["source"]
    clip = call(
        app,
        "create_clip",
        source_id=source.id,
        start=0.4,
        end=6.0,
        burn_captions=False,
        caption_style="karaoke",
    ).clip
    job = finished(app, call(app, "render_clip", clip_id=clip.id).job)
    assert job.status is JobStatus.SUCCEEDED, job.error
    render = call(app, "get_clip", clip_id=clip.id).clip.latest_render
    assert render is not None and not render.burned_captions
    import subprocess

    streams = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            render.video_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert "subtitle" in streams


def test_unrendered_preview_is_framed_like_the_render(app: App, ready: dict[str, Any]) -> None:
    ff = detect_ffmpeg()
    clip = call(
        app,
        "create_clip",
        source_id=ready["source"].id,
        start=1.0,
        end=5.0,
        format="9:16",
        framing="crop",
        crop_x=0.1,
        captions_enabled=False,
    ).clip
    left = call(app, "preview_clip", clip_id=clip.id, frames=1, columns=1)
    assert not left.from_render
    info = probe(ff, Path(left.sheet_path))
    assert info["height"] > info["width"]

    call(app, "update_clip", clip_id=clip.id, crop_x=0.9)
    right = call(app, "preview_clip", clip_id=clip.id, frames=1, columns=1)
    assert Path(right.sheet_path).read_bytes() != Path(left.sheet_path).read_bytes()


def test_burn_in_without_libass_is_a_clear_error(
    app: App, ready: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    real = detect_ffmpeg()
    no_ass = FFmpeg(real.ffmpeg, real.ffprobe, real.version, frozenset(), real.encoders)
    monkeypatch.setattr(app, "ffmpeg", lambda: no_ass)
    clip = call(app, "create_clip", source_id=ready["source"].id, start=0.5, end=4).clip
    with pytest.raises(ReelferryError) as err:
        call(app, "render_clip", clip_id=clip.id)
    assert err.value.code is ErrorCode.MISSING_DEPENDENCY
    assert err.value.hint and "libass" in err.value.hint
    assert call(app, "get_clip", clip_id=clip.id).clip.status is ClipStatus.DRAFT


def test_suggest_needs_key(
    app: App, ready: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY")
    with pytest.raises(ReelferryError) as err:
        call(app, "suggest_clips", source_id=ready["source"].id)
    assert err.value.code is ErrorCode.MISSING_API_KEY


def test_transcribe_requires_existing_source(app: App) -> None:
    with pytest.raises(ReelferryError) as err:
        call(app, "transcribe", source_id="src_missing")
    assert err.value.code is ErrorCode.NOT_FOUND


def test_add_source_rejects_missing_file(app: App, tmp_path: Path) -> None:
    with pytest.raises(ReelferryError) as err:
        call(app, "add_source", uri=str(tmp_path / "nope.mp4"))
    assert err.value.code is ErrorCode.NOT_FOUND


def test_cancel_render(app: App, ready: dict[str, Any]) -> None:
    clip = call(
        app, "create_clip", source_id=ready["source"].id, start=0.5, end=10, burn_captions=False
    ).clip
    out = call(app, "render_clip", clip_id=clip.id)
    canceled = call(app, "cancel_job", job_id=out.job.id).job
    assert canceled.status is JobStatus.CANCELED
    assert call(app, "get_clip", clip_id=clip.id).clip.status is ClipStatus.DRAFT
    job = finished(app, call(app, "render_clip", clip_id=clip.id).job)
    assert job.status is JobStatus.SUCCEEDED, job.error
