"""Transcripts: word-level text with timings, the source of truth for captions."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from reelferry.app import App
from reelferry.engine.media import extract_audio_cmd, progress_seconds
from reelferry.engine.suggest import build_segments
from reelferry.errors import ErrorCode, ReelferryError
from reelferry.jobs import JobContext, ensure_worker, job_handler
from reelferry.models import (
    IndexedWord,
    Job,
    JobKind,
    Model,
    TranscriptInfo,
    TranscriptSegment,
    Word,
)
from reelferry.registry import operation
from reelferry.store import load_transcript, ready_source, save_transcript


class TranscribeInput(Model):
    source_id: str
    language: str | None = Field(
        default=None, description="ISO 639-1 code like 'en' or 'de'. Default: detect."
    )
    engine: Literal["local", "openrouter"] | None = Field(
        default=None, description="Default: the transcription_engine setting ('local')."
    )
    vocabulary: list[str] = Field(
        default_factory=list,
        description="Names and terms the speech model should spell correctly.",
    )


class JobOutput(Model):
    job: Job


class GetTranscriptInput(Model):
    source_id: str
    version: int | None = Field(default=None, description="Default: latest version.")
    start: float | None = Field(default=None, ge=0, description="Only words from here (seconds).")
    end: float | None = Field(default=None, ge=0, description="Only words up to here (seconds).")
    format: Literal["segments", "words", "text"] = Field(
        default="segments",
        description="'segments' (sentences with times), 'words' (every word with index and "
        "times), or 'text' (plain text).",
    )


class TranscriptOutput(Model):
    info: TranscriptInfo
    segments: list[TranscriptSegment] | None = None
    words: list[IndexedWord] | None = None
    text: str | None = None


class WordEdit(Model):
    index: int = Field(ge=0, description="Word index from get_transcript format=words.")
    text: str = Field(description="New text for the word. Empty string removes the word.")


class Replacement(Model):
    find: str = Field(min_length=1, description="Word or phrase to find (punctuation ignored).")
    replace: str = Field(description="Replacement. Timings of the matched words are kept.")
    match_case: bool = False


class EditTranscriptInput(Model):
    source_id: str
    edits: list[WordEdit] = Field(default_factory=list)
    replacements: list[Replacement] = Field(default_factory=list)
    note: str | None = Field(default=None, description="Why this edit was made.")


class EditTranscriptOutput(Model):
    info: TranscriptInfo
    changed_words: int


@operation(
    name="transcribe",
    area="transcript",
    verb="create",
    summary="Transcribe a source with exact word timings (background job)",
    input=TranscribeInput,
    output=JobOutput,
    method="POST",
    path="/v1/sources/{source_id}/transcript",
    errors=(ErrorCode.NOT_FOUND, ErrorCode.CONFLICT),
    positional=("source_id",),
    returns_job=True,
)
def transcribe(app: App, inp: TranscribeInput) -> JobOutput:
    """Start transcription. Runs Whisper large-v3-turbo locally (free; the model downloads
    once, about 1.6 GB) and refines every word's timing with forced alignment.
    Returns a job; poll it with wait_job. If a transcription for this source is already
    running, that job is returned instead of starting a second one."""
    source = ready_source(app.db, inp.source_id)
    if source.has_audio is False:
        raise ReelferryError(ErrorCode.MEDIA_ERROR, f"source {source.id!r} has no audio track")
    running = app.jobs.active_for(JobKind.TRANSCRIBE, source_id=source.id, clip_id=None)
    if running:
        ensure_worker(app)
        return JobOutput(job=running)
    job = app.jobs.enqueue(
        JobKind.TRANSCRIBE,
        inp.model_dump(mode="json"),
        source_id=source.id,
    )
    ensure_worker(app)
    return JobOutput(job=job)


def _segments_in_range(
    segments: list[TranscriptSegment], start: float | None, end: float | None
) -> list[TranscriptSegment]:
    return [
        s for s in segments if (start is None or s.end > start) and (end is None or s.start < end)
    ]


@operation(
    name="get_transcript",
    area="transcript",
    verb="get",
    summary="Read a transcript, whole or for a time range",
    input=GetTranscriptInput,
    output=TranscriptOutput,
    method="GET",
    path="/v1/sources/{source_id}/transcript",
    errors=(ErrorCode.NOT_FOUND, ErrorCode.TRANSCRIPT_MISSING),
    positional=("source_id",),
    read_only=True,
)
def get_transcript(app: App, inp: GetTranscriptInput) -> TranscriptOutput:
    """Read the transcript. Use format=words to get word indexes for edit_transcript."""
    info, words = load_transcript(app.db, inp.source_id, inp.version)
    if inp.format == "words":
        picked = [
            IndexedWord(index=i, **w.model_dump())
            for i, w in enumerate(words)
            if (inp.start is None or w.end > inp.start) and (inp.end is None or w.start < inp.end)
        ]
        return TranscriptOutput(info=info, words=picked)
    segments = _segments_in_range(build_segments(words), inp.start, inp.end)
    if inp.format == "text":
        return TranscriptOutput(info=info, text=" ".join(s.text for s in segments))
    return TranscriptOutput(info=info, segments=segments)


_PUNCT_EDGE = re.compile(r"^(\W*)(.*?)(\W*)$", re.DOTALL)


def _norm(text: str, match_case: bool) -> str:
    core = re.sub(r"[^\w']", "", text)
    return core if match_case else core.casefold()


def apply_replacement(words: list[Word], rep: Replacement) -> tuple[list[Word], int]:
    find = [_norm(t, rep.match_case) for t in rep.find.split()]
    find = [t for t in find if t]
    if not find:
        return words, 0
    repl_tokens = rep.replace.split()
    out: list[Word] = []
    changed = 0
    index = 0
    size = len(find)
    while index < len(words):
        window = words[index : index + size]
        if len(window) == size and [_norm(w.text, rep.match_case) for w in window] == find:
            lead = _PUNCT_EDGE.match(window[0].text)
            trail = _PUNCT_EDGE.match(window[-1].text)
            prefix = lead.group(1) if lead else ""
            suffix = trail.group(3) if trail else ""
            changed += size
            if repl_tokens:
                tokens = list(repl_tokens)
                tokens[0] = prefix + tokens[0]
                tokens[-1] = tokens[-1] + suffix
                if len(tokens) == size:
                    out.extend(
                        w.model_copy(update={"text": t})
                        for w, t in zip(window, tokens, strict=True)
                    )
                else:
                    span_start, span_end = window[0].start, window[-1].end
                    weights = [max(1, len(t)) for t in tokens]
                    cursor = span_start
                    for token, weight in zip(tokens, weights, strict=True):
                        width = (span_end - span_start) * weight / sum(weights)
                        out.append(
                            Word(
                                text=token,
                                start=round(cursor, 3),
                                end=round(cursor + width, 3),
                                probability=None,
                            )
                        )
                        cursor += width
            index += size
            continue
        out.append(words[index])
        index += 1
    return out, changed


@operation(
    name="edit_transcript",
    area="transcript",
    verb="edit",
    summary="Correct words or replace names; timings stay intact",
    input=EditTranscriptInput,
    output=EditTranscriptOutput,
    method="PATCH",
    path="/v1/sources/{source_id}/transcript",
    errors=(ErrorCode.NOT_FOUND, ErrorCode.TRANSCRIPT_MISSING),
    positional=("source_id",),
)
def edit_transcript(app: App, inp: EditTranscriptInput) -> EditTranscriptOutput:
    """Fix the transcript before captions are rendered. `edits` change single words by
    index (from get_transcript format=words); `replacements` fix every occurrence of a
    word or phrase, e.g. {"find": "open ai", "replace": "OpenAI"}. Each call saves a new
    transcript version; earlier versions stay readable."""
    if not inp.edits and not inp.replacements:
        raise ReelferryError(
            ErrorCode.INVALID_INPUT, "nothing to change: give edits or replacements"
        )
    info, words = load_transcript(app.db, inp.source_id)
    changed = 0
    by_index: dict[int, str] = {}
    for edit in inp.edits:
        if edit.index >= len(words):
            raise ReelferryError(
                ErrorCode.INVALID_INPUT,
                f"word index {edit.index} out of range (transcript has {len(words)} words)",
            )
        by_index[edit.index] = edit.text.strip()
    if by_index:
        edited: list[Word] = []
        for index, word in enumerate(words):
            if index in by_index:
                changed += 1
                text = by_index[index]
                if not text:
                    continue
                parts = text.split()
                if len(parts) == 1:
                    edited.append(word.model_copy(update={"text": text, "probability": None}))
                else:
                    width = (word.end - word.start) / len(parts)
                    edited.extend(
                        Word(
                            text=part,
                            start=round(word.start + i * width, 3),
                            end=round(word.start + (i + 1) * width, 3),
                        )
                        for i, part in enumerate(parts)
                    )
            else:
                edited.append(word)
        words = edited
    for rep in inp.replacements:
        words, count = apply_replacement(words, rep)
        changed += count
    if changed == 0:
        return EditTranscriptOutput(info=info, changed_words=0)
    new_info = save_transcript(
        app.db,
        source_id=inp.source_id,
        words=words,
        language=info.language,
        engine=info.engine,
        model=info.model,
        alignment=info.alignment,
        duration=info.duration,
        note=inp.note or "edited",
    )
    return EditTranscriptOutput(info=new_info, changed_words=changed)


def _write_json(path: Path, value: Any) -> str:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value), encoding="utf-8")
    tmp.replace(path)
    return str(path)


def _words_from(data: list[list[Any]]) -> list[Word]:
    return [Word(text=t, start=s, end=e, probability=p) for t, s, e, p in data]


def _words_to(words: list[Word]) -> list[list[Any]]:
    return [[w.text, w.start, w.end, w.probability] for w in words]


@job_handler(JobKind.TRANSCRIBE)
def run_transcribe(app: App, ctx: JobContext) -> dict[str, Any]:
    from reelferry.engine.audio import read_wav
    from reelferry.engine.transcribe import hf_cache_dir

    assert ctx.source_id is not None
    inp = TranscribeInput.model_validate(ctx.input)
    source = ready_source(app.db, ctx.source_id)
    settings = app.settings
    engine = inp.engine or settings.transcription_engine
    ff = app.ffmpeg()
    duration = source.duration or 0.0

    def extract() -> str:
        audio = ctx.artifact("audio.wav")

        def on_line(line: str) -> None:
            done = progress_seconds(line)
            if done is not None and duration:
                ctx.progress(done / duration, "extracting audio")

        ctx.run_process(extract_audio_cmd(ff, Path(str(source.path)), audio), on_line=on_line)
        return str(audio)

    ctx.phase(0.0, 0.05)
    audio = Path(ctx.step("extract_audio", extract, valid=lambda p: Path(p).exists()))

    def recognize() -> dict[str, Any]:
        transcriber = app.transcriber(engine)
        raw = transcriber.transcribe(
            audio,
            language=inp.language,
            vocabulary=inp.vocabulary,
            progress=ctx.progress,
            ctx=ctx,
        )
        ctx.add_cost(raw.cost_usd)
        path = _write_json(
            ctx.artifact("recognized.json"),
            {"segments": [_words_to(s) for s in raw.segments]},
        )
        return {
            "path": path,
            "language": raw.language,
            "model": raw.model,
            "engine": transcriber.engine,
            "duration": raw.duration,
            "cost_usd": raw.cost_usd,
        }

    ctx.phase(0.05, 0.8)
    recognized = ctx.step("recognize", recognize, valid=lambda r: Path(r["path"]).exists())
    segments = [
        _words_from(s)
        for s in json.loads(Path(recognized["path"]).read_text(encoding="utf-8"))["segments"]
    ]
    language = recognized["language"]

    def align() -> dict[str, Any]:
        from reelferry.engine.align import CtcAligner, make_monotonic

        flat = [w for s in segments for w in s]
        aligner = None
        if settings.forced_alignment:
            ctx.progress(0.0, "loading alignment model")
            aligner = CtcAligner.for_language(language, hf_cache_dir())
        if aligner is None:
            return {"words": _words_to(make_monotonic(flat)), "alignment": "whisper"}
        ctx.progress(0.0, "aligning words")
        result = aligner.align(read_wav(audio), segments, lambda f: ctx.progress(f, None))
        if result.aligned_words == 0:
            label = "whisper"
        elif result.aligned_words == result.total_words:
            label = "ctc"
        else:
            label = "mixed"
        return {"words": _words_to(result.words), "alignment": label}

    ctx.phase(0.8, 0.98)
    aligned = ctx.step("align", align)
    words = _words_from(aligned["words"])
    if not words:
        raise ReelferryError(ErrorCode.MEDIA_ERROR, "no speech found in this source")
    saved = ctx.step(
        "save",
        lambda: save_transcript(
            app.db,
            source_id=source.id,
            words=words,
            language=language,
            engine=recognized["engine"],
            model=recognized["model"],
            alignment=aligned["alignment"],
            duration=float(recognized["duration"]),
        ).model_dump(mode="json"),
    )
    info = TranscriptInfo.model_validate(saved)
    shutil.rmtree(app.paths.artifacts_dir / ctx.job_id, ignore_errors=True)
    return {
        "source_id": source.id,
        "version": info.version,
        "language": info.language,
        "word_count": info.word_count,
        "alignment": info.alignment,
        "engine": info.engine,
        "model": info.model,
    }
