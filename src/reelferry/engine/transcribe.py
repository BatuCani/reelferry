"""Speech to text with word timings: local Whisper, or OpenRouter as a fallback."""

from __future__ import annotations

import gc
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from reelferry.engine.audio import SAMPLE_RATE, read_wav, require_free_disk
from reelferry.errors import ErrorCode, ReelferryError
from reelferry.models import Word

if TYPE_CHECKING:
    from reelferry.app import App
    from reelferry.jobs import JobContext

ProgressFn = Callable[[float, str | None], None]
WHISPER_DOWNLOAD_BYTES = 1_700_000_000
CLOUD_CHUNK_SECONDS = 600.0
CLOUD_OVERLAP_SECONDS = 1.0


@dataclass
class RawTranscript:
    segments: list[list[Word]]
    language: str | None
    model: str
    duration: float
    cost_usd: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


class Transcriber(Protocol):
    engine: str

    def transcribe(
        self,
        audio: Path,
        *,
        language: str | None,
        vocabulary: list[str],
        progress: ProgressFn,
        ctx: JobContext | None,
    ) -> RawTranscript: ...


def _clean_words(raw: list[tuple[str, float, float, float | None]]) -> list[Word]:
    words: list[Word] = []
    for raw_text, start, end, prob in raw:
        text = raw_text.strip()
        if not text:
            continue
        words.append(
            Word(
                text=text,
                start=round(float(start), 3),
                end=round(max(float(end), float(start)), 3),
                probability=None if prob is None else round(float(prob), 3),
            )
        )
    return words


def hf_cache_dir() -> Path:
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE)


class LocalWhisper:
    """faster-whisper (CTranslate2). Models download once into the Hugging Face cache."""

    engine = "local"

    def __init__(self, model: str, device: str, compute_type: str) -> None:
        self.model_name = model
        self.device = device
        self.compute_type = compute_type

    def _resolve_device(self) -> tuple[str, str]:
        device = self.device
        if device == "auto":
            try:
                import ctranslate2

                device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
            except Exception:
                device = "cpu"
        compute = self.compute_type
        if compute == "auto":
            compute = "float16" if device == "cuda" else "int8"
        return device, compute

    def transcribe(
        self,
        audio: Path,
        *,
        language: str | None,
        vocabulary: list[str],
        progress: ProgressFn,
        ctx: JobContext | None,
    ) -> RawTranscript:
        from faster_whisper import WhisperModel

        device, compute = self._resolve_device()
        message = f"loading Whisper {self.model_name} on {device}"
        if not Path(self.model_name).exists():
            cache = hf_cache_dir()
            if not _model_cached(self.model_name, cache):
                require_free_disk(cache, WHISPER_DOWNLOAD_BYTES + 500_000_000, "the Whisper model")
                message = f"downloading Whisper {self.model_name} (one time, about 1.6 GB)"
        progress(0.0, message)
        model = WhisperModel(
            self.model_name,
            device=device,
            compute_type=compute,
            cpu_threads=max(1, (os.cpu_count() or 4) - 1),
        )
        samples = read_wav(audio)
        duration = len(samples) / SAMPLE_RATE
        try:
            segments_iter, info = model.transcribe(
                samples,
                language=language,
                word_timestamps=True,
                vad_filter=True,
                beam_size=5,
                condition_on_previous_text=False,
                hotwords=", ".join(vocabulary) if vocabulary else None,
            )
            progress(0.0, "transcribing")
            segments: list[list[Word]] = []
            for segment in segments_iter:
                raw = [(w.word, w.start, w.end, w.probability) for w in segment.words or []]
                words = _clean_words(raw)
                if words:
                    segments.append(words)
                progress(min(1.0, segment.end / max(duration, 0.001)), None)
                if ctx is not None:
                    ctx.check_cancel()
            return RawTranscript(
                segments=segments,
                language=info.language,
                model=self.model_name,
                duration=duration,
            )
        finally:
            del model
            gc.collect()


def _model_cached(name: str, cache: Path) -> bool:
    from faster_whisper.utils import _MODELS

    repo = _MODELS.get(name, name)
    return bool((cache / ("models--" + str(repo).replace("/", "--"))).exists())


class OpenRouterTranscriber:
    """Cloud fallback. Audio is cut into 10-minute pieces, sent, and stitched back."""

    engine = "openrouter"

    def __init__(self, app: App) -> None:
        self.app = app

    def transcribe(
        self,
        audio: Path,
        *,
        language: str | None,
        vocabulary: list[str],  # noqa: ARG002 - the cloud endpoint takes no vocabulary hint
        progress: ProgressFn,
        ctx: JobContext | None,
    ) -> RawTranscript:
        import subprocess

        from reelferry.engine.llm import OpenRouterClient

        model = self.app.settings.openrouter_transcription_model
        client = OpenRouterClient(self.app.openrouter_key())
        ff = self.app.ffmpeg()
        samples = read_wav(audio)
        duration = len(samples) / SAMPLE_RATE
        del samples
        bounds: list[float] = []
        cursor = 0.0
        while cursor < duration:
            bounds.append(cursor)
            cursor += CLOUD_CHUNK_SECONDS
        segments: list[list[Word]] = []
        detected: str | None = language
        cost = 0.0
        for number, start in enumerate(bounds):
            end = min(duration, start + CLOUD_CHUNK_SECONDS)
            piece = audio.with_name(f"{audio.stem}.part{number}.mp3")
            subprocess.run(
                [
                    ff.ffmpeg,
                    "-hide_banner",
                    "-nostdin",
                    "-y",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{start:.3f}",
                    "-t",
                    f"{end - start + CLOUD_OVERLAP_SECONDS:.3f}",
                    "-i",
                    str(audio),
                    "-ac",
                    "1",
                    "-b:a",
                    "48k",
                    str(piece),
                ],
                check=True,
            )
            try:
                data, piece_cost = client.transcribe(
                    model=model, audio=piece, audio_format="mp3", language=language
                )
            finally:
                piece.unlink(missing_ok=True)
            cost += piece_cost
            detected = detected or _language_code(data.get("language"))
            words = _cloud_words(data, offset=start)
            # Keep only words that begin inside this piece's own window (drop the overlap).
            words = [w for w in words if start <= w.start < end or number == len(bounds) - 1]
            if words:
                segments.append(words)
            progress((number + 1) / len(bounds), None)
            if ctx is not None:
                ctx.check_cancel()
        return RawTranscript(
            segments=_split_segments([w for s in segments for w in s]),
            language=detected,
            model=model,
            duration=duration,
            cost_usd=cost,
        )


_LANGUAGE_NAMES = {
    "english": "en",
    "german": "de",
    "french": "fr",
    "spanish": "es",
    "dutch": "nl",
    "polish": "pl",
    "italian": "it",
    "portuguese": "pt",
}


def _language_code(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    value = value.strip().lower()
    return _LANGUAGE_NAMES.get(value, value if len(value) <= 3 else None)


def _cloud_words(data: dict[str, Any], *, offset: float) -> list[Word]:
    raw: list[tuple[str, float, float, float | None]] = []
    for item in data.get("words") or []:
        text = item.get("word", item.get("text", ""))
        raw.append((str(text), float(item["start"]) + offset, float(item["end"]) + offset, None))
    if raw:
        return _clean_words(raw)
    # No word timings: spread each segment's words evenly over its time span.
    for segment in data.get("segments") or []:
        tokens = str(segment.get("text", "")).split()
        if not tokens:
            continue
        seg_start = float(segment["start"]) + offset
        width = (float(segment["end"]) + offset - seg_start) / len(tokens)
        for index, token in enumerate(tokens):
            raw.append((token, seg_start + index * width, seg_start + (index + 1) * width, None))
    if not raw and data.get("text"):
        raise ReelferryError(
            ErrorCode.UPSTREAM_ERROR,
            "the transcription model returned text without timestamps",
            hint="Pick a model that supports verbose_json timestamps, or use the local engine.",
        )
    return _clean_words(raw)


def _split_segments(
    words: list[Word], max_gap: float = 1.0, max_len: float = 20.0
) -> list[list[Word]]:
    segments: list[list[Word]] = []
    current: list[Word] = []
    for word in words:
        if current and (
            word.start - current[-1].end > max_gap
            or word.end - current[0].start > max_len
            or current[-1].text.endswith((".", "?", "!"))
        ):
            segments.append(current)
            current = []
        current.append(word)
    if current:
        segments.append(current)
    return segments


def default_transcriber(app: App, engine: str) -> Transcriber:
    settings = app.settings
    if engine == "openrouter":
        return OpenRouterTranscriber(app)
    if engine == "local":
        return LocalWhisper(
            settings.whisper_model, settings.whisper_device, settings.whisper_compute_type
        )
    raise ReelferryError(ErrorCode.INVALID_INPUT, f"unknown transcription engine {engine!r}")
