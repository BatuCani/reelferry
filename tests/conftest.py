from __future__ import annotations

import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from reelferry.app import App
from reelferry.config import Paths
from reelferry.engine.llm import ChatResult
from reelferry.engine.transcribe import ProgressFn, RawTranscript
from reelferry.jobs import JobContext, Worker
from reelferry.models import Job, Word

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

SCRIPT = (
    "Welcome back to the show. Today I want to share three habits that changed my life. "
    "The first habit is waking up at six. The second habit is reading twenty pages a day. "
    "The third habit surprised everyone: I say no to most meetings. "
    "That one saved me ten hours every week."
)


def script_words(start: float = 0.5, word_seconds: float = 0.36) -> list[Word]:
    words: list[Word] = []
    cursor = start
    for token in SCRIPT.split():
        words.append(
            Word(text=token, start=round(cursor, 3), end=round(cursor + word_seconds - 0.05, 3))
        )
        cursor += word_seconds
    return words


class FakeTranscriber:
    engine = "local"

    def __init__(self, words: list[Word]) -> None:
        self.words = words
        self.calls = 0

    def transcribe(
        self,
        audio: Path,
        *,
        language: str | None,
        vocabulary: list[str],
        progress: ProgressFn,
        ctx: JobContext | None,
    ) -> RawTranscript:
        assert audio.exists()
        self.calls += 1
        progress(1.0, "done")
        return RawTranscript(
            segments=[self.words[:20], self.words[20:]],
            language=language or "en",
            model="fake-whisper",
            duration=self.words[-1].end + 1,
        )


class FakeChat:
    def __init__(self, clips: list[dict[str, Any]]) -> None:
        self.clips = clips
        self.prompts: list[str] = []

    def json_chat(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema_name: str,
        schema: dict[str, Any],
    ) -> ChatResult:
        self.prompts.append(user)
        return ChatResult(data={"clips": self.clips}, model=model, cost_usd=0.0021)


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("REELFERRY_NO_AUTOWORKER", "1")
    monkeypatch.setenv("REELFERRY_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("REELFERRY_MEDIA_DIR", str(tmp_path / "media"))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("REELFERRY_API_TOKEN", raising=False)
    monkeypatch.setattr("reelferry.credentials._keyring_get", lambda name: None)


@pytest.fixture
def app(tmp_path: Path) -> Iterator[App]:
    instance = App(Paths(data_dir=tmp_path / "data", media_dir=tmp_path / "media"))
    instance.transcriber_factory = lambda app, engine: FakeTranscriber(script_words())
    yield instance
    instance.db.close()


def run_jobs(app: App) -> None:
    """Drain the queue in the foreground."""
    Worker(app, exit_when_idle=True, idle_grace=0.0).run_until_idle()


def finished(app: App, job: Job) -> Job:
    run_jobs(app)
    return app.jobs.get(job.id)


@pytest.fixture(scope="session")
def talk_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A synthetic 1080p video with a tone track, made with ffmpeg."""
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg not installed")
    path = tmp_path_factory.mktemp("media") / "talk.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=1280x720:rate=30:duration=20",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000:duration=20",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
    )
    return path
