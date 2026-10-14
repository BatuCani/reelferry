"""Where Reelferry keeps its state, and the user's settings.

State (database, settings, local API token, caches) lives in the user data dir.
Media (downloaded sources, rendered clips) lives in a user media dir.
Both can be moved with ``REELFERRY_DATA_DIR`` and ``REELFERRY_MEDIA_DIR``.
"""

from __future__ import annotations

import json
import os
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import platformdirs
from pydantic import BaseModel, ConfigDict, Field

APP_NAME = "reelferry"
SECRET_SERVICE = "reelferry"
OPENROUTER_KEY_NAME = "openrouter_api_key"
OPENROUTER_ENV = "OPENROUTER_API_KEY"

MODEL_PRESETS: dict[str, str] = {
    "quality": "anthropic/claude-opus-5.5",
    "budget": "google/gemini-3.8-flash",
}


@dataclass(frozen=True)
class Paths:
    data_dir: Path
    media_dir: Path

    @classmethod
    def default(cls) -> Paths:
        data = os.environ.get("REELFERRY_DATA_DIR")
        media = os.environ.get("REELFERRY_MEDIA_DIR")
        return cls(
            data_dir=Path(data).expanduser()
            if data
            else Path(platformdirs.user_data_dir(APP_NAME)),
            media_dir=(
                Path(media).expanduser()
                if media
                else Path(platformdirs.user_videos_dir()) / "Reelferry"
            ),
        )

    @property
    def db_path(self) -> Path:
        return self.data_dir / "reelferry.db"

    @property
    def settings_path(self) -> Path:
        return self.data_dir / "settings.json"

    @property
    def token_path(self) -> Path:
        return self.data_dir / "api-token"

    @property
    def artifacts_dir(self) -> Path:
        return self.data_dir / "artifacts"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / "logs"

    @property
    def sources_dir(self) -> Path:
        return self.media_dir / "sources"

    @property
    def clips_dir(self) -> Path:
        return self.media_dir / "clips"

    def ensure(self) -> None:
        for path in (self.data_dir, self.artifacts_dir, self.logs_dir, self.media_dir):
            path.mkdir(parents=True, exist_ok=True)


class Settings(BaseModel):
    """User settings. Secrets never live here; see ``secrets.py``."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    suggest_model: str = Field(
        default="quality",
        description=(
            "OpenRouter model for clip suggestions: a preset ('quality' = "
            f"{MODEL_PRESETS['quality']}, 'budget' = {MODEL_PRESETS['budget']}) "
            "or any OpenRouter model id."
        ),
    )
    transcription_engine: Literal["local", "openrouter"] = Field(
        default="local",
        description=(
            "'local' runs Whisper on this machine for free; 'openrouter' sends audio "
            "to OpenRouter (needs the key, about one cent per hour of audio)."
        ),
    )
    whisper_model: str = Field(
        default="large-v3-turbo",
        description="Local Whisper model name or path (faster-whisper format).",
    )
    whisper_device: Literal["auto", "cpu", "cuda"] = "auto"
    whisper_compute_type: str = Field(
        default="auto", description="CTranslate2 compute type, e.g. 'int8' or 'float16'."
    )
    forced_alignment: bool = Field(
        default=True,
        description="Refine word timings with a CTC forced aligner where a model exists.",
    )
    openrouter_transcription_model: str = "openai/whisper-large-v3-turbo"
    ffmpeg_path: str | None = Field(
        default=None, description="ffmpeg binary to use. Default: the one on PATH."
    )
    ffprobe_path: str | None = Field(
        default=None, description="ffprobe binary to use. Default: next to ffmpeg, else PATH."
    )
    fonts_dir: str | None = Field(
        default=None, description="Extra font directory for burned-in captions."
    )
    default_caption_style: str = "pop"
    default_format: Literal["9:16", "16:9"] = "9:16"
    max_parallel_jobs: int = Field(default=1, ge=1, le=8)


def resolve_model(name: str) -> str:
    return MODEL_PRESETS.get(name, name)


def load_settings(paths: Paths) -> Settings:
    if not paths.settings_path.exists():
        return Settings()
    raw = json.loads(paths.settings_path.read_text(encoding="utf-8"))
    return Settings.model_validate(raw)


def save_settings(paths: Paths, settings: Settings) -> None:
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    tmp = paths.settings_path.with_suffix(".tmp")
    tmp.write_text(settings.model_dump_json(indent=2, exclude_defaults=True), encoding="utf-8")
    tmp.replace(paths.settings_path)


def load_or_create_token(paths: Paths) -> str:
    """The bearer token that guards the local API and MCP over HTTP."""
    env = os.environ.get("REELFERRY_API_TOKEN")
    if env:
        return env
    if paths.token_path.exists():
        return paths.token_path.read_text(encoding="utf-8").strip()
    paths.data_dir.mkdir(parents=True, exist_ok=True)
    token = "rf_" + secrets.token_urlsafe(32)
    fd = os.open(paths.token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(token + "\n")
    return token
