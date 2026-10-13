"""The application object every operation receives: paths, settings, storage, jobs."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from reelferry import credentials
from reelferry.config import Paths, Settings, load_settings
from reelferry.db import Database
from reelferry.engine.media import FFmpeg, detect_ffmpeg
from reelferry.errors import ErrorCode, ReelferryError
from reelferry.jobs import JobStore

if TYPE_CHECKING:
    from reelferry.engine.llm import ChatClient
    from reelferry.engine.transcribe import Transcriber


class App:
    def __init__(self, paths: Paths | None = None) -> None:
        self.paths = paths or Paths.default()
        self.paths.ensure()
        self.db = Database(self.paths.db_path)
        self.jobs = JobStore(self.db)
        # Seams for tests and future engines; None means "use the built-in default".
        self.transcriber_factory: Callable[[App, str], Transcriber] | None = None
        self.chat_client_factory: Callable[[App], ChatClient] | None = None

    @property
    def settings(self) -> Settings:
        return load_settings(self.paths)

    def ffmpeg(self) -> FFmpeg:
        settings = self.settings
        return detect_ffmpeg(settings.ffmpeg_path, settings.ffprobe_path)

    def openrouter_key(self) -> str:
        key = credentials.openrouter_key()
        if not key:
            raise ReelferryError(
                ErrorCode.MISSING_API_KEY,
                "no OpenRouter API key configured",
                hint=(
                    "Set OPENROUTER_API_KEY, or store it in the system keychain with "
                    "`reelferry config set-key`. Get a key at https://openrouter.ai/keys."
                ),
            )
        return key

    def transcriber(self, engine: str) -> Transcriber:
        if self.transcriber_factory is not None:
            return self.transcriber_factory(self, engine)
        from reelferry.engine.transcribe import default_transcriber

        return default_transcriber(self, engine)

    def chat_client(self) -> ChatClient:
        if self.chat_client_factory is not None:
            return self.chat_client_factory(self)
        from reelferry.engine.llm import OpenRouterClient

        return OpenRouterClient(self.openrouter_key())
