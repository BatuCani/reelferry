"""System check and settings."""

from __future__ import annotations

import os
import shutil
from typing import Any

from pydantic import BaseModel, Field, create_model

from reelferry import __version__, credentials
from reelferry.app import App
from reelferry.config import MODEL_PRESETS, Settings, save_settings
from reelferry.engine.media import LIBASS_HINT, detect_ffmpeg
from reelferry.errors import ErrorCode, ReelferryError
from reelferry.models import Model
from reelferry.registry import operation


class SystemCheckInput(Model):
    verify_key: bool = Field(
        default=False, description="Also ask OpenRouter whether the API key works."
    )


class Check(Model):
    name: str
    ok: bool
    required: bool = Field(description="Reelferry cannot do its core job without this.")
    detail: str
    hint: str | None = None


class SystemCheckOutput(Model):
    ok: bool = Field(description="True when every required check passed.")
    version: str
    data_dir: str
    media_dir: str
    checks: list[Check]


class SettingsOutput(Model):
    settings: Settings
    model_presets: dict[str, str]
    openrouter_key: str | None = Field(
        description="Where the OpenRouter key comes from ('environment' or 'keychain'), "
        "or null. The key itself is never returned."
    )


def _settings_patch_model() -> type[BaseModel]:
    fields: dict[str, Any] = {}
    for name, info in Settings.model_fields.items():
        annotation: Any = info.annotation
        fields[name] = (annotation | None, Field(default=None, description=info.description))
    fields["reset"] = (
        list[str],
        Field(default_factory=list, description="Setting names to reset to their defaults."),
    )
    return create_model("UpdateSettingsInput", __base__=Model, **fields)


UpdateSettingsInput = _settings_patch_model()


def _check_ffmpeg(app: App) -> list[Check]:
    settings = app.settings
    try:
        ff = detect_ffmpeg(settings.ffmpeg_path, settings.ffprobe_path)
    except ReelferryError as exc:
        return [Check(name="ffmpeg", ok=False, required=True, detail=exc.message, hint=exc.hint)]
    encoder = ff.h264_encoder
    return [
        Check(name="ffmpeg", ok=True, required=True, detail=f"{ff.ffmpeg} ({ff.version})"),
        Check(name="ffprobe", ok=True, required=True, detail=ff.ffprobe),
        Check(
            name="libass",
            ok=ff.has_libass,
            required=True,
            detail="ass filter available" if ff.has_libass else "ffmpeg has no 'ass' filter",
            hint=None if ff.has_libass else LIBASS_HINT,
        ),
        Check(
            name="h264_encoder",
            ok=encoder is not None,
            required=True,
            detail=encoder or "no H.264 encoder found",
            hint=None if encoder else "Install an ffmpeg build with libx264.",
        ),
    ]


def _check_python_modules() -> list[Check]:
    checks: list[Check] = []
    try:
        from yt_dlp.version import __version__ as ytdlp_version

        checks.append(Check(name="yt_dlp", ok=True, required=False, detail=ytdlp_version))
    except Exception as exc:
        checks.append(Check(name="yt_dlp", ok=False, required=False, detail=str(exc)))
    try:
        import faster_whisper

        checks.append(
            Check(
                name="local_whisper",
                ok=True,
                required=False,
                detail=f"faster-whisper {faster_whisper.__version__}",
            )
        )
    except Exception as exc:
        checks.append(
            Check(
                name="local_whisper",
                ok=False,
                required=False,
                detail=str(exc),
                hint="Use transcription_engine=openrouter as a fallback.",
            )
        )
    try:
        import onnxruntime

        checks.append(
            Check(
                name="forced_alignment",
                ok=True,
                required=False,
                detail=f"onnxruntime {onnxruntime.__version__}",
            )
        )
    except Exception as exc:
        checks.append(Check(name="forced_alignment", ok=False, required=False, detail=str(exc)))
    return checks


def _check_key(verify: bool) -> Check:
    source = credentials.openrouter_key_source()
    hint = "Needed for suggest_clips. `reelferry config set-key` or set OPENROUTER_API_KEY."
    if source is None:
        return Check(name="openrouter_key", ok=False, required=False, detail="not set", hint=hint)
    if not verify:
        return Check(name="openrouter_key", ok=True, required=False, detail=f"set ({source})")
    import httpx

    key = credentials.openrouter_key()
    try:
        response = httpx.get(
            "https://openrouter.ai/api/v1/key",
            headers={"Authorization": f"Bearer {key}"},
            timeout=15,
        )
    except httpx.HTTPError as exc:
        return Check(
            name="openrouter_key",
            ok=False,
            required=False,
            detail=f"cannot reach OpenRouter: {exc}",
        )
    if response.status_code != 200:
        return Check(
            name="openrouter_key",
            ok=False,
            required=False,
            detail=f"rejected by OpenRouter ({response.status_code})",
            hint=hint,
        )
    return Check(name="openrouter_key", ok=True, required=False, detail=f"valid ({source})")


def _check_storage(app: App) -> list[Check]:
    checks: list[Check] = []
    for name, path in (("data_dir", app.paths.data_dir), ("media_dir", app.paths.media_dir)):
        writable = path.exists() and os.access(path, os.W_OK)
        checks.append(
            Check(
                name=name,
                ok=writable,
                required=True,
                detail=str(path) if writable else f"{path} is not writable",
            )
        )
    free = shutil.disk_usage(app.paths.media_dir).free / 1e9
    checks.append(
        Check(
            name="free_disk",
            ok=free >= 5,
            required=False,
            detail=f"{free:.1f} GB free in the media dir",
            hint=None if free >= 5 else "Models and renders need a few GB.",
        )
    )
    return checks


@operation(
    name="system_check",
    area="system",
    verb="check",
    summary="Check ffmpeg (with libass), speech engines, API key and storage",
    input=SystemCheckInput,
    output=SystemCheckOutput,
    method="GET",
    path="/v1/system/check",
    read_only=True,
)
def system_check(app: App, inp: SystemCheckInput) -> SystemCheckOutput:
    """Report what works and how to fix what does not. Never returns secret values."""
    checks = [
        *_check_ffmpeg(app),
        *_check_python_modules(),
        _check_key(inp.verify_key),
        *_check_storage(app),
    ]
    return SystemCheckOutput(
        ok=all(c.ok for c in checks if c.required),
        version=__version__,
        data_dir=str(app.paths.data_dir),
        media_dir=str(app.paths.media_dir),
        checks=checks,
    )


class NoInput(Model):
    pass


def _settings_output(settings: Settings) -> SettingsOutput:
    return SettingsOutput(
        settings=settings,
        model_presets=MODEL_PRESETS,
        openrouter_key=credentials.openrouter_key_source(),
    )


@operation(
    name="get_settings",
    area="settings",
    verb="get",
    summary="Read the settings",
    input=NoInput,
    output=SettingsOutput,
    method="GET",
    path="/v1/settings",
    read_only=True,
)
def get_settings(app: App, _inp: NoInput) -> SettingsOutput:
    """Current settings and model presets. API keys are never included."""
    return _settings_output(app.settings)


@operation(
    name="update_settings",
    area="settings",
    verb="update",
    summary="Change settings",
    input=UpdateSettingsInput,
    output=SettingsOutput,
    method="PATCH",
    path="/v1/settings",
)
def update_settings(app: App, inp: Any) -> SettingsOutput:
    """Change only the given settings. API keys are not settings: use
    `reelferry config set-key` or the OPENROUTER_API_KEY environment variable."""
    current = app.settings.model_dump()
    patch = inp.model_dump(exclude_none=True)
    reset = patch.pop("reset", [])
    unknown = [name for name in reset if name not in Settings.model_fields]
    if unknown:
        raise ReelferryError(ErrorCode.INVALID_INPUT, f"unknown settings: {', '.join(unknown)}")
    for name in reset:
        current.pop(name, None)
    current.update(patch)
    try:
        settings = Settings.model_validate(current)
    except ValueError as exc:
        raise ReelferryError(ErrorCode.INVALID_INPUT, str(exc)) from exc
    if "default_caption_style" in patch:
        from reelferry.captions.styles import get_style

        get_style(settings.default_caption_style)
    save_settings(app.paths, settings)
    return _settings_output(settings)
