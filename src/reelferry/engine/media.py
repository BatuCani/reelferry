"""FFmpeg discovery, capability checks and probing."""

from __future__ import annotations

import functools
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reelferry.errors import ErrorCode, ReelferryError

LIBASS_HINT = (
    "Burned-in captions need an ffmpeg built with libass (the 'ass' filter). "
    "macOS: `brew install ffmpeg-full` and set ffmpeg_path to its binary, or use any "
    "static build that lists libass. Debian/Ubuntu: `apt install ffmpeg`. "
    "Windows: a 'full' build from gyan.dev or BtbN. "
    "Then point Reelferry at it: `reelferry settings update --ffmpeg-path /path/to/ffmpeg`. "
    "Without libass you can still render with burn_captions=false (soft subtitles + sidecars)."
)


def own_process_group() -> dict[str, Any]:
    """Popen options that keep a child out of the terminal's process group, so Ctrl-C
    reaches only the parent, which then stops its children itself."""
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


@dataclass(frozen=True)
class FFmpeg:
    ffmpeg: str
    ffprobe: str
    version: str
    filters: frozenset[str]
    encoders: frozenset[str]

    @property
    def has_libass(self) -> bool:
        return "ass" in self.filters

    @property
    def h264_encoder(self) -> str | None:
        for name in ("libx264", "h264_videotoolbox", "h264_nvenc", "h264_qsv", "h264_mf"):
            if name in self.encoders:
                return name
        return None

    def require_libass(self) -> None:
        if not self.has_libass:
            raise ReelferryError(
                ErrorCode.MISSING_DEPENDENCY,
                f"ffmpeg at {self.ffmpeg} has no libass ('ass' filter)",
                hint=LIBASS_HINT,
            )


def _which(name: str, explicit: str | None, near: str | None = None) -> str | None:
    if explicit:
        path = Path(explicit).expanduser()
        return str(path) if path.exists() else None
    if near:
        sibling = Path(near).with_name(name + Path(near).suffix)
        if sibling.exists():
            return str(sibling)
    return shutil.which(name)


def _list_names(binary: str, flag: str) -> frozenset[str]:
    out = subprocess.run(
        [binary, "-hide_banner", flag], capture_output=True, text=True, check=False
    ).stdout
    names: set[str] = set()
    for line in out.splitlines():
        match = re.match(r"^\s*[A-Z.|]{2,8}\s+(\S+)\s", line)
        if match and match.group(1) != "=":
            names.add(match.group(1))
    return frozenset(names)


@functools.lru_cache(maxsize=8)
def _detect(ffmpeg_path: str | None, ffprobe_path: str | None) -> FFmpeg:
    ffmpeg = _which("ffmpeg", ffmpeg_path)
    if ffmpeg is None:
        raise ReelferryError(
            ErrorCode.MISSING_DEPENDENCY,
            "ffmpeg not found" + (f" at {ffmpeg_path}" if ffmpeg_path else " on PATH"),
            hint="Install ffmpeg (with libass for burned-in captions). " + LIBASS_HINT,
        )
    ffprobe = _which("ffprobe", ffprobe_path, near=ffmpeg if not ffprobe_path else None)
    if ffprobe is None:
        raise ReelferryError(
            ErrorCode.MISSING_DEPENDENCY,
            "ffprobe not found",
            hint="ffprobe ships with ffmpeg; set ffprobe_path if it lives elsewhere.",
        )
    head = subprocess.run(
        [ffmpeg, "-hide_banner", "-version"], capture_output=True, text=True, check=False
    ).stdout
    version = head.splitlines()[0].removeprefix("ffmpeg version ").split(" ")[0] if head else "?"
    return FFmpeg(
        ffmpeg=ffmpeg,
        ffprobe=ffprobe,
        version=version,
        filters=_list_names(ffmpeg, "-filters"),
        encoders=_list_names(ffmpeg, "-encoders"),
    )


def detect_ffmpeg(ffmpeg_path: str | None = None, ffprobe_path: str | None = None) -> FFmpeg:
    return _detect(ffmpeg_path, ffprobe_path)


def _fraction(value: str | None) -> float | None:
    if not value or value in ("0/0", "N/A"):
        return None
    if "/" in value:
        num, den = value.split("/", 1)
        return float(num) / float(den) if float(den) else None
    return float(value)


def probe(ff: FFmpeg, path: Path) -> dict[str, Any]:
    """Duration, size, frame rate and audio presence of a media file."""
    proc = subprocess.run(
        [
            ff.ffprobe,
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise ReelferryError(
            ErrorCode.MEDIA_ERROR,
            f"cannot read media file {path.name}",
            details={"stderr": proc.stderr[-1000:]},
        )
    data = json.loads(proc.stdout or "{}")
    streams = data.get("streams", [])
    video = next(
        (
            s
            for s in streams
            if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")
        ),
        None,
    )
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    duration = data.get("format", {}).get("duration")
    if video is None and audio is None:
        raise ReelferryError(ErrorCode.MEDIA_ERROR, f"{path.name} has no audio or video stream")
    width = height = None
    if video is not None:
        width, height = video.get("width"), video.get("height")
        rotation = _rotation(video)
        if rotation in (90, 270) and width and height:
            width, height = height, width
    return {
        "duration": float(duration) if duration else None,
        "width": width,
        "height": height,
        "fps": _fraction(video.get("avg_frame_rate")) if video else None,
        "has_audio": audio is not None,
        "has_video": video is not None,
    }


def _rotation(stream: dict[str, Any]) -> int:
    rotate = stream.get("tags", {}).get("rotate")
    if rotate:
        return abs(int(rotate)) % 360
    for side in stream.get("side_data_list", []):
        if "rotation" in side:
            return abs(int(side["rotation"])) % 360
    return 0


def extract_audio_cmd(ff: FFmpeg, source: Path, target: Path) -> list[str]:
    """16 kHz mono 16-bit WAV, the format every speech model here expects."""
    return [
        ff.ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        "-progress",
        "pipe:1",
        "-nostats",
        str(target),
    ]


def progress_seconds(line: str) -> float | None:
    """Parse an ``-progress pipe:1`` line into seconds done, if it carries time."""
    if line.startswith("out_time_us=") or line.startswith("out_time_ms="):
        value = line.split("=", 1)[1]
        if value.isdigit():
            return int(value) / 1_000_000
    return None
