"""Small audio helpers shared by the speech engines."""

from __future__ import annotations

import shutil
import wave
from pathlib import Path

import numpy as np
import numpy.typing as npt

from reelferry.errors import ErrorCode, ReelferryError

SAMPLE_RATE = 16000

FloatArray = npt.NDArray[np.float32]


def read_wav(path: Path) -> FloatArray:
    """Read a 16 kHz mono 16-bit WAV (as written by ``extract_audio_cmd``) as float32."""
    with wave.open(str(path), "rb") as handle:
        if handle.getframerate() != SAMPLE_RATE or handle.getnchannels() != 1:
            raise ReelferryError(ErrorCode.MEDIA_ERROR, f"{path.name} is not 16 kHz mono audio")
        frames = handle.readframes(handle.getnframes())
    return np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0


def require_free_disk(path: Path, needed_bytes: int, what: str) -> None:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    if free < needed_bytes:
        raise ReelferryError(
            ErrorCode.MISSING_DEPENDENCY,
            f"not enough free disk space for {what}: need about "
            f"{needed_bytes / 1e9:.1f} GB, {free / 1e9:.1f} GB free",
        )
