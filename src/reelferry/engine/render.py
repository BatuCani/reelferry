"""FFmpeg command builders for clip renders and preview contact sheets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from reelferry.engine.media import FFmpeg

OUTPUT_SIZES: dict[str, tuple[int, int]] = {"9:16": (1080, 1920), "16:9": (1920, 1080)}


@dataclass(frozen=True)
class RenderPlan:
    source: Path
    start: float
    end: float
    format: str
    framing: str
    crop_x: float
    source_has_audio: bool
    ass_file: str | None  # relative to the working directory, to avoid filter escaping
    soft_subtitles: str | None  # SRT file to mux as a subtitle track
    fonts_dir: str | None
    output: Path

    @property
    def size(self) -> tuple[int, int]:
        return OUTPUT_SIZES[self.format]

    @property
    def duration(self) -> float:
        return self.end - self.start


def _escape_filter_value(value: str) -> str:
    return value.replace("\\", "/").replace(":", "\\:").replace("'", "\\'").replace(",", "\\,")


def framing_filter(fmt: str, framing: str, crop_x: float, *, source: str = "[0:v]") -> str:
    """Frame ``source`` at the output size of ``fmt``; the chain ends in ``[base]``."""
    width, height = OUTPUT_SIZES[fmt]
    cx = min(1.0, max(0.0, crop_x))
    if framing == "crop":
        return (
            f"{source}scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height}:(iw-{width})*{cx:.4f}:(ih-{height})/2,setsar=1[base]"
        )
    small_w, small_h = width // 4, height // 4
    return (
        f"{source}split=2[bgsrc][fgsrc];"
        f"[bgsrc]scale={small_w}:{small_h}:force_original_aspect_ratio=increase,"
        f"crop={small_w}:{small_h},gblur=sigma=12,eq=brightness=-0.08,"
        f"scale={width}:{height}[bg];"
        f"[fgsrc]scale={width}:{height}:force_original_aspect_ratio=decrease[fg];"
        "[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1[base]"
    )


def video_filter(plan: RenderPlan) -> str:
    chain = framing_filter(plan.format, plan.framing, plan.crop_x)
    tail = "[base]"
    if plan.ass_file:
        ass = f"ass=filename={_escape_filter_value(plan.ass_file)}"
        if plan.fonts_dir:
            ass += f":fontsdir={_escape_filter_value(plan.fonts_dir)}"
        chain += f";{tail}{ass}[subbed]"
        tail = "[subbed]"
    chain += f";{tail}format=yuv420p[vout]"
    return chain


def render_cmd(ff: FFmpeg, plan: RenderPlan) -> list[str]:
    cmd = [
        ff.ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-y",
        "-ss",
        f"{plan.start:.3f}",
        "-t",
        f"{plan.duration:.3f}",
        "-i",
        str(plan.source),
    ]
    if not plan.source_has_audio:
        cmd += ["-f", "lavfi", "-t", f"{plan.duration:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
    if plan.soft_subtitles:
        cmd += ["-i", plan.soft_subtitles]
    cmd += ["-filter_complex", video_filter(plan), "-map", "[vout]"]
    cmd += ["-map", "0:a:0" if plan.source_has_audio else "1:a:0"]
    if plan.soft_subtitles:
        sub_index = 1 if plan.source_has_audio else 2
        cmd += ["-map", f"{sub_index}:s:0", "-c:s", "mov_text", "-metadata:s:s:0", "language=und"]
    encoder = ff.h264_encoder or "libx264"
    if encoder == "libx264":
        cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-profile:v", "high"]
    else:
        cmd += ["-c:v", encoder, "-b:v", "10M"]
    cmd += [
        "-r",
        "30",
        "-c:a",
        "aac",
        "-b:a",
        "160k",
        "-ar",
        "48000",
        "-movflags",
        "+faststart",
        "-shortest",
        "-progress",
        "pipe:1",
        "-nostats",
        str(plan.output),
    ]
    return cmd


def contact_sheet_cmd(
    ff: FFmpeg,
    *,
    media: Path,
    start: float,
    duration: float,
    fmt: str,
    framing: str,
    crop_x: float,
    frames: int,
    columns: int,
    output: Path,
) -> list[str]:
    """A grid of frames from ``media`` framed exactly like a render of the clip."""
    rows = -(-frames // columns)
    rate = frames / max(duration, 0.1)
    width, height = OUTPUT_SIZES[fmt]
    tile_width = 240 if height > width else 426
    graph = (
        f"[0:v]fps={rate:.6f}[sampled];"
        + framing_filter(fmt, framing, crop_x, source="[sampled]")
        + f";[base]scale={tile_width}:-2,tile={columns}x{rows}:padding=6:margin=6[sheet]"
    )
    return [
        ff.ffmpeg,
        "-hide_banner",
        "-nostdin",
        "-y",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(media),
        "-filter_complex",
        graph,
        "-map",
        "[sheet]",
        "-frames:v",
        "1",
        "-q:v",
        "4",
        str(output),
    ]
