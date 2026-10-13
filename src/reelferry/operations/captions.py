"""Caption styles and caption export."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from reelferry.app import App
from reelferry.captions.build import build_pages, render_caption
from reelferry.captions.styles import STYLES, get_style, style_info
from reelferry.engine.render import OUTPUT_SIZES
from reelferry.errors import ErrorCode, ReelferryError
from reelferry.models import CaptionStyleInfo, Model
from reelferry.registry import operation
from reelferry.store import get_clip, get_source, load_transcript


class NoInput(Model):
    pass


class StyleList(Model):
    styles: list[CaptionStyleInfo]
    default: str


class ExportCaptionsInput(Model):
    clip_id: str | None = Field(default=None, description="Captions for a clip (clip time).")
    source_id: str | None = Field(
        default=None, description="Or captions for a whole source (source time)."
    )
    format: Literal["srt", "vtt", "ass", "json"] = "srt"
    style: str | None = Field(
        default=None, description="Caption style (affects grouping and ASS look)."
    )
    output_path: str | None = Field(default=None, description="Where to write. Default: media dir.")
    include_content: bool = Field(default=False, description="Also return the file content.")


class ExportCaptionsOutput(Model):
    path: str
    format: str
    cues: int
    content: str | None = None


@operation(
    name="list_caption_styles",
    area="caption",
    verb="styles",
    summary="List the caption styles",
    input=NoInput,
    output=StyleList,
    method="GET",
    path="/v1/caption-styles",
    read_only=True,
)
def list_caption_styles(app: App, _inp: NoInput) -> StyleList:
    """Caption styles to pass as caption_style to create_clip, update_clip or render_clip."""
    return StyleList(
        styles=[style_info(s) for s in STYLES.values()],
        default=app.settings.default_caption_style,
    )


@operation(
    name="export_captions",
    area="caption",
    verb="export",
    summary="Export captions as SRT, VTT, ASS or JSON",
    input=ExportCaptionsInput,
    output=ExportCaptionsOutput,
    method="POST",
    path="/v1/captions/export",
    errors=(ErrorCode.NOT_FOUND, ErrorCode.TRANSCRIPT_MISSING),
)
def export_captions(app: App, inp: ExportCaptionsInput) -> ExportCaptionsOutput:
    """Write captions from the current transcript, without rendering video. VTT carries
    per-word timestamps; JSON carries every word with timing and emphasis."""
    if bool(inp.clip_id) == bool(inp.source_id):
        raise ReelferryError(ErrorCode.INVALID_INPUT, "give exactly one of clip_id or source_id")
    if inp.clip_id:
        clip = get_clip(app.db, inp.clip_id)
        source_id, start, end = clip.source_id, clip.start, clip.end
        style = get_style(inp.style or clip.caption_style)
        emphasis, fmt_size = clip.emphasis, OUTPUT_SIZES[clip.format]
        default_dir, stem = app.paths.clips_dir / clip.id, f"captions-r{clip.revision}"
    else:
        assert inp.source_id is not None
        source = get_source(app.db, inp.source_id)
        info, _ = load_transcript(app.db, source.id)
        source_id, start, end = source.id, 0.0, info.duration + 1.0
        style = get_style(inp.style or "minimal")
        emphasis = []
        fmt_size = (source.width or 1920, source.height or 1080)
        default_dir, stem = app.paths.sources_dir / source.id, f"captions-v{info.version}"
    _, words = load_transcript(app.db, source_id)
    pages, picked = build_pages(words, start=start, end=end, style=style, emphasis=emphasis)
    content = render_caption(inp.format, pages, picked, style, fmt_size)
    target = (
        Path(inp.output_path).expanduser()
        if inp.output_path
        else default_dir / f"{stem}.{inp.format}"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return ExportCaptionsOutput(
        path=str(target.resolve()),
        format=inp.format,
        cues=len(pages),
        content=content if inp.include_content else None,
    )
