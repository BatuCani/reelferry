"""Build caption files for a clip from the latest transcript."""

from __future__ import annotations

from pathlib import Path

from reelferry.captions.formats import to_ass, to_json, to_srt, to_vtt
from reelferry.captions.layout import CaptionWord, Page, clip_words, paginate
from reelferry.captions.styles import CaptionStyle
from reelferry.models import Word


def build_pages(
    words: list[Word],
    *,
    start: float,
    end: float,
    style: CaptionStyle,
    emphasis: list[str],
) -> tuple[list[Page], list[CaptionWord]]:
    picked = clip_words(words, start, end, emphasis)
    pages = paginate(
        picked,
        words_per_page=style.words_per_page,
        max_chars_per_line=style.max_chars_per_line,
        lines=style.lines,
        clip_end=end - start,
    )
    return pages, picked


def render_caption(
    fmt: str,
    pages: list[Page],
    words: list[CaptionWord],
    style: CaptionStyle,
    size: tuple[int, int],
) -> str:
    if fmt == "srt":
        return to_srt(pages)
    if fmt == "vtt":
        return to_vtt(pages)
    if fmt == "json":
        return to_json(pages, words)
    return to_ass(pages, style, size[0], size[1])


def write_captions(
    directory: Path,
    stem: str,
    *,
    pages: list[Page],
    words: list[CaptionWord],
    style: CaptionStyle,
    size: tuple[int, int],
) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for fmt in ("ass", "srt", "vtt", "json"):
        path = directory / f"{stem}.{fmt}"
        path.write_text(render_caption(fmt, pages, words, style, size), encoding="utf-8")
        written[fmt] = str(path)
    return written
