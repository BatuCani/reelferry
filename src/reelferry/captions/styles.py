"""Caption styles. Few, distinct and clean, rather than many look-alikes.

Sizes are fractions of the output height so a style looks the same in 9:16 and 16:9.
Colors are RGB hex strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from reelferry.errors import ErrorCode, ReelferryError
from reelferry.models import CaptionStyleInfo

Animation = Literal["pop", "karaoke", "box", "none", "glow"]


@dataclass(frozen=True)
class CaptionStyle:
    id: str
    name: str
    description: str
    font: str
    bold: bool
    size: float  # font size as a fraction of output height (portrait reference)
    uppercase: bool
    text_color: str
    active_color: str
    emphasis_color: str
    outline_color: str
    outline: float  # fraction of font size
    shadow: float  # fraction of font size
    animation: Animation
    position: Literal["bottom", "middle", "lower_third"]
    words_per_page: int
    max_chars_per_line: int
    lines: int
    box_color: str = "000000"
    letter_spacing: float = 0.0


STYLES: dict[str, CaptionStyle] = {
    style.id: style
    for style in (
        CaptionStyle(
            id="pop",
            name="Pop",
            description=(
                "Bold white words; the spoken word turns yellow and pops. "
                "The classic short-form look."
            ),
            font="Arial Black",
            bold=True,
            size=0.06,
            uppercase=True,
            text_color="FFFFFF",
            active_color="FFE14D",
            emphasis_color="4DFF88",
            outline_color="000000",
            outline=0.12,
            shadow=0.06,
            animation="pop",
            position="lower_third",
            words_per_page=3,
            max_chars_per_line=16,
            lines=1,
        ),
        CaptionStyle(
            id="karaoke",
            name="Karaoke",
            description=("Two calm lines; color sweeps through each word exactly as it is spoken."),
            font="Arial",
            bold=True,
            size=0.045,
            uppercase=False,
            text_color="FFFFFF",
            active_color="38E1FF",
            emphasis_color="FFD43B",
            outline_color="111111",
            outline=0.1,
            shadow=0.05,
            animation="karaoke",
            position="lower_third",
            words_per_page=8,
            max_chars_per_line=24,
            lines=2,
        ),
        CaptionStyle(
            id="box",
            name="Box",
            description=("The spoken word sits on a bright pill; the rest stays clean white."),
            font="Arial Black",
            bold=True,
            size=0.055,
            uppercase=True,
            text_color="FFFFFF",
            active_color="FFFFFF",
            emphasis_color="FFE14D",
            outline_color="000000",
            outline=0.1,
            shadow=0.0,
            animation="box",
            position="lower_third",
            words_per_page=3,
            max_chars_per_line=16,
            lines=1,
            box_color="7C3AED",
        ),
        CaptionStyle(
            id="bold",
            name="Bold",
            description=(
                "One or two huge words at a time in the middle of the frame. Built for hooks."
            ),
            font="Impact",
            bold=False,
            size=0.085,
            uppercase=True,
            text_color="FFFFFF",
            active_color="FFFFFF",
            emphasis_color="FF4D4D",
            outline_color="000000",
            outline=0.1,
            shadow=0.08,
            animation="pop",
            position="middle",
            words_per_page=2,
            max_chars_per_line=12,
            lines=1,
        ),
        CaptionStyle(
            id="neon",
            name="Neon",
            description=("White words with a soft colored glow; the spoken word glows brighter."),
            font="Arial Black",
            bold=True,
            size=0.055,
            uppercase=True,
            text_color="FFFFFF",
            active_color="7DFFE4",
            emphasis_color="FFE14D",
            outline_color="FF2BD6",
            outline=0.06,
            shadow=0.0,
            animation="glow",
            position="lower_third",
            words_per_page=3,
            max_chars_per_line=16,
            lines=1,
        ),
        CaptionStyle(
            id="minimal",
            name="Minimal",
            description=(
                "Plain sentence-case subtitles with a soft shadow. For interviews and talks."
            ),
            font="Helvetica Neue",
            bold=False,
            size=0.036,
            uppercase=False,
            text_color="FFFFFF",
            active_color="FFFFFF",
            emphasis_color="FFFFFF",
            outline_color="000000",
            outline=0.05,
            shadow=0.07,
            animation="none",
            position="bottom",
            words_per_page=10,
            max_chars_per_line=28,
            lines=2,
        ),
    )
}


def get_style(style_id: str) -> CaptionStyle:
    style = STYLES.get(style_id)
    if style is None:
        raise ReelferryError(
            ErrorCode.INVALID_INPUT,
            f"unknown caption style {style_id!r}",
            hint=f"Available: {', '.join(STYLES)}",
        )
    return style


def style_info(style: CaptionStyle) -> CaptionStyleInfo:
    return CaptionStyleInfo(
        id=style.id,
        name=style.name,
        description=style.description,
        font=style.font,
        uppercase=style.uppercase,
        animation=style.animation,
        position=style.position,
        words_per_page=style.words_per_page,
        lines=style.lines,
    )
