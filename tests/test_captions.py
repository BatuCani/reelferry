from __future__ import annotations

import itertools
import re

from reelferry.captions.build import build_pages, render_caption
from reelferry.captions.formats import to_ass, to_srt, to_vtt
from reelferry.captions.layout import CaptionWord, clip_words, paginate
from reelferry.captions.styles import STYLES, get_style
from reelferry.models import Word

from .conftest import script_words


def test_clip_words_shift_to_clip_time_and_mark_emphasis() -> None:
    words = clip_words(script_words(), 2.0, 8.0, emphasis=["habits"])
    assert words[0].start >= 0
    assert all(w.end <= 6.0 for w in words)
    flagged = {w.text for w in words if w.emphasis}
    assert "habits" in flagged


def test_numbers_are_emphasized() -> None:
    words = [Word(text="saved", start=0, end=0.3), Word(text="10", start=0.3, end=0.6)]
    assert [w.emphasis for w in clip_words(words, 0, 1)] == [False, True]


def test_pages_respect_word_limit_and_sentence_ends() -> None:
    style = get_style("pop")
    pages = paginate(
        clip_words(script_words(), 0, 60),
        words_per_page=style.words_per_page,
        max_chars_per_line=style.max_chars_per_line,
        lines=style.lines,
    )
    assert all(len(p.words) <= style.words_per_page for p in pages)
    for page in pages:
        for word in page.words[:-1]:
            assert not word.text.endswith((".", "?", "!"))
    starts = [p.start for p in pages]
    assert starts == sorted(starts)
    for current, nxt in itertools.pairwise(pages):
        assert current.end <= nxt.start + 1e-6


def test_two_line_styles_balance_lines() -> None:
    style = get_style("karaoke")
    pages = paginate(
        clip_words(script_words(), 0, 60),
        words_per_page=style.words_per_page,
        max_chars_per_line=style.max_chars_per_line,
        lines=style.lines,
    )
    for page in pages:
        assert len(page.lines) <= 2
        for line in page.lines:
            assert len(" ".join(w.text for w in line)) <= style.max_chars_per_line + 12


def test_srt_and_vtt_format() -> None:
    pages, _ = build_pages(script_words(), start=0, end=10, style=get_style("pop"), emphasis=[])
    srt = to_srt(pages)
    assert srt.startswith("1\n00:00:00,")
    assert re.search(r"\d\d:\d\d:\d\d,\d{3} --> \d\d:\d\d:\d\d,\d{3}", srt)
    vtt = to_vtt(pages)
    assert vtt.startswith("WEBVTT\n")
    # Word-level timestamps inside cues.
    assert re.search(r"<\d\d:\d\d:\d\d\.\d{3}>", vtt)


def test_every_style_produces_valid_ass() -> None:
    for style_id, style in STYLES.items():
        pages, _ = build_pages(script_words(), start=0, end=12, style=style, emphasis=["six"])
        ass = to_ass(pages, style, 1080, 1920)
        assert "[Script Info]" in ass and "PlayResX: 1080" in ass, style_id
        dialogues = [line for line in ass.splitlines() if line.startswith("Dialogue:")]
        assert dialogues, style_id
        for line in dialogues:
            text = line.split(",", 9)[9]
            assert text.count("{") == text.count("}"), style_id
        if style.animation == "karaoke":
            assert "\\kf" in ass
        if style.uppercase:
            assert "WELCOME" in ass


def test_active_word_highlight_per_word_event() -> None:
    style = get_style("pop")
    pages, _ = build_pages(script_words(), start=0, end=3, style=style, emphasis=[])
    ass = to_ass(pages, style, 1080, 1920)
    events = [line for line in ass.splitlines() if line.startswith("Dialogue:")]
    words_on_screen = sum(len(p.words) for p in pages)
    assert len(events) == words_on_screen
    assert all("\\fscx" in e for e in events)


def test_json_captions_include_every_word() -> None:
    style = get_style("minimal")
    pages, words = build_pages(script_words(), start=0, end=60, style=style, emphasis=[])
    content = render_caption("json", pages, words, style, (1920, 1080))
    assert '"words"' in content and '"pages"' in content
    assert content.count('"text"') == len(words)


def test_braces_are_escaped_in_ass() -> None:
    style = get_style("minimal")
    pages = paginate(
        [CaptionWord("{hack}", 0, 0.5)], words_per_page=5, max_chars_per_line=30, lines=1
    )
    assert "(hack)" in to_ass(pages, style, 1920, 1080)


def test_pages_do_not_end_on_dangling_words() -> None:
    text = "is reading twenty pages a day and saying no to most meetings every week"
    words = [Word(text=t, start=i * 0.35, end=i * 0.35 + 0.3) for i, t in enumerate(text.split())]
    style = get_style("karaoke")
    pages, _ = build_pages(words, start=0, end=10, style=style, emphasis=[])
    for page in pages[:-1]:
        assert page.words[-1].text not in {"a", "to", "and", "the"}, page.text
