from __future__ import annotations

import numpy as np

from reelferry.engine.align import ctc_viterbi, make_monotonic
from reelferry.engine.suggest import SuggestRequest, build_segments, snap_to_words, suggest
from reelferry.engine.transcribe import _cloud_words
from reelferry.models import Word
from reelferry.operations.transcripts import Replacement, apply_replacement

from .conftest import FakeChat, script_words


def _emissions(frames: list[int], vocab: int) -> np.ndarray:
    """Log-probs that strongly prefer the given token per frame."""
    probs = np.full((len(frames), vocab), 0.01, dtype=np.float32)
    for t, token in enumerate(frames):
        probs[t, token] = 0.9
    probs /= probs.sum(axis=1, keepdims=True)
    return np.log(probs)


def test_ctc_viterbi_finds_token_spans() -> None:
    blank = 0
    # blank blank A A blank B B B blank blank A blank
    frames = [0, 0, 1, 1, 0, 2, 2, 2, 0, 0, 1, 0]
    path = ctc_viterbi(_emissions(frames, 3), [1, 2, 1], blank)
    assert [i for i, token in enumerate(path) if token == 0] == [2, 3]
    assert [i for i, token in enumerate(path) if token == 1] == [5, 6, 7]
    assert [i for i, token in enumerate(path) if token == 2] == [10]


def test_ctc_viterbi_separates_repeated_tokens_with_blank() -> None:
    frames = [1, 1, 0, 1, 1]
    path = ctc_viterbi(_emissions(frames, 2), [1, 1], 0)
    assert path[:2] == [0, 0]
    assert path[3:] == [1, 1]


def test_make_monotonic_fixes_overlaps() -> None:
    words = [
        Word(text="a", start=0.0, end=0.5),
        Word(text="b", start=0.4, end=0.45),
    ]
    fixed = make_monotonic(words)
    assert fixed[1].start >= fixed[0].end
    assert fixed[1].end > fixed[1].start


def test_segments_split_on_sentences() -> None:
    segments = build_segments(script_words())
    assert segments[0].text == "Welcome back to the show."
    assert segments[0].first_word == 0
    assert segments[-1].last_word == len(script_words()) - 1


def test_snap_to_word_edges() -> None:
    words = script_words()
    snapped = snap_to_words(words, 1.0, 3.33)
    assert snapped is not None
    start, end = snapped
    assert any(abs(w.start - start - 0.08) < 1e-6 for w in words)
    assert any(abs(w.end - end + 0.25) < 1e-6 for w in words)


def test_suggest_filters_snaps_and_dedupes() -> None:
    words = script_words()
    chat = FakeChat(
        [
            {
                "start": 2.0,
                "end": 10.0,
                "title": "Three habits",
                "score": 88,
                "reason": "clear list",
                "emphasis": ["habits"],
            },
            # Overlaps the first one heavily: dropped.
            {"start": 2.5, "end": 9.0, "title": "Dup", "score": 60, "reason": "", "emphasis": []},
            # Too short for the requested range: dropped.
            {
                "start": 15.0,
                "end": 16.0,
                "title": "Short",
                "score": 99,
                "reason": "",
                "emphasis": [],
            },
            {
                "start": 11.0,
                "end": 19.0,
                "title": "Meetings",
                "score": 75,
                "reason": "payoff",
                "emphasis": ["ten"],
            },
        ]
    )
    picked, cost, model = suggest(
        chat,
        model="test/model",
        words=words,
        req=SuggestRequest(
            goal="habits", count=5, min_seconds=8, max_seconds=20, language="en", title="Talk"
        ),
        duration=words[-1].end + 1,
    )
    assert [c.title for c in picked] == ["Three habits", "Meetings"]
    assert picked[0].score == 88
    assert cost > 0 and model == "test/model"
    assert "Goal: habits" in chat.prompts[0]
    assert "[0.5-" in chat.prompts[0]


def test_replacement_merges_words_and_keeps_span() -> None:
    words = [
        Word(text="We", start=0, end=0.2),
        Word(text="use", start=0.2, end=0.4),
        Word(text="open", start=0.4, end=0.6),
        Word(text="AI,", start=0.6, end=0.9),
        Word(text="daily.", start=0.9, end=1.2),
    ]
    out, changed = apply_replacement(words, Replacement(find="open ai", replace="OpenAI"))
    assert changed == 2
    assert [w.text for w in out] == ["We", "use", "OpenAI,", "daily."]
    assert out[2].start == 0.4 and out[2].end == 0.9


def test_replacement_one_to_one_keeps_each_timing() -> None:
    words = script_words()
    out, changed = apply_replacement(words, Replacement(find="six", replace="6"))
    assert changed == 1
    index = next(i for i, w in enumerate(words) if w.text == "six.")
    assert out[index].text == "6."
    assert (out[index].start, out[index].end) == (words[index].start, words[index].end)


def test_cloud_words_offset_and_fallback() -> None:
    data = {"words": [{"word": " hi", "start": 0.0, "end": 0.4}]}
    assert _cloud_words(data, offset=600)[0].start == 600
    seg_only = {"segments": [{"start": 1.0, "end": 2.0, "text": "two words"}]}
    words = _cloud_words(seg_only, offset=0)
    assert [w.text for w in words] == ["two", "words"]
    assert words[1].end == 2.0
