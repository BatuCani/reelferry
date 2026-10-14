from __future__ import annotations

import wave
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from reelferry.engine.align import _fill_gaps, ctc_viterbi, make_monotonic
from reelferry.engine.suggest import SuggestRequest, build_segments, snap_to_words, suggest
from reelferry.engine.transcribe import LocalWhisper, _cloud_words
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


def _silent_wav(path: Path, seconds: float) -> Path:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\0\0" * int(16_000 * seconds))
    return path


def _fake_whisper(
    monkeypatch: pytest.MonkeyPatch, language: str, words: list[tuple[str, float, float]]
) -> None:
    class FakeModel:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def transcribe(self, *args: Any, **kwargs: Any) -> tuple[Any, Any]:
            segment = SimpleNamespace(
                end=words[-1][2],
                words=[SimpleNamespace(word=w, start=s, end=e, probability=0.9) for w, s, e in words],
            )
            return iter([segment]), SimpleNamespace(language=language)

    monkeypatch.setattr("faster_whisper.WhisperModel", FakeModel)


def test_local_whisper_joins_continuation_tokens_into_one_word(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # faster-whisper splits "10,000" into " 10" and ",000"; no leading space means same word.
    _fake_whisper(
        monkeypatch,
        "en",
        [(" It", 0.0, 0.2), (" got", 0.2, 0.4), (" 10", 0.4, 0.7), (",000", 0.7, 1.1),
         (" views.", 1.1, 1.5)],
    )
    raw = LocalWhisper(str(tmp_path), "cpu", "int8").transcribe(
        _silent_wav(tmp_path / "a.wav", 2), language=None, vocabulary=[],
        progress=lambda f, m: None, ctx=None,
    )
    words = raw.segments[0]
    assert [w.text for w in words] == ["It", "got", "10,000", "views."]
    assert (words[2].start, words[2].end) == (0.4, 1.1)


def test_local_whisper_keeps_words_of_unspaced_languages_apart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_whisper(monkeypatch, "ja", [("今日", 0.0, 0.4), ("は", 0.4, 0.6), ("晴れ", 0.6, 1.0)])
    raw = LocalWhisper(str(tmp_path), "cpu", "int8").transcribe(
        _silent_wav(tmp_path / "a.wav", 2), language=None, vocabulary=[],
        progress=lambda f, m: None, ctx=None,
    )
    assert [w.text for w in raw.segments[0]] == ["今日", "は", "晴れ"]


def test_unalignable_number_keeps_whisper_timing_and_neighbours_yield_its_audio() -> None:
    # Live case: CTC has no tokens for "$49", so "per" swallowed the spoken number.
    words = [
        Word(text="charge", start=5.0, end=5.42),
        Word(text="$49", start=5.42, end=6.12),
        Word(text="per", start=6.12, end=6.4),
        Word(text="month", start=6.4, end=6.8),
    ]
    timed: list[Word | None] = [
        words[0].model_copy(update={"start": 5.02, "end": 5.42}),
        None,
        words[2].model_copy(update={"start": 5.52, "end": 6.43}),
        words[3].model_copy(update={"start": 6.45, "end": 6.82}),
    ]
    out = _fill_gaps(words, timed)
    assert [(w.text, w.start, w.end) for w in out] == [
        ("charge", 5.02, 5.42),
        ("$49", 5.42, 6.12),
        ("per", 6.12, 6.43),
        ("month", 6.45, 6.82),
    ]


def test_unalignable_word_stays_inside_its_aligned_neighbours() -> None:
    words = [
        Word(text="at", start=1.0, end=1.2),
        Word(text="42", start=0.2, end=3.0),
        Word(text="now", start=2.0, end=2.3),
    ]
    timed: list[Word | None] = [
        words[0].model_copy(update={"start": 1.0, "end": 1.3}),
        None,
        words[2].model_copy(update={"start": 1.6, "end": 2.0}),
    ]
    at, number, now = _fill_gaps(words, timed)
    assert at.start < number.start < number.end < now.end
    assert at.end - at.start > 0 and now.end - now.start > 0
