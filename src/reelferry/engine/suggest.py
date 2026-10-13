"""Find the strongest clip-worthy moments in a transcript with an LLM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from reelferry.engine.llm import ChatClient
from reelferry.models import CandidateBase, TranscriptSegment, Word

MAX_PROMPT_CHARS = 360_000  # about 90k tokens per request; longer videos are windowed
WINDOW_OVERLAP_SECONDS = 120.0

SYSTEM_PROMPT = """You are a senior short-form video editor. You pick moments from long videos \
that work as standalone vertical clips (TikTok, Reels, Shorts) and you write their hooks.

A strong clip:
- opens with a hook in the first two seconds (a bold claim, a question, tension, a punchline setup),
- makes sense without the rest of the video,
- has a clear arc and ends on a payoff, not mid-thought,
- is quotable or emotional, surprising, useful or funny.

Rules:
- Use only timestamps that appear in the transcript. Start at the first word of a sentence and \
end right after the last word of a sentence.
- Respect the requested length range.
- Clips must not overlap.
- Title: a short hook headline in the transcript's language, max 70 characters, no hashtags.
- Score 0-100: how likely the clip performs. Be honest and spread the scores.
- Reason: one sentence on why the moment works.
- Emphasis: up to 6 single words from the clip that captions should highlight \
(names, numbers, the punchline word).
"""

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "clips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number", "description": "Start time in seconds"},
                    "end": {"type": "number", "description": "End time in seconds"},
                    "title": {"type": "string"},
                    "score": {"type": "integer"},
                    "reason": {"type": "string"},
                    "emphasis": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["start", "end", "title", "score", "reason", "emphasis"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["clips"],
    "additionalProperties": False,
}


def build_segments(
    words: list[Word], max_gap: float = 1.2, max_len: float = 25.0
) -> list[TranscriptSegment]:
    """Group words into sentence-like segments (for prompts and for reading)."""
    segments: list[TranscriptSegment] = []
    first = 0
    for index, word in enumerate(words):
        last_of_segment = index == len(words) - 1
        if not last_of_segment:
            nxt = words[index + 1]
            last_of_segment = (
                word.text.endswith((".", "?", "!", "…"))
                or nxt.start - word.end > max_gap
                or nxt.end - words[first].start > max_len
            )
        if last_of_segment:
            chunk = words[first : index + 1]
            segments.append(
                TranscriptSegment(
                    start=chunk[0].start,
                    end=chunk[-1].end,
                    text=" ".join(w.text for w in chunk),
                    first_word=first,
                    last_word=index,
                )
            )
            first = index + 1
    return segments


def _stamp(seconds: float) -> str:
    return f"{seconds:.1f}"


def _windows(segments: list[TranscriptSegment]) -> list[list[TranscriptSegment]]:
    windows: list[list[TranscriptSegment]] = []
    current: list[TranscriptSegment] = []
    size = 0
    for segment in segments:
        line = len(segment.text) + 24
        if current and size + line > MAX_PROMPT_CHARS:
            windows.append(current)
            overlap_from = current[-1].end - WINDOW_OVERLAP_SECONDS
            current = [s for s in current if s.start >= overlap_from]
            size = sum(len(s.text) + 24 for s in current)
        current.append(segment)
        size += line
    if current:
        windows.append(current)
    return windows


@dataclass
class SuggestRequest:
    goal: str
    count: int
    min_seconds: float
    max_seconds: float
    language: str | None
    title: str


def _user_prompt(req: SuggestRequest, segments: list[TranscriptSegment], count: int) -> str:
    lines = [f"[{_stamp(s.start)}-{_stamp(s.end)}] {s.text}" for s in segments]
    language = f" Write titles in language '{req.language}'." if req.language else ""
    return (
        f"Video: {req.title}\n"
        f"Goal: {req.goal}\n"
        f"Find up to {count} clips, each between {req.min_seconds:.0f} and "
        f"{req.max_seconds:.0f} seconds long.{language}\n"
        "Transcript (each line is [start-end seconds] text):\n\n" + "\n".join(lines)
    )


def snap_to_words(
    words: list[Word], start: float, end: float, *, tolerance: float = 1.5
) -> tuple[float, float] | None:
    """Move boundaries onto real word edges so clips never cut a word in half."""
    if not words or end <= start:
        return None
    starts = [w for w in words if w.start >= start - tolerance]
    if not starts:
        return None
    first = min(starts, key=lambda w: abs(w.start - start))
    ends = [w for w in words if w.end <= end + tolerance and w.end > first.start]
    if not ends:
        return None
    last = min(ends, key=lambda w: abs(w.end - end))
    if last.end <= first.start:
        return None
    pad_in = 0.08
    pad_out = 0.25
    return max(0.0, first.start - pad_in), last.end + pad_out


def _overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    shorter = min(a[1] - a[0], b[1] - b[0])
    inter = min(a[1], b[1]) - max(a[0], b[0])
    return inter > 0.3 * shorter


def suggest(
    client: ChatClient,
    *,
    model: str,
    words: list[Word],
    req: SuggestRequest,
    duration: float,
) -> tuple[list[CandidateBase], float, str]:
    segments = build_segments(words)
    windows = _windows(segments)
    per_window = req.count if len(windows) == 1 else max(2, -(-req.count * 2 // len(windows)))
    raw: list[dict[str, Any]] = []
    cost = 0.0
    used_model = model
    for window in windows:
        result = client.json_chat(
            model=model,
            system=SYSTEM_PROMPT,
            user=_user_prompt(req, window, per_window),
            schema_name="clip_suggestions",
            schema=RESPONSE_SCHEMA,
        )
        cost += result.cost_usd
        used_model = result.model
        clips = result.data.get("clips")
        if isinstance(clips, list):
            raw.extend(c for c in clips if isinstance(c, dict))

    picked: list[CandidateBase] = []
    for item in sorted(raw, key=lambda c: -_as_int(c.get("score"))):
        try:
            start, end = float(item["start"]), float(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        snapped = snap_to_words(words, start, min(end, duration))
        if snapped is None:
            continue
        length = snapped[1] - snapped[0]
        if length < req.min_seconds * 0.8 or length > req.max_seconds * 1.2:
            continue
        if any(_overlaps(snapped, (c.start, c.end)) for c in picked):
            continue
        emphasis = [str(e) for e in item.get("emphasis") or [] if str(e).strip()][:6]
        picked.append(
            CandidateBase(
                start=round(snapped[0], 3),
                end=round(min(snapped[1], duration), 3),
                title=str(item.get("title") or "Untitled clip").strip()[:120],
                score=max(0, min(100, _as_int(item.get("score")))),
                reason=str(item.get("reason") or "").strip(),
                emphasis=emphasis,
            )
        )
        if len(picked) >= req.count:
            break
    return picked, cost, used_model


def _as_int(value: Any) -> int:
    try:
        return round(float(value))
    except (TypeError, ValueError):
        return 0
