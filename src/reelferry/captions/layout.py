"""Turn word timings into caption pages: short groups of words shown together."""

from __future__ import annotations

import re
from dataclasses import dataclass

from reelferry.models import Word

SENTENCE_END = (".", "?", "!", "…")
CLAUSE_END = (",", ";", ":", "-", "\u2013")
MAX_GAP = 0.7
MAX_PAGE_SECONDS = 3.5
HOLD_SECONDS = 0.5
BRIDGE_GAP = 0.35
# Short words that read badly as the last word on screen ("pages a" / "day").
DANGLING = {
    "a",
    "an",
    "the",
    "to",
    "of",
    "in",
    "on",
    "at",
    "for",
    "and",
    "or",
    "but",
    "my",
    "your",
    "our",
    "is",
    "are",
    "was",
    "with",
    "from",
    "by",
    "as",
    "if",
    "so",
    "no",
    "most",
    "der",
    "die",
    "das",
    "ein",
    "eine",
    "und",
    "oder",
    "zu",
    "im",
    "mit",
    "von",
    "für",
    "auf",
    "le",
    "la",
    "les",
    "un",
    "une",
    "de",
    "et",
    "el",
    "los",
    "las",
    "y",
    "en",
}


@dataclass(frozen=True)
class CaptionWord:
    text: str
    start: float
    end: float
    emphasis: bool = False


@dataclass(frozen=True)
class Page:
    lines: list[list[CaptionWord]]
    start: float
    end: float

    @property
    def words(self) -> list[CaptionWord]:
        return [w for line in self.lines for w in line]

    @property
    def text(self) -> str:
        return "\n".join(" ".join(w.text for w in line) for line in self.lines)


def _norm(text: str) -> str:
    return re.sub(r"[^\w']", "", text.lower())


def clip_words(
    words: list[Word], start: float, end: float, emphasis: list[str] | None = None
) -> list[CaptionWord]:
    """Words inside ``start..end``, shifted so the clip starts at zero."""
    targets = {_norm(token) for phrase in emphasis or [] for token in phrase.split()}
    targets.discard("")
    out: list[CaptionWord] = []
    for word in words:
        if word.end <= start or word.start >= end:
            continue
        mid = (word.start + word.end) / 2
        if mid < start or mid > end:
            continue
        s = max(word.start, start) - start
        e = min(word.end, end) - start
        key = _norm(word.text)
        emph = key in targets or bool(re.search(r"\d", word.text))
        out.append(CaptionWord(text=word.text, start=round(s, 3), end=round(e, 3), emphasis=emph))
    return out


def _split_lines(words: list[CaptionWord], max_chars: int, lines: int) -> list[list[CaptionWord]]:
    if lines <= 1 or len(" ".join(w.text for w in words)) <= max_chars:
        return [words]
    # Two lines: choose the break that balances line lengths best.
    best: list[list[CaptionWord]] = [words]
    best_cost = float("inf")
    for cut in range(1, len(words)):
        top = len(" ".join(w.text for w in words[:cut]))
        bottom = len(" ".join(w.text for w in words[cut:]))
        cost = abs(top - bottom) + (1000 if max(top, bottom) > max_chars else 0)
        if cost < best_cost:
            best, best_cost = [words[:cut], words[cut:]], cost
    return best


def paginate(
    words: list[CaptionWord],
    *,
    words_per_page: int,
    max_chars_per_line: int,
    lines: int,
    clip_end: float | None = None,
) -> list[Page]:
    groups: list[list[CaptionWord]] = []
    current: list[CaptionWord] = []
    budget = max_chars_per_line * lines
    for word in words:
        if current:
            prev = current[-1]
            chars = len(" ".join(w.text for w in [*current, word]))
            if (
                len(current) >= words_per_page
                or chars > budget
                or word.start - prev.end > MAX_GAP
                or prev.text.endswith(SENTENCE_END)
                or word.end - current[0].start > MAX_PAGE_SECONDS
                or (len(current) >= max(2, words_per_page // 2) and prev.text.endswith(CLAUSE_END))
            ):
                carry: list[CaptionWord] = []
                while (
                    len(current) > 2
                    and len(carry) < 2
                    and not current[-1].text.endswith((*SENTENCE_END, *CLAUSE_END))
                    and _norm(current[-1].text) in DANGLING
                    and word.start - prev.end <= MAX_GAP
                ):
                    carry.insert(0, current.pop())
                groups.append(current)
                current = carry
        current.append(word)
    if current:
        groups.append(current)

    pages: list[Page] = []
    for index, group in enumerate(groups):
        start = group[0].start
        end = group[-1].end + HOLD_SECONDS
        if index + 1 < len(groups):
            next_start = groups[index + 1][0].start
            end = next_start if next_start - group[-1].end < BRIDGE_GAP else min(end, next_start)
        if clip_end is not None:
            end = min(end, clip_end)
        end = max(end, group[-1].end)
        pages.append(
            Page(
                lines=_split_lines(group, max_chars_per_line, lines),
                start=round(start, 3),
                end=round(end, 3),
            )
        )
    return pages
