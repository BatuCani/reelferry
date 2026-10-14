"""CTC forced alignment: exact word timings for a known text.

Whisper's own word timings come from attention weights and drift by a few
hundred milliseconds. Here a wav2vec2 CTC acoustic model (run with ONNX
Runtime) scores every 20 ms frame against the characters of the transcript,
and a Viterbi pass over the CTC trellis finds the best path. Word boundaries
fall where that path enters and leaves each word's characters.

Alignment runs per Whisper segment, so memory stays small for long videos.
Languages without an aligner model keep Whisper's timings.
"""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from reelferry.engine.audio import SAMPLE_RATE, FloatArray, require_free_disk
from reelferry.models import Word

# Permissively licensed wav2vec2 CTC models converted to ONNX, by language.
ALIGN_MODELS: dict[str, tuple[str, str, int]] = {
    "en": ("Xenova/wav2vec2-base-960h", "onnx/model_quantized.onnx", 96_000_000),
    "de": ("XDcobra/wav2vec2-large-xlsr-53-german-ONNX", "onnx/model_int8.onnx", 356_000_000),
    "fr": ("XDcobra/wav2vec2-large-xlsr-53-french-ONNX", "onnx/model_int8.onnx", 356_000_000),
    "es": ("XDcobra/wav2vec2-large-xlsr-53-spanish-ONNX", "onnx/model_int8.onnx", 356_000_000),
    "nl": ("XDcobra/wav2vec2-large-xlsr-53-dutch-ONNX", "onnx/model_int8.onnx", 356_000_000),
    "pl": ("XDcobra/wav2vec2-large-xlsr-53-polish-ONNX", "onnx/model_int8.onnx", 356_000_000),
}

PAD_SECONDS = 0.4
MIN_MEAN_LOGPROB = -6.0
NEG_INF = np.float32(-1e30)


def ctc_viterbi(log_probs: npt.NDArray[np.float32], tokens: list[int], blank: int) -> list[int]:
    """Best CTC path. Returns, per frame, the token index it is assigned to (-1 for blank)."""
    frames = log_probs.shape[0]
    ext = np.full(2 * len(tokens) + 1, blank, dtype=np.int64)
    ext[1::2] = tokens
    states = ext.shape[0]
    # Skipping over a blank is allowed only between two different tokens.
    can_skip = np.zeros(states, dtype=bool)
    can_skip[3::2] = ext[3::2] != ext[1:-2:2]

    alpha = np.full(states, NEG_INF, dtype=np.float32)
    alpha[0] = log_probs[0, blank]
    if states > 1:
        alpha[1] = log_probs[0, ext[1]]
    back = np.zeros((frames, states), dtype=np.int8)
    for t in range(1, frames):
        stay = alpha
        step = np.concatenate(([NEG_INF], alpha[:-1]))
        skip = np.where(can_skip, np.concatenate(([NEG_INF, NEG_INF], alpha[:-2])), NEG_INF)
        stacked = np.stack([stay, step, skip])
        choice = np.argmax(stacked, axis=0)
        back[t] = choice
        alpha = stacked[choice, np.arange(states)] + log_probs[t, ext]

    state = states - 1 if states == 1 or alpha[-1] >= alpha[-2] else states - 2
    path = [0] * frames
    for t in range(frames - 1, -1, -1):
        path[t] = (state - 1) // 2 if state % 2 == 1 else -1
        state -= int(back[t, state])
    return path


def _log_softmax(logits: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    result: npt.NDArray[np.float32] = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    return result


@dataclass
class AlignResult:
    words: list[Word]
    aligned_words: int
    total_words: int


class CtcAligner:
    def __init__(self, model_dir_files: tuple[Path, Path, Path, Path | None]) -> None:
        import onnxruntime

        model_path, vocab_path, config_path, preproc_path = model_dir_files
        self.vocab: dict[str, int] = json.loads(vocab_path.read_text(encoding="utf-8"))
        config: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
        preproc: dict[str, Any] = (
            json.loads(preproc_path.read_text(encoding="utf-8")) if preproc_path else {}
        )
        self.blank = int(config.get("pad_token_id", 0))
        self.normalize = bool(preproc.get("do_normalize", True))
        self.delimiter = self.vocab.get("|")
        letters = [k for k in self.vocab if len(k) == 1 and k.isalpha()]
        self.uppercase = bool(letters) and all(k.isupper() for k in letters)
        options = onnxruntime.SessionOptions()
        options.log_severity_level = 3
        self.session = onnxruntime.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    @classmethod
    def for_language(cls, language: str | None, cache_dir: Path) -> CtcAligner | None:
        if not language or language not in ALIGN_MODELS:
            return None
        from huggingface_hub import hf_hub_download

        repo, filename, size = ALIGN_MODELS[language]
        require_free_disk(cache_dir, size + 500_000_000, "the alignment model")
        model = Path(hf_hub_download(repo, filename))
        vocab = Path(hf_hub_download(repo, "vocab.json"))
        config = Path(hf_hub_download(repo, "config.json"))
        try:
            preproc: Path | None = Path(hf_hub_download(repo, "preprocessor_config.json"))
        except Exception:
            preproc = None
        return cls((model, vocab, config, preproc))

    def _tokens(self, text: str) -> list[int]:
        out: list[int] = []
        for char in text.upper() if self.uppercase else text.lower():
            if char in self.vocab:
                out.append(self.vocab[char])
                continue
            base = "".join(
                c for c in unicodedata.normalize("NFKD", char) if not unicodedata.combining(c)
            )
            out.extend(self.vocab[c] for c in base if c in self.vocab)
        return [t for t in out if t not in (self.blank, self.delimiter)]

    def _emissions(self, audio: FloatArray) -> npt.NDArray[np.float32]:
        x = audio.astype(np.float32)
        if self.normalize:
            x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
        logits = self.session.run(None, {self.input_name: x[None, :]})[0][0]
        return _log_softmax(np.asarray(logits, dtype=np.float32))

    def align_segment(
        self,
        audio: FloatArray,
        words: list[Word],
        *,
        lower: float = 0.0,
        upper: float | None = None,
    ) -> list[Word] | None:
        """Align one segment's words. Returns None when alignment is not trustworthy.

        The audio window is padded around the segment but never reaches into the
        neighbouring segments (``lower``/``upper``), so their speech cannot pull words away.
        """
        if not words:
            return words
        total = len(audio) / SAMPLE_RATE
        upper = total if upper is None else min(total, upper)
        win_start = max(0.0, lower, words[0].start - PAD_SECONDS)
        win_end = min(upper, words[-1].end + PAD_SECONDS)
        if win_end <= win_start:
            return None
        chunk = audio[int(win_start * SAMPLE_RATE) : int(win_end * SAMPLE_RATE)]
        if len(chunk) < SAMPLE_RATE // 10:
            return None
        emissions = self._emissions(chunk)
        frames = emissions.shape[0]
        seconds_per_frame = (len(chunk) / SAMPLE_RATE) / frames

        sequence: list[int] = []
        owner: list[int] = []  # word index for each token
        for index, word in enumerate(words):
            ids = self._tokens(word.text)
            if not ids:
                continue
            if sequence and self.delimiter is not None:
                sequence.append(self.delimiter)
                owner.append(-1)
            sequence.extend(ids)
            owner.extend([index] * len(ids))
        if not sequence or frames < len(sequence):
            return None

        path = ctc_viterbi(emissions, sequence, self.blank)
        spans: dict[int, list[int]] = {}
        scores: list[float] = []
        for frame, token in enumerate(path):
            if token < 0:
                continue
            word_index = owner[token]
            scores.append(float(emissions[frame, sequence[token]]))
            if word_index < 0:
                continue
            span = spans.setdefault(word_index, [frame, frame])
            span[1] = frame
        if not scores or float(np.mean(scores)) < MIN_MEAN_LOGPROB:
            return None

        timed: list[Word | None] = [None] * len(words)
        for index, (first, last) in spans.items():
            start = win_start + first * seconds_per_frame
            end = win_start + (last + 1) * seconds_per_frame
            timed[index] = words[index].model_copy(update={"start": start, "end": end})
        return _fill_gaps(words, timed)

    def align(
        self,
        audio: FloatArray,
        segments: list[list[Word]],
        progress: Callable[[float], None],
    ) -> AlignResult:
        out: list[Word] = []
        aligned = 0
        total = sum(len(s) for s in segments)
        for number, segment in enumerate(segments):
            lower = segments[number - 1][-1].end if number > 0 and segments[number - 1] else 0.0
            nxt = segments[number + 1] if number + 1 < len(segments) else None
            upper = nxt[0].start if nxt else None
            result = self.align_segment(audio, segment, lower=lower, upper=upper)
            if result is None:
                out.extend(segment)
            else:
                out.extend(result)
                aligned += len(segment)
            progress((number + 1) / max(1, len(segments)))
        return AlignResult(words=make_monotonic(out), aligned_words=aligned, total_words=total)


MIN_WORD_SECONDS = 0.05


def _fill_gaps(words: list[Word], timed: list[Word | None]) -> list[Word]:
    """Time words without alignable characters (e.g. '$49') by Whisper's own timing.

    The CTC sequence has no tokens for their speech, so the path hands that audio to the
    aligned neighbours. The run keeps Whisper's span (kept inside the neighbours) and the
    neighbours' facing edges are trimmed back to it.
    """
    timed = list(timed)
    result: list[Word] = []
    index = 0
    while index < len(words):
        item = timed[index]
        if item is not None:
            result.append(item)
            index += 1
            continue
        run_end = index
        while run_end < len(words) and timed[run_end] is None:
            run_end += 1
        left = result[-1] if result else None
        right = timed[run_end] if run_end < len(words) else None
        lo = left.start + MIN_WORD_SECONDS if left is not None else 0.0
        hi = right.end - MIN_WORD_SECONDS if right is not None else float("inf")
        hi = max(hi, lo)
        run_start = min(max(words[index].start, lo), hi)
        run_stop = min(max(words[run_end - 1].end, run_start), hi)
        if left is not None and left.end > run_start:
            result[-1] = left.model_copy(update={"end": run_start})
        for word in words[index:run_end]:
            start = min(max(word.start, run_start), run_stop)
            end = min(max(word.end, start), run_stop)
            result.append(word.model_copy(update={"start": start, "end": end}))
        if right is not None and right.start < run_stop:
            timed[run_end] = right.model_copy(update={"start": run_stop})
        index = run_end
    return result


def make_monotonic(words: list[Word]) -> list[Word]:
    out: list[Word] = []
    prev_end = 0.0
    for word in words:
        start = max(word.start, prev_end)
        end = max(word.end, start + 0.02)
        out.append(word.model_copy(update={"start": round(start, 3), "end": round(end, 3)}))
        prev_end = end
    return out
