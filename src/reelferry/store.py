"""Reading and writing domain records. Jobs live in ``jobs.py``."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from reelferry.db import Database, dumps, new_id, now_iso
from reelferry.errors import ErrorCode, ReelferryError, not_found
from reelferry.models import (
    Clip,
    ClipCandidate,
    ClipStatus,
    Render,
    Source,
    SourceStatus,
    TranscriptInfo,
    Word,
)


def _bool(value: Any) -> bool | None:
    return None if value is None else bool(value)


# Sources


def _source(db: Database, row: sqlite3.Row) -> Source:
    version = db.one("SELECT MAX(version) FROM transcripts WHERE source_id = ?", (row["id"],))
    return Source(
        id=row["id"],
        kind=row["kind"],
        uri=row["uri"],
        title=row["title"],
        path=row["path"],
        status=SourceStatus(row["status"]),
        duration=row["duration"],
        width=row["width"],
        height=row["height"],
        fps=row["fps"],
        has_audio=_bool(row["has_audio"]),
        language=row["language"],
        transcript_version=version[0] if version else None,
        error=row["error"],
        created_at=row["created_at"],
    )


def insert_source(
    db: Database,
    *,
    kind: str,
    uri: str,
    title: str,
    status: SourceStatus,
    path: str | None = None,
    probe: dict[str, Any] | None = None,
) -> Source:
    source_id = new_id("src")
    ts = now_iso()
    probe = probe or {}
    db.execute(
        "INSERT INTO sources (id, kind, uri, title, path, status, duration, width, height, fps,"
        " has_audio, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            source_id,
            kind,
            uri,
            title,
            path,
            status.value,
            probe.get("duration"),
            probe.get("width"),
            probe.get("height"),
            probe.get("fps"),
            probe.get("has_audio"),
            ts,
            ts,
        ),
    )
    return get_source(db, source_id)


def update_source(db: Database, source_id: str, **fields: Any) -> Source:
    if fields:
        cols = ", ".join(f"{name} = ?" for name in fields)
        db.execute(
            f"UPDATE sources SET {cols}, updated_at = ? WHERE id = ?",
            (*fields.values(), now_iso(), source_id),
        )
    return get_source(db, source_id)


def get_source(db: Database, source_id: str) -> Source:
    row = db.one("SELECT * FROM sources WHERE id = ?", (source_id,))
    if row is None:
        raise not_found("source", source_id)
    return _source(db, row)


def ready_source(db: Database, source_id: str) -> Source:
    source = get_source(db, source_id)
    if source.status is not SourceStatus.READY or not source.path:
        raise ReelferryError(
            ErrorCode.CONFLICT,
            f"source {source_id!r} is {source.status.value}, not ready",
            hint="Wait for the download job to finish (wait_job).",
        )
    return source


def list_sources(db: Database, limit: int) -> list[Source]:
    rows = db.all("SELECT * FROM sources ORDER BY created_at DESC LIMIT ?", (limit,))
    return [_source(db, row) for row in rows]


# Transcripts


def _transcript_info(row: sqlite3.Row, word_count: int) -> TranscriptInfo:
    return TranscriptInfo(
        source_id=row["source_id"],
        version=row["version"],
        language=row["language"],
        engine=row["engine"],
        model=row["model"],
        alignment=row["alignment"],
        word_count=word_count,
        duration=row["duration"],
        created_at=row["created_at"],
        note=row["note"],
    )


def save_transcript(
    db: Database,
    *,
    source_id: str,
    words: list[Word],
    language: str | None,
    engine: str,
    model: str,
    alignment: str,
    duration: float,
    note: str | None = None,
) -> TranscriptInfo:
    payload = dumps([[w.text, round(w.start, 3), round(w.end, 3), w.probability] for w in words])
    with db.tx() as conn:
        row = conn.execute(
            "SELECT COALESCE(MAX(version), 0) FROM transcripts WHERE source_id = ?", (source_id,)
        ).fetchone()
        version = int(row[0]) + 1
        conn.execute(
            "INSERT INTO transcripts (source_id, version, language, engine, model, alignment,"
            " words, duration, note, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                source_id,
                version,
                language,
                engine,
                model,
                alignment,
                payload,
                duration,
                note,
                now_iso(),
            ),
        )
        if language:
            conn.execute(
                "UPDATE sources SET language = ?, updated_at = ? WHERE id = ?",
                (language, now_iso(), source_id),
            )
    info, _ = load_transcript(db, source_id, version)
    return info


def load_transcript(
    db: Database, source_id: str, version: int | None = None
) -> tuple[TranscriptInfo, list[Word]]:
    get_source(db, source_id)
    if version is None:
        row = db.one(
            "SELECT * FROM transcripts WHERE source_id = ? ORDER BY version DESC LIMIT 1",
            (source_id,),
        )
    else:
        row = db.one(
            "SELECT * FROM transcripts WHERE source_id = ? AND version = ?", (source_id, version)
        )
    if row is None:
        raise ReelferryError(
            ErrorCode.TRANSCRIPT_MISSING,
            f"source {source_id!r} has no transcript"
            + (f" version {version}" if version is not None else ""),
            hint="Run transcribe first.",
        )
    words = [
        Word(text=item[0], start=item[1], end=item[2], probability=item[3])
        for item in json.loads(row["words"])
    ]
    return _transcript_info(row, len(words)), words


# Candidates


def _candidate(row: sqlite3.Row) -> ClipCandidate:
    return ClipCandidate(
        id=row["id"],
        source_id=row["source_id"],
        job_id=row["job_id"],
        start=row["start"],
        end=row["end"],
        duration=round(row["end"] - row["start"], 3),
        title=row["title"],
        score=row["score"],
        reason=row["reason"],
        emphasis=json.loads(row["emphasis"]),
        created_at=row["created_at"],
    )


def insert_candidate(
    db: Database,
    *,
    source_id: str,
    job_id: str | None,
    start: float,
    end: float,
    title: str,
    score: int,
    reason: str,
    emphasis: list[str],
) -> ClipCandidate:
    cand_id = new_id("cand")
    db.execute(
        'INSERT INTO candidates (id, source_id, job_id, start, "end", title, score, reason,'
        " emphasis, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (cand_id, source_id, job_id, start, end, title, score, reason, dumps(emphasis), now_iso()),
    )
    return get_candidate(db, cand_id)


def get_candidate(db: Database, cand_id: str) -> ClipCandidate:
    row = db.one("SELECT * FROM candidates WHERE id = ?", (cand_id,))
    if row is None:
        raise not_found("candidate", cand_id)
    return _candidate(row)


def list_candidates(db: Database, source_id: str, job_id: str | None) -> list[ClipCandidate]:
    get_source(db, source_id)
    if job_id:
        rows = db.all(
            "SELECT * FROM candidates WHERE source_id = ? AND job_id = ? ORDER BY score DESC",
            (source_id, job_id),
        )
    else:
        rows = db.all(
            "SELECT * FROM candidates WHERE source_id = ? ORDER BY created_at DESC, score DESC",
            (source_id,),
        )
    return [_candidate(row) for row in rows]


# Clips and renders


def _render(row: sqlite3.Row) -> Render:
    return Render(
        id=row["id"],
        clip_id=row["clip_id"],
        job_id=row["job_id"],
        clip_revision=row["clip_revision"],
        video_path=row["video_path"],
        captions=json.loads(row["captions"]),
        format=row["format"],
        width=row["width"],
        height=row["height"],
        duration=row["duration"],
        size_bytes=row["size_bytes"],
        caption_style=row["caption_style"],
        burned_captions=bool(row["burned_captions"]),
        created_at=row["created_at"],
    )


def latest_render(db: Database, clip_id: str) -> Render | None:
    row = db.one(
        "SELECT * FROM renders WHERE clip_id = ? ORDER BY created_at DESC, rowid DESC LIMIT 1",
        (clip_id,),
    )
    return _render(row) if row else None


def _clip(db: Database, row: sqlite3.Row) -> Clip:
    return Clip(
        id=row["id"],
        source_id=row["source_id"],
        candidate_id=row["candidate_id"],
        title=row["title"],
        start=row["start"],
        end=row["end"],
        duration=round(row["end"] - row["start"], 3),
        format=row["format"],
        framing=row["framing"],
        crop_x=row["crop_x"],
        caption_style=row["caption_style"],
        burn_captions=bool(row["burn_captions"]),
        captions_enabled=bool(row["captions_enabled"]),
        status=ClipStatus(row["status"]),
        score=row["score"],
        reason=row["reason"],
        emphasis=json.loads(row["emphasis"]),
        revision=row["revision"],
        latest_render=latest_render(db, row["id"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def insert_clip(db: Database, **fields: Any) -> Clip:
    clip_id = new_id("clip")
    ts = now_iso()
    fields = {
        **fields,
        "id": clip_id,
        "emphasis": dumps(fields.get("emphasis", [])),
        "status": ClipStatus.DRAFT.value,
        "revision": 1,
        "created_at": ts,
        "updated_at": ts,
    }
    cols = ", ".join(f'"{name}"' for name in fields)
    marks = ", ".join("?" for _ in fields)
    db.execute(f"INSERT INTO clips ({cols}) VALUES ({marks})", tuple(fields.values()))
    return get_clip(db, clip_id)


def update_clip(db: Database, clip_id: str, *, bump_revision: bool, **fields: Any) -> Clip:
    get_clip(db, clip_id)
    if "emphasis" in fields:
        fields["emphasis"] = dumps(fields["emphasis"])
    sets = [f'"{name}" = ?' for name in fields]
    params: list[Any] = list(fields.values())
    if bump_revision:
        sets.append("revision = revision + 1")
    sets.append("updated_at = ?")
    params.append(now_iso())
    db.execute(f"UPDATE clips SET {', '.join(sets)} WHERE id = ?", (*params, clip_id))
    return get_clip(db, clip_id)


def get_clip(db: Database, clip_id: str) -> Clip:
    row = db.one("SELECT * FROM clips WHERE id = ?", (clip_id,))
    if row is None:
        raise not_found("clip", clip_id)
    return _clip(db, row)


def list_clips(
    db: Database,
    *,
    source_id: str | None,
    status: str | None,
    query: str | None,
    min_score: int | None,
    limit: int,
) -> list[Clip]:
    where: list[str] = []
    params: list[Any] = []
    if source_id:
        where.append("source_id = ?")
        params.append(source_id)
    if status:
        where.append("status = ?")
        params.append(status)
    if query:
        where.append("(title LIKE ? OR reason LIKE ?)")
        params.extend([f"%{query}%", f"%{query}%"])
    if min_score is not None:
        where.append("score >= ?")
        params.append(min_score)
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    rows = db.all(
        f"SELECT * FROM clips {clause} ORDER BY created_at DESC LIMIT ?", (*params, limit)
    )
    return [_clip(db, row) for row in rows]


def insert_render(db: Database, **fields: Any) -> Render:
    render_id = new_id("rnd")
    fields = {
        **fields,
        "id": render_id,
        "captions": dumps(fields["captions"]),
        "created_at": now_iso(),
    }
    cols = ", ".join(f'"{name}"' for name in fields)
    marks = ", ".join("?" for _ in fields)
    db.execute(f"INSERT INTO renders ({cols}) VALUES ({marks})", tuple(fields.values()))
    row = db.one("SELECT * FROM renders WHERE id = ?", (render_id,))
    assert row is not None
    return _render(row)
