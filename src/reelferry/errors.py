"""Typed errors shared by every interface.

Each error carries a stable machine-readable code so agents can branch on it,
and each code maps to one HTTP status for the local API.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    INVALID_INPUT = "invalid_input"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    MISSING_DEPENDENCY = "missing_dependency"
    MISSING_API_KEY = "missing_api_key"
    UPSTREAM_ERROR = "upstream_error"
    MEDIA_ERROR = "media_error"
    TRANSCRIPT_MISSING = "transcript_missing"
    JOB_FAILED = "job_failed"
    CANCELED = "canceled"
    UNAUTHORIZED = "unauthorized"
    INTERNAL = "internal"


HTTP_STATUS: dict[ErrorCode, int] = {
    ErrorCode.INVALID_INPUT: 422,
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.CONFLICT: 409,
    ErrorCode.MISSING_DEPENDENCY: 424,
    ErrorCode.MISSING_API_KEY: 424,
    ErrorCode.UPSTREAM_ERROR: 502,
    ErrorCode.MEDIA_ERROR: 422,
    ErrorCode.TRANSCRIPT_MISSING: 409,
    ErrorCode.JOB_FAILED: 409,
    ErrorCode.CANCELED: 409,
    ErrorCode.UNAUTHORIZED: 401,
    ErrorCode.INTERNAL: 500,
}


class ReelferryError(Exception):
    """An expected failure with a stable code, a human message and optional details."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        hint: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint
        self.details = details or {}

    @property
    def http_status(self) -> int:
        return HTTP_STATUS[self.code]

    def to_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {"code": self.code.value, "message": self.message}
        if self.hint:
            body["hint"] = self.hint
        if self.details:
            body["details"] = self.details
        return body


def not_found(kind: str, ident: str) -> ReelferryError:
    return ReelferryError(ErrorCode.NOT_FOUND, f"{kind} {ident!r} does not exist")


class JobCanceled(Exception):
    """Raised inside a running job when cancellation was requested."""
