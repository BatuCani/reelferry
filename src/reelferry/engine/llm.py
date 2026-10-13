"""OpenRouter: one key for scene selection and cloud transcription."""

from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx

from reelferry import __version__
from reelferry.errors import ErrorCode, ReelferryError

OPENROUTER_BASE = "https://openrouter.ai/api/v1"
RETRY_STATUS = {408, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class ChatResult:
    data: dict[str, Any]
    model: str
    cost_usd: float


class ChatClient(Protocol):
    def json_chat(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema_name: str,
        schema: dict[str, Any],
    ) -> ChatResult: ...


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    if not text.startswith("{"):
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            text = text[start : end + 1]
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ReelferryError(
            ErrorCode.UPSTREAM_ERROR,
            "the model did not return valid JSON",
            details={"excerpt": text[:500]},
        ) from exc
    if not isinstance(value, dict):
        raise ReelferryError(
            ErrorCode.UPSTREAM_ERROR, "the model returned JSON that is not an object"
        )
    return value


class OpenRouterClient:
    def __init__(self, api_key: str, *, transport: httpx.BaseTransport | None = None) -> None:
        self._client = httpx.Client(
            base_url=OPENROUTER_BASE,
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://github.com/BatuCani/reelferry",
                "X-Title": "Reelferry",
                "User-Agent": f"reelferry/{__version__}",
            },
            timeout=httpx.Timeout(600.0, connect=20.0),
            transport=transport,
        )

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        delay = 2.0
        for attempt in range(4):
            try:
                response = self._client.post(path, json=body)
            except httpx.TransportError as exc:
                if attempt == 3:
                    raise ReelferryError(
                        ErrorCode.UPSTREAM_ERROR, f"cannot reach OpenRouter: {exc}"
                    ) from exc
            else:
                if response.status_code == 200:
                    data: dict[str, Any] = response.json()
                    if "error" in data and not data.get("choices") and "text" not in data:
                        raise ReelferryError(
                            ErrorCode.UPSTREAM_ERROR,
                            f"OpenRouter error: {data['error']}",
                        )
                    return data
                if response.status_code in (401, 403):
                    raise ReelferryError(
                        ErrorCode.MISSING_API_KEY,
                        f"OpenRouter rejected the API key ({response.status_code})",
                        hint="Check the key with `reelferry system check`.",
                    )
                if response.status_code not in RETRY_STATUS or attempt == 3:
                    raise ReelferryError(
                        ErrorCode.UPSTREAM_ERROR,
                        f"OpenRouter returned {response.status_code}",
                        details={"body": response.text[:1000]},
                    )
            time.sleep(delay)
            delay *= 2
        raise AssertionError("unreachable")

    def json_chat(
        self,
        *,
        model: str,
        system: str,
        user: str,
        schema_name: str,
        schema: dict[str, Any],
    ) -> ChatResult:
        body: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema},
            },
            "usage": {"include": True},
        }
        try:
            data = self._post("/chat/completions", body)
        except ReelferryError as exc:
            body_text = str(exc.details.get("body", ""))
            if "response_format" not in body_text and "json_schema" not in body_text:
                raise
            body["response_format"] = {"type": "json_object"}
            data = self._post("/chat/completions", body)
        choices = data.get("choices") or []
        if not choices:
            raise ReelferryError(ErrorCode.UPSTREAM_ERROR, "OpenRouter returned no choices")
        content = choices[0].get("message", {}).get("content") or ""
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        usage = data.get("usage") or {}
        return ChatResult(
            data=_extract_json(str(content)),
            model=str(data.get("model", model)),
            cost_usd=float(usage.get("cost") or 0.0),
        )

    def transcribe(
        self, *, model: str, audio: Path, audio_format: str, language: str | None
    ) -> tuple[dict[str, Any], float]:
        body: dict[str, Any] = {
            "model": model,
            "input_audio": {
                "data": base64.b64encode(audio.read_bytes()).decode("ascii"),
                "format": audio_format,
            },
            "response_format": "verbose_json",
            "timestamp_granularities": ["word"],
        }
        if language:
            body["language"] = language
        data = self._post("/audio/transcriptions", body)
        usage = data.get("usage") or {}
        return data, float(usage.get("cost") or 0.0)
