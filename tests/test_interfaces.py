"""HTTP API, MCP and CLI all drive the same operations."""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import uvicorn
from click.testing import CliRunner
from fastapi.testclient import TestClient

from reelferry import jobs
from reelferry.app import App
from reelferry.cli import main
from reelferry.interfaces.http_api import create_api
from reelferry.interfaces.mcp_server import build_server
from reelferry.jobs import JobContext
from reelferry.models import JobKind

from .conftest import HAS_FFMPEG

TOKEN = "rf_test_token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def client(app: App) -> Iterator[TestClient]:
    api = create_api(app, token=TOKEN, run_worker=True)
    with TestClient(api) as test_client:
        yield test_client


def test_api_requires_token(client: TestClient) -> None:
    assert client.get("/v1/clips").status_code == 401
    assert client.get("/v1/clips", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/v1/clips", headers=AUTH).json() == {"clips": []}


def test_public_endpoints(client: TestClient) -> None:
    assert client.get("/v1/health").json()["status"] == "ok"
    schema = client.get("/openapi.json").json()
    assert schema["components"]["securitySchemes"]["bearer"]["scheme"] == "bearer"
    assert "/v1/clips/{clip_id}/render" in schema["paths"]


def test_api_errors_have_codes(client: TestClient) -> None:
    missing = client.get("/v1/clips/clip_nope", headers=AUTH)
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "not_found"
    bad = client.post("/v1/clips", headers=AUTH, json={"start": "soon"})
    assert bad.status_code == 422
    assert bad.json()["error"]["code"] == "invalid_input"
    empty = client.post("/v1/clips", headers=AUTH, json={})
    assert empty.status_code == 422


def test_api_settings_roundtrip(client: TestClient) -> None:
    out = client.patch("/v1/settings", headers=AUTH, json={"default_caption_style": "neon"})
    assert out.status_code == 200, out.text
    assert out.json()["settings"]["default_caption_style"] == "neon"
    assert out.json()["openrouter_key"] is None
    bad = client.patch("/v1/settings", headers=AUTH, json={"default_caption_style": "nope"})
    assert bad.status_code == 422
    reset = client.patch("/v1/settings", headers=AUTH, json={"reset": ["default_caption_style"]})
    assert reset.json()["settings"]["default_caption_style"] == "pop"


@pytest.mark.ffmpeg
def test_api_runs_jobs_in_the_service(client: TestClient, talk_video: Path) -> None:
    added = client.post("/v1/sources", headers=AUTH, json={"uri": str(talk_video)})
    assert added.status_code == 202, added.text
    source_id = added.json()["source"]["id"]
    started = client.post(f"/v1/sources/{source_id}/transcript", headers=AUTH, json={})
    assert started.status_code == 202, started.text
    job_id = started.json()["job"]["id"]
    waited = client.post(f"/v1/jobs/{job_id}/wait", headers=AUTH, json={"timeout": 60}).json()
    assert waited["done"] and waited["job"]["status"] == "succeeded", waited
    transcript = client.get(
        f"/v1/sources/{source_id}/transcript", headers=AUTH, params={"format": "text"}
    ).json()
    assert transcript["text"].startswith("Welcome back")


def _mcp_call(app: App, calls: list[tuple[str, dict[str, Any]]]) -> list[Any]:
    from mcp import Client

    async def run() -> list[Any]:
        results = []
        async with Client(build_server(app)) as mcp_client:
            for name, args in calls:
                results.append(await mcp_client.call_tool(name, args))
        return results

    return asyncio.run(run())


def test_mcp_tools_return_structured_content(app: App) -> None:
    styles, bad, missing = _mcp_call(
        app,
        [
            ("list_caption_styles", {}),
            ("create_clip", {"start": "later"}),
            ("get_clip", {"clip_id": "clip_nope"}),
        ],
    )
    assert not styles.is_error
    assert {s["id"] for s in styles.structured_content["styles"]} >= {"pop", "karaoke"}
    assert bad.is_error
    assert bad.structured_content["error"]["code"] == "invalid_input"
    assert missing.is_error
    assert missing.structured_content["error"]["code"] == "not_found"


@pytest.mark.ffmpeg
def test_mcp_preview_returns_an_image(app: App, talk_video: Path) -> None:
    (added,) = _mcp_call(app, [("add_source", {"uri": str(talk_video)})])
    source_id = added.structured_content["source"]["id"]
    (clip,) = _mcp_call(
        app, [("create_clip", {"source_id": source_id, "start": 1, "end": 5, "format": "16:9"})]
    )
    (preview,) = _mcp_call(
        app, [("preview_clip", {"clip_id": clip.structured_content["clip"]["id"], "frames": 2})]
    )
    assert not preview.is_error
    kinds = [block.type for block in preview.content]
    assert kinds == ["text", "image"]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_mcp_over_http_behind_token(app: App) -> None:
    port = _free_port()
    api = create_api(app, token=TOKEN, run_worker=False)
    server = uvicorn.Server(uvicorn.Config(api, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started and time.monotonic() < deadline:
            time.sleep(0.05)
        url = f"http://127.0.0.1:{port}/mcp"
        init = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0"},
            },
        }
        headers = {"Accept": "application/json, text/event-stream"}
        assert httpx.post(url, json=init, headers=headers).status_code == 401
        ok = httpx.post(url, json=init, headers={**headers, **AUTH})
        assert ok.status_code == 200, ok.text
        assert "reelferry" in ok.text
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def test_cli_json_output_and_errors(tmp_path: Path) -> None:
    runner = CliRunner()
    styles = runner.invoke(main, ["caption", "styles", "--json"])
    assert styles.exit_code == 0, styles.output
    assert json.loads(styles.output)["default"] == "pop"

    missing = runner.invoke(main, ["source", "add", str(tmp_path / "nope.mp4"), "--json"])
    assert missing.exit_code == 1
    assert json.loads(missing.stderr)["error"]["code"] == "not_found"

    invalid = runner.invoke(main, ["clip", "create", "--start", "1", "--json"])
    assert invalid.exit_code == 2

    via_input = runner.invoke(main, ["settings", "update", "--input", '{"max_parallel_jobs": 2}'])
    assert via_input.exit_code == 0, via_input.output
    assert "max_parallel_jobs: 2" in via_input.output


@pytest.mark.timeout(60)
def test_cli_wait_takes_over_when_the_worker_dies(
    app: App, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(app: App, ctx: JobContext) -> dict[str, Any]:
        return {"ok": True}

    monkeypatch.setitem(jobs.HANDLERS, JobKind.SUGGEST_CLIPS, handler)
    job = app.jobs.enqueue(JobKind.SUGGEST_CLIPS, {})
    app.jobs.register_worker("wrk_doomed")
    assert app.jobs.claim("wrk_doomed") == job.id

    def worker_dies() -> None:
        time.sleep(1.5)
        app.db.execute("UPDATE workers SET heartbeat_at = 0 WHERE id = 'wrk_doomed'")
        app.db.execute("UPDATE jobs SET lease_expires_at = 0 WHERE id = ?", (job.id,))
        app.db.close()

    killer = threading.Thread(target=worker_dies)
    killer.start()
    out = CliRunner().invoke(main, ["job", "get", job.id, "--wait", "--json"])
    killer.join()
    assert out.exit_code == 0, out.output
    assert json.loads(out.output)["job"]["status"] == "succeeded"


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")
def test_cli_wait_runs_the_job_inline(talk_video: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from reelferry import app as app_module

    from .conftest import FakeTranscriber, script_words

    original = app_module.App.__init__

    def patched(self: App, paths: Any = None) -> None:
        original(self, paths)
        self.transcriber_factory = lambda app, engine: FakeTranscriber(script_words())

    monkeypatch.setattr(app_module.App, "__init__", patched)
    runner = CliRunner()
    added = json.loads(runner.invoke(main, ["source", "add", str(talk_video), "--json"]).output)
    source_id = added["source"]["id"]
    out = runner.invoke(main, ["transcript", "create", source_id, "--wait", "--json"])
    assert out.exit_code == 0, out.output
    assert json.loads(out.output)["job"]["status"] == "succeeded"
    text = runner.invoke(main, ["transcript", "get", source_id, "--format", "text", "--json"])
    assert json.loads(text.output)["text"].startswith("Welcome back")
