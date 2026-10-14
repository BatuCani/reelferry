"""The three interfaces are generated from one registry and must stay in sync."""

from __future__ import annotations

import asyncio

import click
from fastapi.testclient import TestClient

from reelferry.app import App
from reelferry.cli import main
from reelferry.interfaces.http_api import create_api
from reelferry.interfaces.mcp_server import build_server
from reelferry.registry import load_operations

ENGINE_OPERATIONS = {
    "add_source",
    "transcribe",
    "get_transcript",
    "edit_transcript",
    "suggest_clips",
    "create_clip",
    "update_clip",
    "render_clip",
    "preview_clip",
    "list_clips",
    "get_clip",
    "list_caption_styles",
    "export_captions",
    "get_job",
    "wait_job",
    "cancel_job",
}


def test_registry_covers_the_engine_operations() -> None:
    assert set(load_operations()) >= ENGINE_OPERATIONS


def test_operations_are_well_formed() -> None:
    ops = load_operations()
    seen_cli: set[tuple[str, str]] = set()
    seen_http: set[tuple[str, str]] = set()
    for op in ops.values():
        assert (op.area, op.verb) not in seen_cli, op.name
        seen_cli.add((op.area, op.verb))
        assert (op.method, op.path) not in seen_http, op.name
        seen_http.add((op.method, op.path))
        for name in op.path_params:
            assert name in op.input_model.model_fields, (op.name, name)
        for name in op.positional:
            assert name in op.input_model.model_fields, (op.name, name)
        assert op.output_model.model_json_schema()["type"] == "object"
        assert op.description and op.summary
        if op.returns_job:
            assert "job" in op.output_model.model_fields, op.name


def test_cli_mirrors_registry() -> None:
    ops = load_operations()
    for op in ops.values():
        group = main.commands[op.area]
        assert isinstance(group, click.Group)
        command = group.commands[op.verb]
        names = {p.name for p in command.params}
        assert {"as_json", "wait", "input_json"} <= names
        assert set(op.input_model.model_fields) <= names, op.name


def test_mcp_mirrors_registry(app: App) -> None:
    from mcp import Client

    async def tools() -> set[str]:
        async with Client(build_server(app)) as client:
            listed = await client.list_tools()
            return {tool.name for tool in listed.tools}

    assert asyncio.run(tools()) == set(load_operations())


def test_openapi_mirrors_registry(app: App) -> None:
    api = create_api(app, token="t", run_worker=False, mount_mcp=False)
    schema = TestClient(api).get("/openapi.json").json()
    operation_ids = {
        spec["operationId"]
        for path in schema["paths"].values()
        for spec in path.values()
        if "operationId" in spec
    }
    assert set(load_operations()) <= operation_ids
