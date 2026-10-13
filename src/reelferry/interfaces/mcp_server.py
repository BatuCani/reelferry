"""MCP server generated from the operation registry (stdio and streamable HTTP)."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import anyio
import mcp_types as types
from mcp.server import Server, ServerRequestContext
from pydantic import ValidationError

from reelferry import __version__
from reelferry.app import App
from reelferry.errors import ErrorCode, ReelferryError
from reelferry.registry import Operation, load_operations

INSTRUCTIONS = """Reelferry turns long videos into captioned short clips, on this machine.

Typical flow:
1. add_source (local file path or public link). Links download in a job: wait_job.
2. transcribe -> wait_job. Local Whisper with exact word timings.
3. get_transcript to read it; edit_transcript to fix names before captions are made.
4. suggest_clips with a goal -> wait_job. The job result lists scored candidates.
5. create_clip from a candidate (or from start/end), update_clip to adjust.
6. render_clip -> wait_job. preview_clip returns a frame sheet so you can check framing.
7. export_captions for SRT/VTT/ASS/JSON files without rendering.

Long work returns a job right away. wait_job blocks up to `timeout` seconds; call it again
while done is false. Times are seconds from the start of the source. Errors come back with a
stable `code` and often a `hint`.
"""


def tool_for(op: Operation) -> types.Tool:
    return types.Tool(
        name=op.name,
        title=op.summary,
        description=op.description,
        input_schema=op.input_model.model_json_schema(),
        output_schema=op.output_model.model_json_schema(),
        annotations=types.ToolAnnotations(
            title=op.summary,
            read_only_hint=op.read_only,
            destructive_hint=op.destructive,
            idempotent_hint=op.read_only,
            open_world_hint=op.name in ("add_source", "suggest_clips"),
        ),
    )


def _error_result(error: dict[str, Any]) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps({"error": error}))],
        structured_content={"error": error},
        is_error=True,
    )


def call_operation(app: App, op: Operation, arguments: dict[str, Any]) -> types.CallToolResult:
    try:
        result = op.run(app, arguments)
    except ValidationError as exc:
        return _error_result(
            {
                "code": ErrorCode.INVALID_INPUT.value,
                "message": "invalid arguments",
                "details": {"errors": json.loads(exc.json(include_url=False))},
            }
        )
    except ReelferryError as exc:
        return _error_result(exc.to_dict())
    finally:
        app.db.close()
    data = result.model_dump(mode="json")
    content: list[types.ContentBlock] = [
        types.TextContent(type="text", text=json.dumps(data, ensure_ascii=False))
    ]
    if op.image_field:
        image = Path(str(data.get(op.image_field, "")))
        if image.is_file():
            mime = "image/png" if image.suffix == ".png" else "image/jpeg"
            content.append(
                types.ImageContent(
                    type="image",
                    data=base64.b64encode(image.read_bytes()).decode("ascii"),
                    mime_type=mime,
                )
            )
    return types.CallToolResult(content=content, structured_content=data)


def build_server(app: App) -> Server[Any]:
    operations = load_operations()
    tools = [tool_for(op) for op in operations.values()]

    async def list_tools(
        _ctx: ServerRequestContext[Any], _params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        return types.ListToolsResult(tools=tools)

    async def call_tool(
        _ctx: ServerRequestContext[Any], params: types.CallToolRequestParams
    ) -> types.CallToolResult:
        op = operations.get(params.name)
        if op is None:
            return _error_result(
                {"code": ErrorCode.NOT_FOUND.value, "message": f"unknown tool {params.name!r}"}
            )
        arguments = params.arguments or {}
        return await anyio.to_thread.run_sync(call_operation, app, op, arguments)

    return Server(
        "reelferry",
        version=__version__,
        title="Reelferry",
        description="Agent-first clip studio: long videos in, captioned shorts out.",
        instructions=INSTRUCTIONS,
        website_url="https://github.com/BatuCani/reelferry",
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )


async def run_stdio(app: App) -> None:
    from mcp.server.stdio import stdio_server

    server = build_server(app)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())
