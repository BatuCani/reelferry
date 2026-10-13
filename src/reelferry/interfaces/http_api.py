"""Local HTTP API generated from the operation registry, with MCP mounted at /mcp."""

from __future__ import annotations

import hmac
import inspect
from collections.abc import AsyncIterator, Awaitable, Callable, MutableMapping
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Body, FastAPI, Path, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, ValidationError, create_model

from reelferry import __version__
from reelferry.app import App
from reelferry.errors import HTTP_STATUS, ErrorCode, ReelferryError
from reelferry.registry import Operation, load_operations

PUBLIC_PATHS = {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/v1/health"}
Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


class ErrorDetail(BaseModel):
    code: ErrorCode
    message: str
    hint: str | None = None
    details: dict[str, Any] | None = None


class ErrorBody(BaseModel):
    error: ErrorDetail


class TokenAuth:
    """Require ``Authorization: Bearer <token>`` on everything except public paths."""

    def __init__(self, app: ASGIApp, token: str) -> None:
        self.app = app
        self.token = token.encode()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] in PUBLIC_PATHS:
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers") or [])
        auth = headers.get(b"authorization", b"")
        scheme, _, value = auth.partition(b" ")
        if scheme.lower() != b"bearer" or not hmac.compare_digest(value.strip(), self.token):
            response = JSONResponse(
                {
                    "error": {
                        "code": ErrorCode.UNAUTHORIZED.value,
                        "message": "missing or wrong bearer token",
                        "hint": "Send 'Authorization: Bearer <token>'. "
                        "`reelferry token` prints the local token.",
                    }
                },
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


def _without(model: type[BaseModel], drop: list[str], suffix: str) -> type[BaseModel]:
    fields: dict[str, Any] = {
        name: (info.annotation, info)
        for name, info in model.model_fields.items()
        if name not in drop
    }
    return create_model(
        model.__name__ + suffix,
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )


def _endpoint(app: App, op: Operation) -> Callable[..., Any]:
    path_params = op.path_params
    params: list[inspect.Parameter] = [
        inspect.Parameter(
            name,
            inspect.Parameter.KEYWORD_ONLY,
            annotation=Annotated[str, Path(description=f"The {name.replace('_', ' ')}.")],
        )
        for name in path_params
    ]
    rest = [n for n in op.input_model.model_fields if n not in path_params]
    if op.method in ("GET", "DELETE"):
        query_fields = {
            name: info
            for name, info in op.input_model.model_fields.items()
            if name not in path_params
        }
        for name, info in query_fields.items():
            params.append(
                inspect.Parameter(
                    name,
                    inspect.Parameter.KEYWORD_ONLY,
                    default=info.get_default(call_default_factory=True),
                    annotation=Annotated[info.annotation, Query(description=info.description)],
                )
            )
    elif rest:
        body_model = _without(op.input_model, path_params, "Body")
        required = any(f.is_required() for f in body_model.model_fields.values())
        params.append(
            inspect.Parameter(
                "body",
                inspect.Parameter.KEYWORD_ONLY,
                default=inspect.Parameter.empty if required else None,
                annotation=Annotated[body_model if required else body_model | None, Body()],
            )
        )

    def endpoint(**kwargs: Any) -> Any:
        payload: dict[str, Any] = {}
        body = kwargs.pop("body", None)
        if isinstance(body, BaseModel):
            payload.update(body.model_dump(exclude_unset=True))
        payload.update(kwargs)
        try:
            return op.run(app, payload)
        except ValidationError as exc:
            raise RequestValidationError(exc.errors()) from exc

    endpoint.__signature__ = inspect.Signature(params)  # type: ignore[attr-defined]
    endpoint.__name__ = op.name
    endpoint.__doc__ = op.description
    return endpoint


def create_api(
    app: App,
    *,
    token: str,
    host: str = "127.0.0.1",
    run_worker: bool = True,
    mount_mcp: bool = True,
) -> FastAPI:
    operations = load_operations()
    mcp_server = None
    mcp_app = None
    if mount_mcp:
        from mcp.server.transport_security import TransportSecuritySettings

        from reelferry.interfaces.mcp_server import build_server

        mcp_server = build_server(app)
        loopback = host in ("127.0.0.1", "localhost", "::1")
        security = (
            None if loopback else TransportSecuritySettings(enable_dns_rebinding_protection=False)
        )
        mcp_app = mcp_server.streamable_http_app(
            streamable_http_path="/mcp", host=host, transport_security=security
        )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        from reelferry.jobs import Worker

        worker = Worker(app) if run_worker else None
        if worker:
            worker.start()
        try:
            if mcp_server is not None:
                async with mcp_server.session_manager.run():
                    yield
            else:
                yield
        finally:
            if worker:
                worker.stop()

    api = FastAPI(
        title="Reelferry",
        version=__version__,
        description=(
            "Local API of Reelferry. Every route mirrors an MCP tool and a CLI command. "
            "Authenticate with `Authorization: Bearer <token>` (`reelferry token`)."
        ),
        lifespan=lifespan,
        redoc_url=None,
    )

    @api.exception_handler(ReelferryError)
    async def reelferry_error(_: Request, exc: ReelferryError) -> JSONResponse:
        return JSONResponse({"error": exc.to_dict()}, status_code=exc.http_status)

    @api.exception_handler(RequestValidationError)
    async def validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        errors = [{k: v for k, v in e.items() if k in ("loc", "msg", "type")} for e in exc.errors()]
        return JSONResponse(
            {
                "error": {
                    "code": ErrorCode.INVALID_INPUT.value,
                    "message": "invalid request",
                    "details": {"errors": errors},
                }
            },
            status_code=422,
        )

    @api.get("/v1/health", tags=["system"], summary="Liveness probe (no token needed)")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    for op in operations.values():
        responses: dict[int | str, dict[str, Any]] = {}
        for code in op.errors:
            status = HTTP_STATUS[code]
            entry = responses.setdefault(status, {"model": ErrorBody, "description": ""})
            entry["description"] = ", ".join(filter(None, [entry["description"], code.value]))
        api.add_api_route(
            op.path,
            _endpoint(app, op),
            methods=[op.method],
            response_model=op.output_model,
            status_code=202 if op.returns_job else 200,
            summary=op.summary,
            description=op.description,
            operation_id=op.name,
            tags=[op.area],
            responses=responses,
        )

    def openapi() -> dict[str, Any]:
        if api.openapi_schema:
            return api.openapi_schema
        from fastapi.openapi.utils import get_openapi

        schema = get_openapi(
            title=api.title, version=api.version, description=api.description, routes=api.routes
        )
        schema.setdefault("components", {})["securitySchemes"] = {
            "bearer": {"type": "http", "scheme": "bearer"}
        }
        schema["security"] = [{"bearer": []}]
        api.openapi_schema = schema
        return schema

    api.openapi = openapi  # type: ignore[method-assign]
    if mcp_app is not None:
        api.mount("/", mcp_app)
    api.add_middleware(TokenAuth, token=token)
    return api
