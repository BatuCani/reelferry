"""The single source of truth for every operation.

Each operation is described once: name, input model, output model, possible
errors and a handler. The MCP server, the CLI and the local HTTP API are all
generated from this registry, so the three interfaces cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from reelferry.errors import ErrorCode

if TYPE_CHECKING:
    from reelferry.app import App

HttpMethod = Literal["GET", "POST", "PATCH", "DELETE"]
BASE_ERRORS = (ErrorCode.INVALID_INPUT, ErrorCode.INTERNAL)


@dataclass(frozen=True)
class Operation:
    name: str
    area: str
    verb: str
    summary: str
    description: str
    input_model: type[BaseModel]
    output_model: type[BaseModel]
    handler: Callable[[App, Any], BaseModel]
    method: HttpMethod
    path: str
    errors: tuple[ErrorCode, ...]
    positional: tuple[str, ...]
    returns_job: bool
    read_only: bool
    destructive: bool
    image_field: str | None

    @property
    def path_params(self) -> list[str]:
        import re

        return re.findall(r"{(\w+)}", self.path)

    def run(self, app: App, payload: dict[str, Any]) -> BaseModel:
        """Validate raw input, call the handler, validate the output."""
        data = self.input_model.model_validate(payload)
        result = self.handler(app, data)
        return self.output_model.model_validate(result.model_dump())


REGISTRY: dict[str, Operation] = {}


def operation(
    *,
    name: str,
    area: str,
    verb: str,
    summary: str,
    input: type[BaseModel],
    output: type[BaseModel],
    method: HttpMethod,
    path: str,
    errors: tuple[ErrorCode, ...] = (),
    positional: tuple[str, ...] = (),
    returns_job: bool = False,
    read_only: bool = False,
    destructive: bool = False,
    image_field: str | None = None,
) -> Callable[[Callable[[App, Any], BaseModel]], Callable[[App, Any], BaseModel]]:
    def register(fn: Callable[[App, Any], BaseModel]) -> Callable[[App, Any], BaseModel]:
        if name in REGISTRY:
            raise ValueError(f"operation {name!r} registered twice")
        doc = (fn.__doc__ or "").strip()
        REGISTRY[name] = Operation(
            name=name,
            area=area,
            verb=verb,
            summary=summary,
            description=doc or summary,
            input_model=input,
            output_model=output,
            handler=fn,
            method=method,
            path=path,
            errors=tuple(dict.fromkeys((*errors, *BASE_ERRORS))),
            positional=positional,
            returns_job=returns_job,
            read_only=read_only,
            destructive=destructive,
            image_field=image_field,
        )
        return fn

    return register


def load_operations() -> dict[str, Operation]:
    """Import every operation module so the registry is complete."""
    import importlib

    for module in ("sources", "transcripts", "clips", "captions", "jobs", "system"):
        importlib.import_module(f"reelferry.operations.{module}")
    return REGISTRY
