"""Command line interface. Area/verb commands are generated from the operation registry."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import sys
import types as pytypes
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin

import click
from pydantic import ValidationError
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from reelferry import __version__
from reelferry.errors import ErrorCode, ReelferryError
from reelferry.registry import Operation, load_operations

EXIT_ERROR = 1
EXIT_INVALID = 2
EXIT_JOB_FAILED = 3

AREA_HELP = {
    "source": "Long videos to cut clips from.",
    "transcript": "Word-level transcripts.",
    "clip": "Suggest, create, render and preview clips.",
    "caption": "Caption styles and caption files.",
    "job": "Background jobs.",
    "system": "Health checks.",
    "settings": "User settings.",
}


def _unwrap_optional(annotation: Any) -> Any:
    origin = get_origin(annotation)
    if origin in (Union, pytypes.UnionType):
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _click_type(annotation: Any) -> tuple[Any, bool, bool]:
    """(click type, is_multiple, is_json) for a pydantic field annotation."""
    annotation = _unwrap_optional(annotation)
    origin = get_origin(annotation)
    if origin is list:
        (item,) = get_args(annotation) or (str,)
        if item in (str, int, float):
            return item, True, False
        return str, False, True
    if origin is Literal:
        return click.Choice([str(a) for a in get_args(annotation)]), False, False
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return click.Choice([str(m.value) for m in annotation]), False, False
    if annotation in (str, int, float, bool):
        return annotation, False, False
    return str, False, True


def _option_name(name: str) -> str:
    return "--" + name.replace("_", "-")


def _params_for(op: Operation) -> tuple[list[click.Parameter], set[str]]:
    params: list[click.Parameter] = []
    json_fields: set[str] = set()
    fields: dict[str, FieldInfo] = op.input_model.model_fields
    for name in op.positional:
        info = fields[name]
        ctype, _, _ = _click_type(info.annotation)
        params.append(click.Argument([name], type=ctype, required=info.is_required()))
    for name, info in fields.items():
        if name in op.positional:
            continue
        ctype, multiple, is_json = _click_type(info.annotation)
        help_text = info.description or ""
        if is_json:
            json_fields.add(name)
            help_text = (help_text + " (JSON)").strip()
        if not info.is_required() and info.default not in (None, [], (), PydanticUndefined):
            default = info.default
            if isinstance(default, Enum):
                default = default.value
            help_text += f" [default: {default}]"
        if ctype is bool:
            params.append(
                click.Option(
                    [f"{_option_name(name)}/--no-{name.replace('_', '-')}"],
                    default=None,
                    help=help_text,
                )
            )
            continue
        params.append(
            click.Option(
                [_option_name(name)],
                type=ctype,
                multiple=multiple,
                required=False,
                help=help_text,
            )
        )
    return params, json_fields


def _human_lines(value: Any, indent: int = 0) -> list[str]:
    """Readable YAML-like rendering for terminal output."""
    pad = "  " * indent
    lines: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if item is None or item in ([], {}):
                continue
            if isinstance(item, dict | list):
                lines.append(f"{pad}{key}:")
                lines.extend(_human_lines(item, indent + 1))
            else:
                lines.append(f"{pad}{key}: {item}")
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict | list):
                nested = _human_lines(item, indent + 1)
                if nested:
                    lines.append(f"{pad}- {nested[0].lstrip()}")
                    lines.extend(nested[1:])
            else:
                lines.append(f"{pad}- {item}")
    else:
        lines.append(f"{pad}{value}")
    return lines


def _print_human(value: Any) -> None:
    click.echo("\n".join(_human_lines(value)))


def _fail(exc: ReelferryError, as_json: bool) -> None:
    if as_json:
        click.echo(json.dumps({"error": exc.to_dict()}, ensure_ascii=False), err=True)
    else:
        click.secho(f"error: {exc.message}", fg="red", err=True)
        if exc.hint:
            click.echo(f"hint: {exc.hint}", err=True)
        for key, value in exc.details.items():
            if key == "errors" and isinstance(value, list):
                for err in value:
                    loc = ".".join(str(p) for p in err.get("loc", []))
                    click.echo(f"  {loc}: {err.get('msg')}", err=True)
    code = EXIT_INVALID if exc.code is ErrorCode.INVALID_INPUT else EXIT_ERROR
    sys.exit(code)


def _keep_worker(app: Any, job_id: str, worker: Any) -> Any:
    """Make sure a worker serves the job: a detached one, else one in this process."""
    from reelferry.jobs import Worker, ensure_worker

    if worker is not None:
        return worker
    ensure_worker(app)
    if app.jobs.live_workers() > 0:
        return None
    worker = Worker(app, concurrency=1, only_job_id=job_id)
    worker.start()
    return worker


def _wait_for(app: Any, job_id: str, as_json: bool) -> dict[str, Any]:
    from reelferry.jobs import stop_on_signals

    worker = None
    last = ""
    with stop_on_signals():
        try:
            while True:
                worker = _keep_worker(app, job_id, worker)
                job = app.jobs.wait(job_id, 1.0)
                if not as_json:
                    line = f"{job.status.value} {job.progress * 100:5.1f}%  {job.message or ''}"
                    if line != last and sys.stderr.isatty():
                        click.echo("\r\033[2K" + line, nl=False, err=True)
                    last = line
                if job.status.terminal:
                    break
        except KeyboardInterrupt:
            click.echo("", err=True)
            if worker is None:
                click.echo(
                    f"stopped waiting; job {job_id} keeps running in the background", err=True
                )
            else:
                click.echo(
                    f"stopped job {job_id}; it resumes when a worker runs (`reelferry worker`)",
                    err=True,
                )
            sys.exit(130)
        finally:
            if worker is not None:
                worker.stop()
            if not as_json and sys.stderr.isatty():
                click.echo("", err=True)
    return dict(job.model_dump(mode="json"))


def _refresh(app: Any, data: dict[str, Any]) -> None:
    """After waiting, show the records the job changed in their final state."""
    from reelferry import store

    getters = {"source": store.get_source, "clip": store.get_clip}
    for key, getter in getters.items():
        record = data.get(key)
        if isinstance(record, dict) and "id" in record:
            data[key] = getter(app.db, record["id"]).model_dump(mode="json")


def _make_command(op: Operation) -> click.Command:
    params, json_fields = _params_for(op)
    params.append(
        click.Option(
            ["--input", "input_json"],
            help="Whole input as JSON, '@file.json' or '-' for stdin. Options override it.",
        )
    )
    params.append(click.Option(["--json", "as_json"], is_flag=True, help="Print JSON output."))
    params.append(
        click.Option(
            ["--wait"],
            is_flag=True,
            help="Wait for the background job this command starts (no-op if it starts none).",
        )
    )

    def callback(input_json: str | None, as_json: bool, wait: bool, **kwargs: Any) -> None:
        from reelferry.app import App

        payload: dict[str, Any] = {}
        try:
            if input_json:
                if input_json == "-":
                    raw = sys.stdin.read()
                elif input_json.startswith("@"):
                    raw = Path(input_json[1:]).expanduser().read_text(encoding="utf-8")
                else:
                    raw = input_json
                loaded = json.loads(raw)
                if not isinstance(loaded, dict):
                    raise ReelferryError(ErrorCode.INVALID_INPUT, "--input must be a JSON object")
                payload.update(loaded)
            for name, given in kwargs.items():
                if given is None or given == ():
                    continue
                if name in json_fields:
                    payload[name] = json.loads(given)
                elif isinstance(given, tuple):
                    payload[name] = list(given)
                else:
                    payload[name] = given
        except (json.JSONDecodeError, OSError) as exc:
            _fail(ReelferryError(ErrorCode.INVALID_INPUT, f"bad JSON input: {exc}"), as_json)

        app = App()
        try:
            result = op.run(app, payload)
        except ValidationError as exc:
            _fail(
                ReelferryError(
                    ErrorCode.INVALID_INPUT,
                    "invalid input",
                    details={"errors": json.loads(exc.json(include_url=False))},
                ),
                as_json,
            )
            return
        except ReelferryError as exc:
            _fail(exc, as_json)
            return
        data: dict[str, Any] = result.model_dump(mode="json")
        exit_code = 0
        job = data.get("job")
        if (
            wait
            and isinstance(job, dict)
            and job.get("status") not in ("succeeded", "failed", "canceled")
        ):
            data["job"] = _wait_for(app, job["id"], as_json)
            _refresh(app, data)
        final = data.get("job")
        if wait and isinstance(final, dict) and final.get("status") in ("failed", "canceled"):
            exit_code = EXIT_JOB_FAILED
        if as_json:
            click.echo(json.dumps(data, ensure_ascii=False, indent=2))
        else:
            _print_human(data)
        if exit_code:
            sys.exit(exit_code)

    return click.Command(
        op.verb,
        params=params,
        callback=callback,
        help=op.description,
        short_help=op.summary,
    )


class _Root(click.Group):
    def list_commands(self, _ctx: click.Context) -> list[str]:
        return list(self.commands)


@click.group(cls=_Root, context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, "-V", "--version", prog_name="reelferry")
def main() -> None:
    """Reelferry: turn long videos into captioned short clips, driven by AI agents.

    Every command mirrors an MCP tool and a local API route. Add --json for
    machine-readable output and --wait to wait for background jobs.
    """


def _install_generated(root: click.Group) -> None:
    groups: dict[str, click.Group] = {}
    for op in load_operations().values():
        group = groups.get(op.area)
        if group is None:
            group = _Root(op.area, help=AREA_HELP.get(op.area))
            groups[op.area] = group
            root.add_command(group)
        group.add_command(_make_command(op))


@main.command()
def mcp() -> None:
    """Run the MCP server over stdio (for Claude Code, Codex, Cursor, ...)."""
    import anyio

    from reelferry.app import App
    from reelferry.interfaces.mcp_server import run_stdio
    from reelferry.jobs import Worker

    app = App()
    worker = Worker(app)

    def shutdown(signum: int, _frame: Any) -> None:
        worker.stop(timeout=0)
        os._exit(128 + signum)

    for name in ("SIGINT", "SIGTERM", "SIGHUP"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), shutdown)
    worker.start()
    try:
        anyio.run(run_stdio, app)
    finally:
        worker.stop()


@main.command()
@click.option("--host", default="127.0.0.1", show_default=True, help="Interface to bind.")
@click.option("--port", default=7878, show_default=True, type=int)
@click.option("--no-worker", is_flag=True, help="Do not run background jobs in this process.")
def serve(host: str, port: int, no_worker: bool) -> None:
    """Run the local API (with MCP over HTTP at /mcp) and a job worker."""
    import uvicorn

    from reelferry.app import App
    from reelferry.config import load_or_create_token
    from reelferry.interfaces.http_api import create_api
    from reelferry.jobs import stop_on_signals

    app = App()
    token = load_or_create_token(app.paths)
    api = create_api(app, token=token, host=host, run_worker=not no_worker)
    click.echo(f"Reelferry {__version__} on http://{host}:{port}", err=True)
    click.echo(f"  API docs: http://{host}:{port}/docs   MCP: http://{host}:{port}/mcp", err=True)
    click.echo("  Token: `reelferry token`", err=True)
    with stop_on_signals(), contextlib.suppress(KeyboardInterrupt):
        uvicorn.run(api, host=host, port=port, log_level="warning")


@main.command()
@click.option("--exit-when-idle", is_flag=True, help="Stop once no jobs are left.")
@click.option("--id", "worker_id", hidden=True, help="Worker id reserved by the starter.")
def worker(exit_when_idle: bool, worker_id: str | None) -> None:
    """Run background jobs (started automatically when needed)."""
    import logging

    from reelferry.app import App
    from reelferry.jobs import Worker, stop_on_signals

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with stop_on_signals(), contextlib.suppress(KeyboardInterrupt):
        Worker(App(), worker_id=worker_id, exit_when_idle=exit_when_idle).run_until_idle()


@main.command()
def token() -> None:
    """Print the bearer token for the local API and MCP over HTTP."""
    from reelferry.config import Paths, load_or_create_token

    click.echo(load_or_create_token(Paths.default()))


@main.group()
def config() -> None:
    """Secrets that are not settings (stored in the system keychain)."""


@config.command("set-key")
@click.option("--stdin", "from_stdin", is_flag=True, help="Read the key from stdin.")
def set_key(from_stdin: bool) -> None:
    """Store the OpenRouter API key in the system keychain."""
    from reelferry import credentials

    value = (
        sys.stdin.readline() if from_stdin else click.prompt("OpenRouter API key", hide_input=True)
    )
    value = value.strip()
    if not value:
        raise click.UsageError("empty key")
    try:
        credentials.store_openrouter_key(value)
    except Exception as exc:
        raise click.ClickException(
            f"could not store the key in the keychain ({exc}). "
            "Set the OPENROUTER_API_KEY environment variable instead."
        ) from exc
    click.echo("OpenRouter key stored in the system keychain.")


@config.command("clear-key")
def clear_key() -> None:
    """Remove the OpenRouter API key from the system keychain."""
    from reelferry import credentials

    removed = credentials.delete_openrouter_key()
    click.echo("Key removed." if removed else "No key was stored.")


_install_generated(main)
