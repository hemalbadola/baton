"""Typer entry points for every `baton` command (PRD 15.1).

Bodies are deliberately unimplemented. Each lane fills in its own command by
calling into its module; the names, flags, defaults and environment variables
here are the contract the other lanes build against.

Configuration precedence (PRD 15.2): CLI flag > BATON_* env var >
~/.config/baton/config.toml > default. Typer resolves the first two via
`envvar=`; the TOML layer is read by `load_config` before a command body runs.
"""

from __future__ import annotations

import logging
import sys
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

__all__ = ["app", "main"]

# Defaults referenced by more than one command. Ports are fixed by PRD 15.3.
HEAD_HTTP_PORT = 7700
HEAD_CONTROL_PORT = 7711
DEFAULT_HEAD = f"http://127.0.0.1:{HEAD_HTTP_PORT}"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "baton"
DEFAULT_CONFIG = Path.home() / ".config" / "baton" / "config.toml"


class Objective(str, Enum):
    """Planner objective (PRD 9.4, 9.5). `latency` is the default per D7."""

    latency = "latency"
    balance = "balance"


class Quant(str, Enum):
    """Weight quantization applied at load (PRD 5.5, D8)."""

    none = "none"
    int8 = "int8"
    int4 = "int4"


class OutputFormat(str, Enum):
    table = "table"
    json = "json"


app = typer.Typer(
    name="baton",
    help="LAN-distributed LLM inference. One model, many laptops, one relay.",
    no_args_is_help=True,
    add_completion=False,
)

cache_app = typer.Typer(
    name="cache",
    help=f"Manage the local shard cache ({DEFAULT_CACHE_DIR}).",
    no_args_is_help=True,
)
app.add_typer(cache_app)


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    """Read ~/.config/baton/config.toml. Returns {} when the file is absent.

    The file is optional and is never required for a working cluster (PRD 15.2).
    """
    raise NotImplementedError("config: TOML layer of the precedence chain")


@app.callback()
def root(
    config: Annotated[
        Path | None,
        typer.Option("--config", envvar="BATON_CONFIG", help="Path to config.toml."),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", envvar="BATON_VERBOSE", help="Debug logging."),
    ] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="Errors only.")] = False,
) -> None:
    """Global options applied before any subcommand."""
    logging.basicConfig(
        level="DEBUG" if verbose else "ERROR" if quiet else "INFO",
        format="%(asctime)s %(name)s: %(message)s",
    )
    # One INFO line per HTTP request would bury the head's own output.
    for noisy in ("httpx", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _require_torch() -> None:
    """Fail with a plain message when PyTorch cannot load, instead of a traceback."""
    try:
        import torch  # noqa: F401
    except OSError as exc:  # a missing DLL on Windows
        hint = (
            "Install the Microsoft Visual C++ runtime from "
            "https://aka.ms/vs/17/release/vc_redist.x64.exe, then run baton again."
            if sys.platform == "win32"
            else "Reinstall Baton."
        )
        typer.echo(f"baton: PyTorch cannot load: {exc}\n{hint}", err=True)
        raise typer.Exit(1) from None


@app.command(name="app")
def open_app(
    name: Annotated[
        str | None,
        typer.Option("--name", envvar="BATON_NAME", help="Name shown to nearby laptops."),
    ] = None,
    port: Annotated[
        int, typer.Option("--port", envvar="BATON_APP_PORT", help="Port of the local page.")
    ] = 7800,
    browser: Annotated[
        bool, typer.Option("--browser/--no-browser", help="Open the page in the browser.")
    ] = True,
    update: Annotated[
        bool, typer.Option("--update/--no-update", help="Install a newer Baton before it starts.")
    ] = True,
) -> None:
    """Open the page: find nearby laptops, host a model or join one, chat. No other command."""
    import asyncio
    import contextlib

    _require_torch()
    if update:
        from baton import updater

        if updater.update_at_start():
            typer.echo("baton: a newer version exists. Updating, then Baton starts again.")
            raise typer.Exit(0)
    from baton.agent import run_agent

    with contextlib.suppress(KeyboardInterrupt, asyncio.CancelledError):
        asyncio.run(run_agent(name, port, browser))


@app.command()
def worker(
    name: Annotated[
        str | None,
        typer.Option(
            "--name", envvar="BATON_NAME", help="Advertised node name. Default: hostname."
        ),
    ] = None,
    head: Annotated[
        str | None,
        typer.Option(
            "--head",
            envvar="BATON_HEAD",
            help="Head control plane address. Default: discover over mDNS.",
        ),
    ] = None,
    data_port: Annotated[
        int,
        typer.Option(
            "--data-port",
            envvar="BATON_DATA_PORT",
            help="Pin the data plane port. 0 picks an ephemeral port.",
        ),
    ] = 0,
    cache_dir: Annotated[
        Path, typer.Option("--cache-dir", envvar="BATON_CACHE_DIR", help="Shard cache directory.")
    ] = DEFAULT_CACHE_DIR,
    backend: Annotated[
        str | None,
        typer.Option(
            "--backend", envvar="BATON_BACKEND", help="Force cuda|mps|cpu. Default: probe."
        ),
    ] = None,
    mem_budget: Annotated[
        float | None,
        typer.Option(
            "--mem-budget",
            envvar="BATON_MEM_BUDGET",
            help="Bytes of device memory this worker may use. Default: probed.",
        ),
    ] = None,
    no_mdns: Annotated[
        bool, typer.Option("--no-mdns", help="Skip mDNS discovery; --head is then required.")
    ] = False,
) -> None:
    """Join the cluster and serve a layer range (PRD 6.1)."""
    if no_mdns and head is None:
        raise typer.BadParameter("--no-mdns needs --head HOST[:PORT]")
    _require_torch()
    from baton.worker.daemon import WorkerConfig, run_worker

    config = WorkerConfig(
        head=head or "auto",
        max_mem_bytes=int(mem_budget) if mem_budget is not None else None,
        device=backend or "auto",
        name=name,
        data_port=data_port,
        cache_dir=cache_dir,
    )
    raise typer.Exit(run_worker(config))


@app.command()
def serve(
    model: Annotated[
        str, typer.Option("--model", envvar="BATON_MODEL", help="Hugging Face model id or path.")
    ],
    host: Annotated[
        str, typer.Option("--host", envvar="BATON_HOST", help="HTTP bind address.")
    ] = "0.0.0.0",
    port: Annotated[
        int, typer.Option("--port", envvar="BATON_PORT", help="HTTP API and dashboard port.")
    ] = HEAD_HTTP_PORT,
    control_port: Annotated[
        int,
        typer.Option("--control-port", envvar="BATON_CONTROL_PORT", help="Control plane port."),
    ] = HEAD_CONTROL_PORT,
    ctx: Annotated[
        int, typer.Option("--ctx", envvar="BATON_CTX", help="Guaranteed context length in tokens.")
    ] = 8192,
    quant: Annotated[
        Quant, typer.Option("--quant", envvar="BATON_QUANT", help="Weight quantization at load.")
    ] = Quant.none,  # int8 and int4 do not load yet (BAT-10)
    objective: Annotated[
        Objective, typer.Option("--objective", envvar="BATON_OBJECTIVE", help="Planner objective.")
    ] = Objective.latency,
    kv_fraction: Annotated[
        float | None,
        typer.Option(
            "--kv-fraction",
            envvar="BATON_KV_FRACTION",
            help="Share of each worker's budget reserved for the KV cache.",
        ),
    ] = None,
    min_workers: Annotated[
        int,
        typer.Option("--min-workers", help="Wait for this many workers before planning."),
    ] = 1,
    wait: Annotated[
        float, typer.Option("--wait", help="Seconds to wait for workers before planning.")
    ] = 30.0,
    dashboard: Annotated[
        bool, typer.Option("--dashboard/--no-dashboard", help="Serve dashboard/dist at `/`.")
    ] = True,
    local_worker: Annotated[
        bool,
        typer.Option("--local-worker/--no-local-worker", help="Also run a worker on this machine."),
    ] = True,
) -> None:
    """Run the head, plan the cluster, load the model (PRD 7.1)."""
    import asyncio

    _require_torch()
    from baton.head import serve as head

    options = head.ServeOptions(
        model=model,
        # The CLI names are the PRD 15 ones; the planner and the loader use the
        # PRD 5.5 and 9 names.
        quant={"none": "bf16"}.get(quant.value, quant.value),  # type: ignore[arg-type]
        ctx=ctx,
        objective={"balance": "throughput"}.get(objective.value, objective.value),
        port=port,
        control_port=control_port,
        no_local_worker=not local_worker,
        kv_fraction=0.2 if kv_fraction is None else kv_fraction,
        min_workers=min_workers,
        wait_s=wait,
        http=True,
        dashboard=dashboard,
    )
    try:
        asyncio.run(head.serve(options))
    except KeyboardInterrupt:
        raise typer.Exit(130) from None
    except (ValueError, FileNotFoundError) as exc:
        typer.echo(f"baton serve: {exc}", err=True)
        raise typer.Exit(1) from None


@app.command()
def status(
    head: Annotated[
        str, typer.Option("--head", envvar="BATON_HEAD", help="Head HTTP address.")
    ] = DEFAULT_HEAD,
    output: Annotated[
        OutputFormat, typer.Option("--output", "-o", help="Render as a table or raw JSON.")
    ] = OutputFormat.table,
    watch: Annotated[
        float | None, typer.Option("--watch", "-w", help="Refresh every N seconds.")
    ] = None,
) -> None:
    """Print the `/cluster` snapshot as a table (PRD 13.1, 14.3)."""
    raise NotImplementedError("status: GET /cluster")


@app.command()
def bench(
    head: Annotated[
        str, typer.Option("--head", envvar="BATON_HEAD", help="Head HTTP address.")
    ] = DEFAULT_HEAD,
    suite: Annotated[
        str | None,
        typer.Option("--suite", help="Named row of the benchmark matrix. Default: all rows."),
    ] = None,
    prompt_tokens: Annotated[int, typer.Option("--prompt-tokens", help="Prompt length.")] = 512,
    max_tokens: Annotated[int, typer.Option("--max-tokens", help="Tokens to decode.")] = 128,
    concurrency: Annotated[int, typer.Option("--concurrency", help="Parallel requests.")] = 1,
    repeat: Annotated[int, typer.Option("--repeat", help="Runs per row.")] = 3,
    warmup: Annotated[int, typer.Option("--warmup", help="Discarded runs before measuring.")] = 1,
    out: Annotated[Path | None, typer.Option("--out", help="Append results to this CSV.")] = None,
) -> None:
    """Run the benchmark matrix (PRD 17.3)."""
    raise NotImplementedError("bench: bench/")


@app.command()
def selftest(
    backend: Annotated[
        str | None, typer.Option("--backend", help="Force cuda|mps|cpu instead of probing.")
    ] = None,
    output: Annotated[
        OutputFormat, typer.Option("--output", "-o", help="Render as a table or raw JSON.")
    ] = OutputFormat.table,
) -> None:
    """Verify the torch backend and the tier of this device (PRD 5.6)."""
    raise NotImplementedError("selftest: baton.worker.probe")


@app.command()
def replan(
    head: Annotated[
        str, typer.Option("--head", envvar="BATON_HEAD", help="Head HTTP address.")
    ] = DEFAULT_HEAD,
    objective: Annotated[
        Objective | None, typer.Option("--objective", help="Override the planner objective.")
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print the new plan without applying it.")
    ] = False,
) -> None:
    """Re-run the planner, including standby workers (PRD 9)."""
    raise NotImplementedError("replan: baton.head.planner")


@app.command()
def prefetch(
    model: Annotated[
        str, typer.Option("--model", envvar="BATON_MODEL", help="Hugging Face model id or path.")
    ],
    head: Annotated[
        str,
        typer.Option("--head", envvar="BATON_HEAD", help="Head HTTP address; source of the plan."),
    ] = DEFAULT_HEAD,
    layers: Annotated[
        str | None,
        typer.Option("--layers", help="Override the range, as START:END. Default: from the plan."),
    ] = None,
    cache_dir: Annotated[
        Path, typer.Option("--cache-dir", envvar="BATON_CACHE_DIR", help="Shard cache directory.")
    ] = DEFAULT_CACHE_DIR,
) -> None:
    """Download this device's likely shard without loading it (PRD 12.1)."""
    raise NotImplementedError("prefetch: baton.model.safetensors_io")


@cache_app.command("ls")
def cache_ls(
    cache_dir: Annotated[
        Path, typer.Option("--cache-dir", envvar="BATON_CACHE_DIR", help="Shard cache directory.")
    ] = DEFAULT_CACHE_DIR,
    output: Annotated[
        OutputFormat, typer.Option("--output", "-o", help="Render as a table or raw JSON.")
    ] = OutputFormat.table,
) -> None:
    """List cached shards with their model, layer range and size."""
    raise NotImplementedError("cache ls: baton.model.cache")


@cache_app.command("rm")
def cache_rm(
    model: Annotated[
        str | None, typer.Argument(help="Model id to remove. Omit with --all.")
    ] = None,
    all_: Annotated[bool, typer.Option("--all", help="Remove every cached shard.")] = False,
    cache_dir: Annotated[
        Path, typer.Option("--cache-dir", envvar="BATON_CACHE_DIR", help="Shard cache directory.")
    ] = DEFAULT_CACHE_DIR,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Do not prompt.")] = False,
) -> None:
    """Remove cached shards."""
    raise NotImplementedError("cache rm: baton.model.cache")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
