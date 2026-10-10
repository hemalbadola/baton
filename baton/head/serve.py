"""Head node startup sequence. PRD 7.1.

    baton serve --model REPO_ID [--revision main] [--quant bf16|int8|int4]
                [--ctx 8192] [--objective latency|throughput] [--port 7700]
                [--no-local-worker] [--token SECRET] [--hf-token TOKEN]
                [--kv-fraction 0.2]

The steps of 7.1 are one method each, so the CLI and the tests can drive them
one at a time. `token`, `release_ack` and request `error` frames come back on
the control socket, which already exists and is already authenticated, so the
head needs no second listener for the data plane.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import ipaddress
import json
import secrets
import signal
import socket
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

from baton import __version__
from baton.common.framing import FrameError
from baton.common.net import (
    CONTROL_CONNECT_TIMEOUT,
    CONTROL_PORT,
    LinkClosed,
    close_writer,
    format_addr,
    lan_addresses,
    parse_addr,
    read_frame,
    send_frame,
    set_nodelay,
)
from baton.head.driver import Admission, Driver
from baton.head.planner import Device, Fleet, Model, Plan, PlannerError, Workload, format_plan, plan
from baton.head.registry import Capabilities, Health, NameInUse, Registry, Worker
from baton.model import safetensors_io as sio
from baton.model.spec import ModelSpec

DEFAULT_HTTP_PORT = 7700
MDNS_SERVICE = "_baton._tcp.local."
JOIN_QUIET_S = 10.0
JOIN_DEADLINE_S = 30.0
BENCH_TIMEOUT_S = 60.0
SWEEP_INTERVAL_S = 1.0

#: Bytes per weight in each compute dtype. The planner counts in bf16 bytes, so
#: a worker that computes in fp32 can hold half as many layers per byte.
DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4}

State = Literal["starting", "idle", "loading", "ready"]
Quant = Literal["bf16", "int8", "int4"]


@dataclass
class ServeOptions:
    """Every `baton serve` flag (PRD 7.1)."""

    model: str
    revision: str = "main"
    quant: Quant = "int4"
    ctx: int = 8192
    objective: str = "latency"
    port: int = DEFAULT_HTTP_PORT
    control_port: int = CONTROL_PORT
    """0 binds an ephemeral port, which is what a test wants."""

    no_local_worker: bool = False
    token: str | None = None
    hf_token: str | None = None
    kv_fraction: float = 0.2
    plan_override: str | None = None
    min_workers: int = 1
    wait_s: float = JOIN_DEADLINE_S
    dashboard: bool = True
    http: bool = False
    """Serve the OpenAI API and the dashboard on `port`. `baton serve` sets it."""


@dataclass
class ModelMetadata:
    """What step 4 fetches.

    `index` maps a tensor name to its shard file. `shard_headers` maps a shard
    file name to `(header, header_len)`: the JSON header that the head read
    with a ranged request over the first `8 + header_len` bytes. The head never
    downloads a weight: each worker range-fetches its own layers (PRD D9, 12.1).
    """

    config: dict[str, Any]
    tokenizer_config: dict[str, Any]
    index: dict[str, str]
    shard_headers: dict[str, tuple[dict[str, Any], int]]
    tokenizer: Any = None
    eos_ids: frozenset[int] = frozenset()

    @property
    def spec(self) -> ModelSpec:
        return ModelSpec.from_config(self.config)

    def _bf16_bytes(self, names: Any) -> int:
        refs: dict[str, sio.TensorRef] = {}
        for file, (header, header_len) in self.shard_headers.items():
            refs.update(sio.refs_from_header(file, header, header_len))
        return 2 * sum(refs[name].numel() for name in names)

    @property
    def embed_bytes(self) -> int:
        """The embedding table in bf16 bytes. N1 holds it resident (BAT-11)."""
        return self._bf16_bytes([self.spec.tensor_names["embed"]])

    def planner_model(self, quant: Quant) -> Model:
        """The planner's view of this model (PRD 9.1), in bf16 bytes.

        ponytail: `quant` does not change the sizes, because only bf16 loads
        today (BAT-10). The planner model has no term for the embedding table
        on N1: `Head.make_plan` charges that to whichever device ends up first.
        """
        spec = self.spec
        return Model(
            layers=spec.n_layers,
            weight_bytes_per_layer=self._bf16_bytes(spec.layer_names(0).values()),
            lm_head_bytes=self._bf16_bytes(spec.head_names().values()),
            kv_bytes_per_token_per_layer=2 * spec.n_kv_heads * spec.head_dim * 2,
        )


def read_metadata(model: str, token: str | None = None, revision: str = "main") -> ModelMetadata:
    """Read `config.json`, the tensor index and every shard header. Blocking.

    `model` is a checkpoint directory or a Hugging Face repo id. A checkpoint
    with one `model.safetensors` and no index file gets an index made from its
    header.
    """

    def read_json(name: str) -> dict[str, Any] | None:
        if Path(model).is_dir():
            path = Path(model) / name
            return json.loads(path.read_text()) if path.exists() else None
        import httpx

        response = httpx.get(
            f"https://huggingface.co/{model}/resolve/{revision}/{name}",
            headers={"Authorization": f"Bearer {token}"} if token else {},
            follow_redirects=True,
            timeout=60.0,
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    config = read_json("config.json")
    if config is None:
        raise FileNotFoundError(f"{model}: no config.json. Is it a model directory or repo id?")
    index: dict[str, str] = (read_json("model.safetensors.index.json") or {}).get("weight_map", {})
    files = sorted(set(index.values())) or ["model.safetensors"]
    source = sio.open_source(model, token, revision)
    try:
        headers = {file: sio.fetch_header(source, file) for file in files}
    finally:
        if hasattr(source, "close"):
            source.close()
    if not index:
        index = {name: files[0] for name in headers[files[0]][0] if name != "__metadata__"}
    return ModelMetadata(config=config, tokenizer_config={}, index=index, shard_headers=headers)


def load_tokenizer(model: str, token: str | None = None, revision: str = "main") -> Any:
    """The Hugging Face tokenizer of `model`: a checkpoint directory or a repo id."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model, token=token or None, revision=revision)


def eos_ids(config: dict[str, Any], tokenizer: Any) -> frozenset[int]:
    """Every id that ends a reply: the config's eos list and the tokenizer's end-of-turn token."""
    found: set[int] = set()
    eos = config.get("eos_token_id")
    found.update(eos if isinstance(eos, list) else [] if eos is None else [eos])
    if tokenizer is not None:
        if tokenizer.eos_token_id is not None:
            found.add(int(tokenizer.eos_token_id))
        eot = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if isinstance(eot, int) and eot != tokenizer.unk_token_id:
            found.add(eot)
    return frozenset(int(i) for i in found)


def _echo(line: str) -> None:
    """Flushed, so `baton serve | tee log` shows each line as it happens."""
    print(line, flush=True)


@dataclass
class Head:
    """The head node process (PRD 7.1).

    `state` follows the CLI status line: starting, idle, loading, ready. An
    infeasible plan leaves the head in `idle` with the shortfall printed, and
    the next worker to join starts a new attempt.
    """

    options: ServeOptions
    registry: Registry = field(default_factory=Registry)
    state: State = "starting"
    metadata: ModelMetadata | None = None
    plan: Plan | None = None
    driver: Driver | None = None
    cluster_id: str = field(default_factory=lambda: secrets.token_hex(4))
    plan_rev: int = 0
    control_port: int = 0
    """The bound control port. Differs from the option when that is 0."""

    echo: Callable[[str], None] = _echo

    _server: asyncio.Server | None = field(default=None, repr=False)
    _mdns: Any = field(default=None, repr=False)
    _sweeper: asyncio.Task[None] | None = field(default=None, repr=False)
    _local_worker: asyncio.subprocess.Process | None = field(default=None, repr=False)
    _changed: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    _joins: int = field(default=0, repr=False)
    _last_join: float = field(default=0.0, repr=False)
    _errors: dict[str, str] = field(default_factory=dict, repr=False)
    _progress: dict[str, int] = field(default_factory=dict, repr=False)
    _http: Any = field(default=None, repr=False)
    _tasks: set[asyncio.Task[None]] = field(default_factory=set, repr=False)
    events: list[dict[str, Any]] = field(default_factory=list, repr=False)
    """Newest last, capped at 200. The dashboard log reads this (PRD 14.2)."""

    # --- the sequence ---------------------------------------------------------

    async def run(self) -> None:
        """Run steps 1 to 8, then keep the cluster formed until cancelled.

        A worker lost from the ring drops the head back to `idle`, and the loop
        plans again over whoever is left.
        """
        if self.options.quant != "bf16":
            raise ValueError(
                f"--quant {self.options.quant} is not loadable yet (BAT-10). Use --quant none."
            )
        try:
            await self.start_servers()
            await self.start_http()
            await self.advertise()
            if not self.options.no_local_worker:
                await self.spawn_local_worker()
            self.metadata = await self.fetch_metadata()
            self.state = "idle"
            while True:
                await self.wait_for_workers()
                # Read before the attempt: a worker that joins during the bench
                # or the load must start the next attempt, not be missed.
                seen = self._joins
                if await self.form_cluster():
                    await self._wait(lambda: self.state != "ready")
                else:  # idle: the next worker to join is the next attempt
                    self.echo("not ready. Waiting for another worker to join.")
                    await self._wait(lambda seen=seen: self._joins != seen)
        finally:
            await self.close()

    async def start_servers(self) -> None:
        """Step 1. Control server on 7711. The HTTP server is the API lane's."""
        self._server = await asyncio.start_server(
            self._on_control, "0.0.0.0", self.options.control_port
        )
        self.control_port = self._server.sockets[0].getsockname()[1]
        self._sweeper = asyncio.create_task(self._sweep())
        self.echo(f"baton head {self.cluster_id}: control on port {self.control_port}")
        # mDNS does not cross every Wi-Fi: phone hotspots and guest networks
        # often drop it. The address always works.
        address = format_addr(lan_addresses()[0], self.control_port)
        self.echo(
            f"workers on this network join by themselves. If one cannot: "
            f"baton worker --head {address}"
        )

    async def start_http(self) -> None:
        """Step 1, second half: the OpenAI API and the dashboard (PRD 13)."""
        if not self.options.http:
            return
        import uvicorn

        from baton.head.api import ClusterApi, create_app

        dist = Path(__file__).resolve().parents[2] / "dashboard" / "dist"
        mount = str(dist) if self.options.dashboard and dist.is_dir() else None
        app = create_app(ClusterApi(self), mount)
        config = uvicorn.Config(app, host="0.0.0.0", port=self.options.port, log_level="warning")
        self._http = uvicorn.Server(config)
        self._http.install_signal_handlers = lambda: None  # `serve` owns SIGTERM
        self._tasks.add(asyncio.create_task(self._http.serve()))

    async def advertise(self) -> None:
        """Step 2. Announce `_baton._tcp.local.` over mDNS.

        The TXT record carries `{version, control_port, cluster_id}`. Workers
        join the first head they see. Two heads on one LAN hold no election in
        v1: each dashboard warns about the other record (PRD 7.2).

        Every LAN address goes in the record. The worker dials each one and
        keeps the first that answers, so a head with Wi-Fi, Ethernet and a VPN
        is still found on the right interface.
        """
        from zeroconf import ServiceInfo
        from zeroconf.asyncio import AsyncZeroconf

        info = ServiceInfo(
            MDNS_SERVICE,
            f"baton-{self.cluster_id}.{MDNS_SERVICE}",
            addresses=[socket.inet_aton(address) for address in lan_addresses()],
            port=self.control_port,
            properties={
                "version": __version__,
                "control_port": str(self.control_port),
                "cluster_id": self.cluster_id,
            },
            server=f"baton-{self.cluster_id}.local.",
        )
        self._mdns = AsyncZeroconf()
        await self._mdns.async_register_service(info)

    async def spawn_local_worker(self) -> None:
        """Step 3. Start a worker subprocess with `--head 127.0.0.1`.

        The head skips this step when `--no-local-worker` is set.
        """
        self._local_worker = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "baton.cli",
            "worker",
            "--head",
            f"127.0.0.1:{self.control_port}",
        )

    async def fetch_metadata(self) -> ModelMetadata:
        """Step 4. Fetch the model metadata (PRD 7.1, 12.1)."""
        o = self.options
        self.echo(f"reading model metadata: {o.model}")
        meta = await asyncio.to_thread(read_metadata, o.model, o.hf_token, o.revision)
        try:
            meta.tokenizer = await asyncio.to_thread(
                load_tokenizer, o.model, o.hf_token, o.revision
            )
        except Exception as exc:  # noqa: BLE001 - the cluster can still form, the API cannot
            self.echo(f"no tokenizer ({exc!r}): the HTTP API cannot take text")
        meta.eos_ids = eos_ids(meta.config, meta.tokenizer)
        return meta

    async def wait_for_workers(self) -> None:
        """Step 5. Wait for the fleet, then print the roster.

        Planning starts when `min_workers` have joined and either no new worker
        joined for 10 s, or `wait_s` passed. Workers do not advertise, so the
        head cannot count them before they join: `min_workers` is how the user
        says "wait for both laptops".
        """
        deadline = time.monotonic() + self.options.wait_s
        quiet = min(JOIN_QUIET_S, self.options.wait_s)

        def live() -> int:
            return sum(1 for worker in self.registry if worker.state != "lost")

        while True:
            now = time.monotonic()
            if live() >= self.options.min_workers and (
                now - self._last_join >= quiet or now >= deadline
            ):
                break
            await asyncio.sleep(0.1)
        self.echo(self.registry.roster_table())

    async def form_cluster(self) -> bool:
        """Steps 6 to 8: bench, plan, load. True when the cluster is READY."""
        try:
            self.plan = self.make_plan(await self.benchmark())
        except PlannerError as exc:
            self.echo(f"cannot plan: {exc}")
            return False
        if not self.plan.feasible:
            return False
        self.state = "loading"
        if not await self.load_plan(self.plan):
            self.state = "idle"
            return False
        self.driver = self.make_driver(self.plan)
        self.state = "ready"
        self.event("replan" if self.plan_rev > 1 else "join", f"plan {self.plan_rev} is loaded")
        self.echo(f"READY: plan {self.plan_rev} is loaded on {len(self.plan.assignments)} node(s)")
        self.echo(self.registry.roster_table())
        self.print_urls()
        return True

    def make_driver(self, plan_: Plan) -> Driver | None:
        """The request driver for one loaded plan (PRD 7.3). None without a tokenizer."""
        assert self.metadata is not None
        if self.metadata.tokenizer is None:
            return None
        spec = self.metadata.spec
        per_token: dict[str, int] = {}
        budgets: dict[str, int] = {}
        for a in plan_.assignments:
            caps = self.registry.get(a.name).capabilities  # type: ignore[union-attr]
            per_token[a.name] = (
                2 * spec.n_kv_heads * spec.head_dim * DTYPE_BYTES[caps.compute_dtype]
            )
            budgets[a.name] = int(caps.usable_bytes * plan_.kv_fraction)
        admission = Admission(plan_, budgets, per_token=per_token)
        return Driver(
            self.registry, admission, self.metadata.tokenizer, self.options.ctx,
            eos_ids=self.metadata.eos_ids,
        )  # fmt: skip

    def event(self, kind: str, msg: str) -> None:
        self.events.append({"t": time.time(), "kind": kind, "msg": msg})
        del self.events[:-200]

    async def benchmark(self) -> Fleet:
        """Step 6. Send `bench` to every worker that has no timing yet, wait up to 60 s.

        A worker that does not answer is left out of the fleet, not waited on
        forever. The RTT matrix of PRD 9.1 is BAT-12: the planner uses its
        default hop cost until then.
        """
        assert self.metadata is not None
        spec = self.metadata.spec.to_dict()
        self._errors.clear()
        asked = []
        for worker in self.registry:
            caps = worker.capabilities
            if worker.state == "lost" or caps.t_dec_ms > 0:
                continue
            frame = {
                "t": "bench",
                "spec": spec,
                "quant": self.options.quant,
                "dtype": caps.compute_dtype,
            }
            if await self._send(worker, frame):
                asked.append(worker.name)

        def answered() -> bool:
            for name in asked:
                worker = self.registry.get(name)
                if worker is None or worker.state == "lost" or name in self._errors:
                    continue
                if worker.capabilities.t_dec_ms <= 0:
                    return False
            return True

        if asked:
            self.echo(f"benchmarking {', '.join(asked)}")
            if not await self._wait(answered, BENCH_TIMEOUT_S):
                self.echo("benchmark timed out; planning without the silent workers")

        devices = []
        for name, usable, t_dec, t_pre, fast in self.registry.devices():
            worker = self.registry.get(name)
            assert worker is not None
            scale = DTYPE_BYTES.get(worker.capabilities.compute_dtype, 4)
            devices.append(Device(name, usable * 2 // scale, t_dec, t_pre, fast))
        return Fleet(devices=tuple(devices))

    def make_plan(self, fleet: Fleet) -> Plan:
        """Step 7. Run the planner and print the plan (PRD 9).

        An infeasible fleet prints the shortfall and leaves the head in `idle`.

        N1 holds the embedding table as well as its layers (BAT-32). The
        planner picks N1 itself, so the plan is made, the table is charged to
        the device that came out first, and the plan is made again until the
        first device is one that was charged. A worker would otherwise be
        handed a shard that its own budget check refuses.
        """
        assert self.metadata is not None
        o = self.options
        if not fleet.devices:
            made = Plan(feasible=False, objective=o.objective, reason="no worker is benchmarked")
            self.echo(format_plan(made))
            return made
        model = self.metadata.planner_model(o.quant)
        workload = Workload(objective=o.objective, ctx=o.ctx, kv_fraction=o.kv_fraction)
        embed, tied = self.metadata.embed_bytes, self.metadata.spec.tie_embeddings
        charged: frozenset[str] = frozenset()
        tried: set[str] = set()
        while True:
            devices = tuple(
                replace(d, usable_bytes=max(0, d.usable_bytes - embed)) if d.name in charged else d
                for d in fleet.devices
            )
            made = plan(Fleet(devices, fleet.rtt_ms, fleet.default_rtt_ms), model, workload)
            if not made.feasible:
                break
            first = made.assignments[0].name
            # One node with a tied head already pays for the table as its lm_head.
            if first in charged or (tied and len(made.assignments) == 1):
                break
            # A device seen twice means the planner swaps N1 each time one is
            # charged. Charging every candidate ends that, at some cost in room.
            charged = frozenset(tried | {first}) if first in tried else frozenset({first})
            tried.add(first)
        self.echo(format_plan(made))
        return made

    async def load_plan(self, plan_: Plan) -> bool:
        """Step 8. Send `load` to every node in the plan and wait for `loaded`.

        True when every node loaded. On any failure the nodes that did load are
        told to unload, so no laptop keeps a shard of a ring that never formed.
        """
        assert self.metadata is not None
        o = self.options
        self.plan_rev += 1
        self._errors.clear()
        self._progress.clear()
        ring = plan_.assignments
        names = [a.name for a in ring]
        for worker in self.registry:  # a node the new plan leaves out must not keep its shard
            if worker.name not in names and worker.layers:
                if worker.state != "lost":
                    await self._send(worker, {"t": "unload"})
                self.registry.release(worker.name)
        for position, a in enumerate(ring):
            worker = self.registry.get(a.name)
            if worker is None or worker.state == "lost":
                self._errors[a.name] = "lost before load"
                break
            # Nk dials N1, so `next` and `release` ride the same ring (PRD 10.2).
            # One node needs no link: the ring is that process.
            if position + 1 < len(ring):
                following = self.registry.get(names[position + 1])
            else:
                following = self.registry.get(names[0]) if len(ring) > 1 else None
            self.registry.assign(a.name, a.first_layer, a.last_layer, a.holds_lm_head)
            frame: dict[str, Any] = {
                "t": "load",
                "plan_rev": self.plan_rev,
                "model": o.model,
                "spec": self.metadata.spec.to_dict(),
                "range": [a.first_layer, a.last_layer],
                "quant": o.quant,
                "roles": {"embed": a.first_layer == 0, "head": a.holds_lm_head},
                "ctx_max": o.ctx,
                "kv_budget_bytes": int(worker.capabilities.usable_bytes * plan_.kv_fraction),
                "next_node": self._ring_addr(worker, following) if following else "",
                "head_data_addr": "",
                "index": self.metadata.index,
                "headers": {},
            }
            if o.hf_token:
                frame["hf_token"] = o.hf_token
            await self._send(worker, frame)

        def states() -> list[str]:
            return [w.state if (w := self.registry.get(n)) else "lost" for n in names]

        def failed() -> bool:  # a ring member reported an error, or left the ring
            return any(n in self._errors for n in names) or bool(
                set(states()) - {"loading", "loaded"}
            )

        await self._wait(lambda: failed() or "loading" not in states())
        if not failed():
            return True
        self.echo("load failed; unloading the ring")
        for name in names:
            worker = self.registry.get(name)
            if worker is not None and worker.state != "lost":
                await self._send(worker, {"t": "unload"})
            self.registry.release(name)
        return False

    def print_urls(self) -> None:
        """Step 9. Print the API URL and the dashboard URL."""
        if not self.options.http:
            return
        base = f"http://{lan_addresses()[0]}:{self.options.port}"
        self.echo(f"dashboard: {base}/\napi:       {base}/v1  (OpenAI compatible)")

    async def replan(self, lost: str) -> Plan:
        """Re-plan after a worker loss (PRD 7.6, 11).

        `run` already plans again from scratch when the ring loses a node. This
        method is the incremental form (PRD 9.7, 5.4), where a device whose new
        range overlaps its old one fetches only the missing layers (BAT-13).
        """
        raise NotImplementedError

    async def close(self) -> None:
        """Stop advertising, close every socket, stop the local worker."""
        if self._sweeper is not None:
            self._sweeper.cancel()
        if self._http is not None:
            self._http.should_exit = True
            await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._mdns is not None:
            with contextlib.suppress(Exception):
                await self._mdns.async_unregister_all_services()
                await self._mdns.async_close()
            self._mdns = None
        if self._server is not None:
            self._server.close()
        for worker in self.registry:
            if worker.control is not None:
                await close_writer(worker.control)
        if self._local_worker is not None and self._local_worker.returncode is None:
            self._local_worker.terminate()
            await self._local_worker.wait()

    # --- the control plane ------------------------------------------------------

    @staticmethod
    def _ring_addr(worker: Worker, following: Worker) -> str:
        """The address `worker` must dial to reach `following` (PRD 6.4 step 4).

        The local worker joins through 127.0.0.1, so that is the data address it
        reports. To a worker on another machine that address means itself. The
        local worker lives on the head, and the remote worker already reaches
        the head at the local end of its own control socket.
        """
        host, port = following.data_address
        peer = worker.control.get_extra_info("peername")
        if _is_loopback(host) and peer and not _is_loopback(peer[0]):
            host = worker.control.get_extra_info("sockname")[0]
        return format_addr(host, port)

    async def _wait(self, done: Callable[[], bool], timeout: float | None = None) -> bool:
        """Block until `done()` holds. It is checked again after every frame."""
        try:
            async with asyncio.timeout(timeout):
                while not done():
                    self._changed.clear()
                    await self._changed.wait()
        except TimeoutError:
            return False
        return True

    async def _send(self, worker: Worker, frame: dict[str, Any]) -> bool:
        try:
            await send_frame(worker.control, frame)
        except OSError:
            self._lose(worker.name, "control connection closed")
            return False
        return True

    def _lose(self, name: str, why: str) -> None:
        worker = self.registry.get(name)
        if worker is None or worker.state == "lost":
            return
        in_ring = worker.state in ("loading", "loaded")
        self.registry.mark_lost(name)
        self.echo(f"lost {name}: {why}")
        self.event("loss", f"{name}: {why}")
        if in_ring and self.driver is not None:
            self.driver.fail_all("worker_lost", f"{name} left the ring: {why}")
            self.driver = None
        if in_ring and self.state == "ready":
            self.state = "idle"
        self._changed.set()

    async def _sweep(self) -> None:
        """Mark a worker lost after 6 s without `health` (PRD 8.6)."""
        while True:
            await asyncio.sleep(SWEEP_INTERVAL_S)
            for worker in self.registry.timed_out(self.registry.clock()):
                self._lose(worker.name, "no health frame for 6 s")
                if worker.control is not None:
                    worker.control.close()

    async def _on_control(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """One control connection: `hello`, then frames until it closes."""
        set_nodelay(writer)
        name: str | None = None
        try:
            async with asyncio.timeout(CONTROL_CONNECT_TIMEOUT):
                hello, _ = await read_frame(reader)
            name = await self._admit(hello, writer)
            while name is not None:
                frame, _ = await read_frame(reader)
                owner = self.registry.get(name)
                if owner is None or owner.control is not writer:
                    break  # a newer hello owns this name; this socket is stale
                self._on_frame(name, frame)
        except (LinkClosed, FrameError, TimeoutError, OSError):
            pass
        finally:
            worker = self.registry.get(name) if name else None
            # A reconnect replaces `control`. Only the socket that still owns the
            # entry may take the worker down with it.
            if worker is not None and worker.control is writer:
                self._lose(worker.name, "control connection closed")
            await close_writer(writer)

    async def _admit(self, hello: dict[str, Any], writer: asyncio.StreamWriter) -> str | None:
        """Check `hello` and register the worker (PRD 7.2, 8.5). None means refused.

        `hello` is the trust boundary of the control plane: every field is
        checked or coerced before it reaches the registry.
        """

        async def refuse(code: str, message: str) -> None:
            await send_frame(writer, {"t": "error", "code": code, "message": message})

        if hello.get("t") != "hello":
            return await refuse("protocol", "the first frame must be hello")
        token = self.options.token
        if token and not hmac.compare_digest(str(hello.get("token", "")).encode(), token.encode()):
            return await refuse("auth", "token does not match")
        name, caps = hello.get("name"), hello.get("caps")
        try:
            if not isinstance(name, str):
                raise TypeError("hello carries no name")
            data_address = parse_addr(str(hello.get("data_addr", "")))
            old = self.registry.get(name)
            if old is not None and old.state != "lost":
                # PRD 7.2 refuses a second address only for a loading or loaded
                # name. A standby name is refused too: two machines that share a
                # hostname would otherwise share one registry entry.
                if old.data_address != data_address:
                    raise NameInUse(f"name in use: {name} is live at {old.data_addr}")
                old.control.close()  # the same worker redialled; drop its old socket
            self.registry.hello(
                name,
                data_address,
                Capabilities.from_caps(caps if isinstance(caps, dict) else {}),
                writer,
            )
        except NameInUse as exc:
            return await refuse("name_in_use", str(exc))
        except (TypeError, ValueError) as exc:
            return await refuse("bad_hello", str(exc))

        welcome: dict[str, Any] = {"t": "welcome", "cluster_id": self.cluster_id, "your_name": name}
        if self.plan_rev:
            welcome["plan_rev"] = self.plan_rev
        await send_frame(writer, welcome)
        self._joins += 1
        self._last_join = time.monotonic()
        self.echo(f"joined: {name}")
        self.event("join", f"{name} joined")
        self._changed.set()
        return name

    def _on_frame(self, name: str, frame: dict[str, Any]) -> None:
        """Apply one worker-to-head frame. A malformed one is dropped, not fatal."""
        kind = frame.get("t")
        try:
            if kind == "health":
                self.registry.on_health(name, Health.from_frame(frame, self.registry.clock()))
            elif kind == "bench_result":
                self.registry.on_bench(name, frame["t_dec_ms"], frame["t_pre_ms"])
            elif kind == "token" and self.driver is not None:
                self.driver.on_token(
                    str(frame["req"]),
                    int(frame["id"]),
                    int(frame["pos"]),
                    bool(frame["final"]),
                    str(frame.get("reason", "")),
                )
            elif kind == "release_ack" and self.driver is not None:
                self.driver.on_release_ack(str(frame["req"]))
            elif kind == "link_down" and self.driver is not None:
                self.driver.fail_all("worker_lost", f"ring link to {frame.get('peer')} is down")
            elif kind == "error" and "req" in frame:
                if self.driver is not None:
                    self.driver.on_error(
                        str(frame["req"]), str(frame.get("code")), str(frame.get("message"))
                    )
            elif kind == "load_progress":
                tenth = 10 * int(frame["done"]) // max(1, int(frame["total"]))
                if tenth != self._progress.get(name):
                    self._progress[name] = tenth
                    self.echo(
                        f"  {name}: {frame['done']}/{frame['total']} tensors, "
                        f"{int(frame['bytes']) / 1024**2:.0f} MB"
                    )
            elif kind == "loaded":
                if int(frame["rev"]) == self.plan_rev:
                    self.registry.mark_loaded(name)
                    self.echo(
                        f"  {name}: loaded {int(frame['resident_bytes']) / 1024**2:.0f} MB "
                        f"in {float(frame['seconds']):.1f} s"
                    )
            # A load error names its plan revision. One from an old plan must
            # not fail the plan that replaced it.
            elif (
                kind == "error"
                and "req" not in frame
                and int(frame.get("rev", self.plan_rev)) == self.plan_rev
            ):
                self._errors[name] = f"{frame.get('code')}: {frame.get('message')}"
                self.echo(f"error from {name}: {self._errors[name]}")
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            self.echo(f"bad {kind!r} frame from {name}: {exc!r}")
        self._changed.set()


def _is_loopback(host: str) -> bool:
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:  # a hostname
        return False


async def serve(options: ServeOptions) -> None:
    """Entry point for `baton serve`. Builds a Head and runs it."""
    task = asyncio.current_task()
    assert task is not None
    # SIGTERM must run `Head.close`, or the local worker outlives the head.
    with contextlib.suppress(NotImplementedError):  # no signal handlers on Windows
        asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)
    with contextlib.suppress(asyncio.CancelledError):
        await Head(options).run()
