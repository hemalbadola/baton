"""A hardcoded ring of worker processes over raw TCP (the M0 exit test, PRD 19).

No head node, no discovery, no planner. The test process plays the head: it
sends `prompt` to N1 and reads `token` from Nk. Each node is one process with
one asyncio loop, one listener, one link to the next node and one FIFO queue
between them (PRD 10.5). Layer ranges are fixed by the caller.

Ring of k nodes: N1 embeds, Nk carries `norm`, `lm_head` and the sampler. Nk's
next node is N1, so `next` and `release` travel the same link as `act`
(PRD 10.2). With k = 1 the ring is one process and a frame for "the next node"
goes on the node's own queue without a socket. That single-node ring is the
baseline the multi-process rings are compared against.

The `prompt` frame is forwarded around the ring ahead of the first `act`, so
Nk learns `max_len`, `sampling` and `stop_ids` from the frame PRD 8.3 defines
for them, and nothing rides on `act` beyond its schema (PRD 10.3 stores them
per request on Nk).

Per-frame timing never crosses the wire beyond the `trace` entries PRD 17.2
names. Each node keeps its own `(t_recv, t_send, compute)` samples and writes
them to `stats_path` when the head terminates it.

Weights come from one of two sources. `("random", seed)` draws every tensor
from a generator seeded by the tensor's checkpoint name, so two processes
holding different layer ranges still hold the same numbers. `("safetensors",
path)` reads only the tensors the node owns.
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import os
import signal
import socket
import sys
import tempfile
import time
import traceback
import zlib
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from baton.common.net import (
    LinkClosed,
    accept_data_link,
    close_writer,
    open_data_link,
    read_frame,
    send_frame,
)
from baton.model.layers import DecoderStack
from baton.model.spec import ModelSpec
from baton.worker.sampler import Sampler, SamplingParams

CLUSTER = "m0"
CHUNK = 256  # PRD 10.1, decision D10
DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}
GREEDY: dict[str, Any] = {"temperature": 0}

# A 1B shard in fp32 loads in well under a minute; a peer that takes longer is
# stuck, not slow.
LOAD_TIMEOUT = 300.0
# No frame carries a deadline (PRD 8.6). The head still needs one so a dead
# node fails the test instead of hanging it.
FRAME_TIMEOUT = 300.0

WeightSource = tuple[str, Any]


@dataclass(frozen=True)
class NodeConfig:
    name: str
    port: int
    next_addr: str  # "" when the node is its own next (k == 1)
    head_addr: str
    spec: ModelSpec
    layer_start: int
    layer_end: int
    weights: WeightSource
    device: str = "cpu"
    compute: str = "fp32"
    wire: str = "fp32"
    max_ctx: int = 512
    # Dtype the weights sit in. A bf16 checkpoint computed in fp32 (PRD 16.2,
    # `--quant bf16` with compute forced to fp32) is stored as bf16 and upcast
    # inside each matmul: the upcast is exact, and the shard is half the size.
    storage: str = "fp32"
    stats_path: str = ""  # where the node dumps its timing samples on SIGTERM

    @property
    def embed(self) -> bool:
        return self.layer_start == 0

    @property
    def head(self) -> bool:
        return self.layer_end == self.spec.n_layers


# --- weights ---------------------------------------------------------------


def _shape(spec: ModelSpec, key: str) -> tuple[int, ...]:
    h, q, kv, inter, v = spec.hidden, spec.q_dim, spec.kv_dim, spec.intermediate, spec.vocab
    return {
        "embed": (v, h),
        "input_layernorm": (h,),
        "q_proj": (q, h),
        "k_proj": (kv, h),
        "v_proj": (kv, h),
        "o_proj": (h, q),
        "q_bias": (q,),
        "k_bias": (kv,),
        "v_bias": (kv,),
        "post_attention_layernorm": (h,),
        "gate_proj": (inter, h),
        "up_proj": (inter, h),
        "down_proj": (h, inter),
        "final_norm": (h,),
        "lm_head": (v, h),
    }[key]


def _random_tensor(name: str, shape: tuple[int, ...], seed: int) -> torch.Tensor:
    """Seeded by name, so the same tensor comes out in any process.

    The scale is chosen so the layers, not the embedding, decide the argmax.
    With a tied `lm_head` the residual stream carries a copy of the input
    embedding, and if the layers add little the model echoes the last token
    for the whole generation: measured, embedding std 1 gives one distinct
    token per 32, std 0.1 with projections at `2 * fan_in ** -0.5` gives 28 to
    32. A ring test that passes on an echo proves little, so the chaotic
    regime is the one used.
    """
    gen = torch.Generator().manual_seed((seed << 32) ^ zlib.crc32(name.encode()))
    t = torch.randn(shape, generator=gen)
    if name.endswith("norm.weight"):
        return 1 + 0.1 * t
    if len(shape) == 2 and "embed" not in name:
        return t * 2 * shape[1] ** -0.5
    return 0.1 * t


def tensor_keys(spec: ModelSpec, start: int, end: int, *, embed: bool, head: bool):
    """`(key, checkpoint name)` pairs a node holding `[start, end)` needs."""
    pairs: list[tuple[str, str]] = []
    if embed:
        pairs.append(("embed", spec.tensor_names["embed"]))
    for i in range(start, end):
        pairs.extend(spec.layer_names(i).items())
    if head:
        pairs.extend(spec.head_names().items())
    return pairs


def load_weights(
    spec: ModelSpec, source: WeightSource, start: int, end: int, *, embed: bool, head: bool
) -> dict[str, torch.Tensor]:
    kind, arg = source
    pairs = tensor_keys(spec, start, end, embed=embed, head=head)
    if kind == "random":
        return {name: _random_tensor(name, _shape(spec, key), arg) for key, name in pairs}
    if kind == "safetensors":
        from safetensors import safe_open

        with safe_open(arg, "pt") as f:
            return {name: f.get_tensor(name) for _, name in pairs}
    raise ValueError(f"unknown weight source {kind!r}")


class UpcastLinear(nn.Linear):
    """`nn.Linear` whose weight is cast to the input's dtype on every call.

    This is the quant-tier versus compute-dtype split of PRD 5.5 in its
    simplest form: bf16 storage, fp32 maths. A bf16 value converts to fp32
    without rounding, so a stack built this way computes the same bits as one
    holding fp32 copies of the same weights, in a third of the memory.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = None if self.bias is None else self.bias.to(x.dtype)
        return F.linear(x, self.weight.to(x.dtype), bias)


def build_stack(cfg: NodeConfig) -> DecoderStack:
    stack = DecoderStack(
        cfg.spec,
        cfg.layer_start,
        cfg.layer_end,
        embed=cfg.embed,
        head=cfg.head,
        max_ctx=cfg.max_ctx,
        device=cfg.device,
        dtype=DTYPES[cfg.storage],
    )
    weights = load_weights(
        cfg.spec, cfg.weights, cfg.layer_start, cfg.layer_end, embed=cfg.embed, head=cfg.head
    )
    stack.load_hf_weights(weights)
    if cfg.storage != cfg.compute:
        for m in stack.modules():
            if type(m) is nn.Linear:
                m.__class__ = UpcastLinear
    return stack.eval()


# --- KV cache --------------------------------------------------------------


class KV:
    """The smallest thing that satisfies `layers.LayerKVCache`, one per layer."""

    def __init__(self, spec: ModelSpec, max_len: int, device: str, dtype: torch.dtype) -> None:
        shape = (max_len, spec.n_kv_heads, spec.head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.length = 0

    def append(self, k: torch.Tensor, v: torch.Tensor, pos_start: int):
        n = k.shape[0]
        self.k[pos_start : pos_start + n] = k
        self.v[pos_start : pos_start + n] = v
        self.length = max(self.length, pos_start + n)
        return self.k[: self.length], self.v[: self.length]


def make_kv(spec: ModelSpec, n_layers: int, max_len: int, device: str, dtype: torch.dtype):
    return [KV(spec, max_len, device, dtype) for _ in range(n_layers)]


# --- payload ---------------------------------------------------------------


def to_wire(x: torch.Tensor, wire: str) -> bytes:
    return x.to("cpu", DTYPES[wire]).contiguous().view(torch.uint8).numpy().tobytes()


def from_wire(payload: bytes, n: int, hidden: int, wire: str) -> torch.Tensor:
    # `bytearray` because `torch.frombuffer` wants a writable buffer.
    return torch.frombuffer(bytearray(payload), dtype=DTYPES[wire]).view(n, hidden)


# --- one node --------------------------------------------------------------


@dataclass
class Request:
    """Per-request state on Nk (PRD 10.3)."""

    prompt_ids: list[int]
    max_tokens: int
    stop_ids: set[int]
    sampler: Sampler
    gen: list[int] = field(default_factory=list)


class Node:
    def __init__(self, cfg: NodeConfig) -> None:
        self.cfg = cfg
        self.stack = build_stack(cfg)
        self.dtype = DTYPES[cfg.compute]
        self.kv: dict[str, list[KV]] = {}
        self.reqs: dict[str, Request] = {}
        self.stats: list[tuple[float, float, float]] = []  # (t_recv, t_send, compute)
        self.inbox: asyncio.Queue[tuple[float, dict, bytes]] = asyncio.Queue()
        self.next_writer: asyncio.StreamWriter | None = None
        self.head_writer: asyncio.StreamWriter | None = None

    async def serve(self) -> None:
        cfg = self.cfg
        asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, self._dump_and_exit)
        # Listen only once the shard is resident, so "accepting" means "ready".
        server = await asyncio.start_server(self._accept, "127.0.0.1", cfg.port)
        if cfg.next_addr:
            _, self.next_writer = await open_data_link(
                cfg.next_addr, CLUSTER, total_timeout=LOAD_TIMEOUT, label="next node"
            )
        if cfg.head:
            _, self.head_writer = await open_data_link(
                cfg.head_addr, CLUSTER, total_timeout=LOAD_TIMEOUT, label="head"
            )
        async with server:
            await self._work()

    async def _accept(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await accept_data_link(reader, writer, CLUSTER)
        try:
            while True:
                meta, payload = await read_frame(reader)
                self.inbox.put_nowait((time.perf_counter(), meta, payload))
        except LinkClosed:
            pass
        finally:
            await close_writer(writer)

    async def _work(self) -> None:
        while True:
            t_recv, meta, payload = await self.inbox.get()
            t = meta["t"]
            req = meta["req"]
            if t == "prompt":
                await self._prompt(req, meta, t_recv)
            elif t == "next":
                t0 = time.perf_counter()
                x = self._embed([meta["id"]])
                await self._run(req, x, meta["pos"], True, meta["trace"], t_recv, t0)
            elif t == "act":
                x = from_wire(payload, meta["n"], self.cfg.spec.hidden, meta["dtype"])
                x = x.to(self.cfg.device, self.dtype)
                await self._run(req, x, meta["pos"], meta["last"], meta["trace"], t_recv, None)
            elif t == "release":
                self.kv.pop(req, None)
                self.reqs.pop(req, None)
                if self.cfg.head:
                    await send_frame(self.head_writer, {"t": "release_ack", "req": req})
                else:
                    await self._send_next(meta)
            else:
                raise ValueError(f"{self.cfg.name}: unexpected frame {t!r}")

    def _embed(self, ids: list[int]) -> torch.Tensor:
        return self.stack.embed(torch.tensor(ids, device=self.cfg.device)).to(self.dtype)

    async def _prompt(self, req: str, meta: dict[str, Any], t_recv: float) -> None:
        """The `prompt` frame: open the request, pass the frame on, and on N1 prefill."""
        spec = self.cfg.spec
        ids = meta["ids"]
        self.kv[req] = make_kv(
            spec, self.stack.n_local_layers, meta["max_len"], self.cfg.device, self.dtype
        )
        if self.cfg.head:
            self.reqs[req] = Request(
                prompt_ids=ids,
                max_tokens=meta["max_len"] - len(ids),
                stop_ids=set(meta["stop_ids"]),
                sampler=Sampler(SamplingParams(**meta["sampling"]), device=self.cfg.device),
            )
        else:
            # The link is FIFO, so the next node opens the request before the
            # first `act` of it arrives. Nk does not forward: the ring would
            # bring the frame back to N1.
            await self._send_next(meta)
        if not self.cfg.embed:
            return
        for start in range(0, len(ids), CHUNK):
            chunk = ids[start : start + CHUNK]
            last = start + len(chunk) == len(ids)
            t0 = time.perf_counter()
            x = self._embed(chunk)
            await self._run(req, x, start, last, meta["trace"], t_recv, t0)

    async def _run(
        self,
        req: str,
        x: torch.Tensor,
        pos: int,
        last: bool,
        trace: list,
        t_recv: float,
        t0: float | None,
    ) -> None:
        """Run the local layers on `x`, then pass it on or sample from it.

        `t0` is when compute began. N1 passes the moment before its embedding
        lookup, which PRD 10.2 counts as N1's compute. Other nodes pass None:
        their compute starts here, after the payload became a tensor.
        """
        t0 = time.perf_counter() if t0 is None else t0
        with torch.inference_mode():
            # ponytail: on Nk a non-final prefill chunk only needs its KV rows,
            # but `DecoderStack.forward` always applies `lm_head`. The extra
            # logits are discarded. Ceiling: one wasted matmul per 256 prompt
            # tokens on Nk. Upgrade: a `head=False` flag on `forward`.
            y = self.stack(x, pos_start=pos, cache=self.kv[req], last_only=last)
        if not self.cfg.head:
            entry = self._entry(t_recv, t0)
            meta = {
                "t": "act",
                "req": req,
                "pos": pos,
                "n": x.shape[0],
                "dtype": self.cfg.wire,
                "last": last,
                "trace": [*trace, entry],
            }
            await self._send_next(meta, to_wire(y, self.cfg.wire))
        elif last:
            await self._sample(req, y, pos + x.shape[0], trace, t_recv, t0)

    async def _sample(
        self, req: str, logits: torch.Tensor, pos: int, trace: list, t_recv: float, t0: float
    ) -> None:
        """Nk: pick the token for position `pos`, then send it both ways (PRD 10.2)."""
        r = self.reqs[req]
        with torch.inference_mode():
            tid = r.sampler.sample(logits, seen=[*r.prompt_ids, *r.gen])
        r.gen.append(tid)
        stopped = tid in r.stop_ids
        final = stopped or len(r.gen) >= r.max_tokens
        self._entry(t_recv, t0)  # `token` carries no trace (PRD 8.3); keep the sample
        token: dict[str, Any] = {"t": "token", "req": req, "id": tid, "pos": pos, "final": final}
        if final:
            token["reason"] = "stop" if stopped else "length"
        await send_frame(self.head_writer, token)
        if final:
            await self._send_next({"t": "release", "req": req, "trace": []})
        else:
            await self._send_next({"t": "next", "req": req, "id": tid, "pos": pos, "trace": []})

    def _entry(self, t_recv: float, t0: float) -> dict[str, Any]:
        """One trace entry (PRD 17.2). The compute time stays local, in `stats`."""
        now = time.perf_counter()
        self.stats.append((t_recv, now, now - t0))
        return {"node": self.cfg.name, "t_recv": t_recv, "t_send": now}

    def _dump_and_exit(self) -> None:
        if self.cfg.stats_path:
            Path(self.cfg.stats_path).write_text(json.dumps(self.stats))
        os._exit(0)

    async def _send_next(self, meta: dict[str, Any], payload: bytes = b"") -> None:
        if self.next_writer is None:  # k == 1: the ring is this process (PRD 10.2)
            self.inbox.put_nowait((time.perf_counter(), meta, payload))
        else:
            await send_frame(self.next_writer, meta, payload)


def run_node(cfg: NodeConfig) -> None:
    """Process entry point."""
    try:
        asyncio.run(Node(cfg).serve())
    except Exception:  # noqa: BLE001 - a node prints its traceback, then exits non-zero
        traceback.print_exc()
        sys.exit(1)


# --- the head side ---------------------------------------------------------


@dataclass
class Generation:
    prompt: list[int]
    tokens: list[int] = field(default_factory=list)
    reason: str = ""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def ring_configs(
    spec: ModelSpec,
    ranges: list[tuple[int, int]],
    weights: WeightSource,
    head_port: int,
    stats_dir: str = "",
    **node_kw: Any,
) -> list[NodeConfig]:
    """One config per node. `ranges` must tile `[0, n_layers)` in order."""
    assert ranges[0][0] == 0 and ranges[-1][1] == spec.n_layers
    assert all(a[1] == b[0] for a, b in pairwise(ranges))
    ports = [free_port() for _ in ranges]
    k = len(ranges)
    return [
        NodeConfig(
            name=f"n{i + 1}",
            port=ports[i],
            next_addr="" if k == 1 else f"127.0.0.1:{ports[(i + 1) % k]}",
            head_addr=f"127.0.0.1:{head_port}",
            spec=spec,
            layer_start=start,
            layer_end=end,
            weights=weights,
            stats_path=f"{stats_dir}/n{i + 1}.json" if stats_dir else "",
            **node_kw,
        )
        for i, (start, end) in enumerate(ranges)
    ]


def run_ring(
    spec: ModelSpec,
    ranges: list[tuple[int, int]],
    weights: WeightSource,
    prompts: list[list[int]],
    max_tokens: int,
    *,
    stop_ids: tuple[int, ...] = (),
    sampling: dict[str, Any] | None = None,
    concurrent: bool = False,
    stats: dict[str, list] | None = None,
    **node_kw: Any,
) -> list[Generation]:
    """Start the ring, run every prompt through it, tear it down.

    Pass a dict as `stats` to receive each node's `(t_recv, t_send, compute)`
    samples, keyed by node name.
    """
    head_port = free_port()
    with tempfile.TemporaryDirectory() as stats_dir:
        cfgs = ring_configs(spec, ranges, weights, head_port, stats_dir, **node_kw)
        gens = asyncio.run(
            _drive(cfgs, head_port, prompts, max_tokens, stop_ids, sampling or GREEDY, concurrent)
        )
        if stats is not None:
            for cfg in cfgs:
                path = Path(cfg.stats_path)
                if path.exists():
                    stats[cfg.name] = json.loads(path.read_text())
    return gens


async def _drive(cfgs, head_port, prompts, max_tokens, stop_ids, sampling, concurrent):
    inbox: asyncio.Queue[tuple[dict, bytes]] = asyncio.Queue()

    async def on_link(reader, writer):
        await accept_data_link(reader, writer, CLUSTER)
        try:
            while True:
                inbox.put_nowait(await read_frame(reader))
        except LinkClosed:
            pass
        finally:
            await close_writer(writer)

    server = await asyncio.start_server(on_link, "127.0.0.1", head_port)
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=run_node, args=(c,), daemon=True) for c in cfgs]
    for p in procs:
        p.start()

    def check_alive() -> None:
        dead = [c.name for c, p in zip(cfgs, procs, strict=True) if not p.is_alive()]
        if dead:
            raise RuntimeError(f"node(s) {dead} died; see their traceback on stderr")

    async def recv() -> dict:
        while True:
            try:
                async with asyncio.timeout(5.0):
                    meta, _ = await inbox.get()
                    return meta
            except TimeoutError:
                check_alive()
                if time.perf_counter() - t_wait > FRAME_TIMEOUT:
                    raise RuntimeError(f"no frame from the ring in {FRAME_TIMEOUT:g} s") from None

    writer = None
    try:
        while True:  # dial N1, but notice a node that crashed while loading
            try:
                _, writer = await open_data_link(
                    f"127.0.0.1:{cfgs[0].port}", CLUSTER, total_timeout=5.0, label="N1"
                )
                break
            except TimeoutError:
                check_alive()

        gens = {f"r{i}": Generation(prompt=ids) for i, ids in enumerate(prompts)}
        batches = [list(gens)] if concurrent else [[req] for req in gens]
        for batch in batches:
            for req in batch:
                ids = gens[req].prompt
                await send_frame(
                    writer,
                    {
                        "t": "prompt",
                        "req": req,
                        "ids": ids,
                        "max_len": len(ids) + max_tokens,
                        "sampling": sampling,
                        "stop_ids": list(stop_ids),
                        "trace": [],
                    },
                )
            pending = set(batch)
            t_wait = time.perf_counter()
            while pending:
                meta = await recv()
                t_wait = time.perf_counter()
                g = gens[meta["req"]]
                if meta["t"] == "token":
                    g.tokens.append(meta["id"])
                    assert meta["pos"] == len(g.prompt) + len(g.tokens) - 1
                    if meta["final"]:
                        g.reason = meta["reason"]
                elif meta["t"] == "release_ack":
                    assert g.reason, f"release_ack for {meta['req']} before its final token"
                    pending.discard(meta["req"])
                else:
                    raise RuntimeError(f"head got unexpected frame {meta['t']!r}")
        return list(gens.values())
    finally:
        if writer is not None:
            await close_writer(writer)
        for p in procs:
            p.terminate()
        for p in procs:
            p.join(10)
        server.close()
        await server.wait_closed()


# --- the in-process reference -------------------------------------------


def generate_in_process(
    spec: ModelSpec,
    weights: WeightSource,
    prompts: list[list[int]],
    max_tokens: int,
    *,
    device: str = "cpu",
    compute: str = "fp32",
    storage: str = "fp32",
    max_ctx: int = 512,
) -> list[list[int]]:
    """Plain greedy loop over one full stack. No sockets, no queue, no chunking.

    A ring must match this too. The k = 1 ring runs the same engine loop as
    k = 2; this runs no engine at all, so a mismatch here and not there would
    point at the loop, not the maths.
    """
    cfg = NodeConfig(
        "ref", 0, "", "", spec, 0, spec.n_layers, weights, device, compute, "fp32", max_ctx, storage
    )
    stack = build_stack(cfg)
    dtype = DTYPES[compute]
    out = []
    with torch.inference_mode():
        for ids in prompts:
            kv = make_kv(spec, spec.n_layers, len(ids) + max_tokens, device, dtype)
            x = stack.embed(torch.tensor(ids, device=device)).to(dtype)
            logits = stack(x, pos_start=0, cache=kv, last_only=True)
            gen: list[int] = []
            while len(gen) < max_tokens:
                tid = int(torch.argmax(logits.reshape(-1).to(torch.float32)))
                gen.append(tid)
                if len(gen) == max_tokens:
                    break
                pos = len(ids) + len(gen) - 1
                x = stack.embed(torch.tensor([tid], device=device)).to(dtype)
                logits = stack(x, pos_start=pos, cache=kv, last_only=True)
            out.append(gen)
    return out


def overhead_seconds(samples: list) -> list[float]:
    """Node-side per-frame overhead from one node's samples: wall minus compute.

    That is decode of the meta, bytes to tensor, tensor to bytes, encode of
    the reply, and the queue hop between the reader task and the worker task.
    The socket write itself comes after `t_send` and is not included.
    """
    return [t_send - t_recv - compute for t_recv, t_send, compute in samples]
