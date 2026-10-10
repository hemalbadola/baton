"""Shard loading, KV cache and the forward step (PRD 5.3, 6.5, 6.6).

`ForwardEngine.handle` is synchronous: one frame in, a list of frames out. The
daemon calls it through `asyncio.to_thread` from one consumer task, so frames run
one at a time in arrival order and the heartbeat never waits on the model.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import torch

from baton.worker.sampler import Sampler, SamplingParams

if TYPE_CHECKING:  # imported for types only
    from baton.model.layers import DecoderStack
    from baton.model.safetensors_io import ByteSource
    from baton.model.spec import ModelSpec

#: Prefill chunk width in tokens, constant by decision D10 (PRD 10.1).
CHUNK_TOKENS = 256

#: Bytes per token per layer of KV cache, both K and V, in a 2-byte compute
#: dtype: 2 tensors * n_kv_heads * head_dim * 2 B. 4 KB is the PRD 6.5 figure
#: for the 70B shape and is used only as a default estimate.
KV_BYTES_PER_LAYER_PER_TOKEN = 4 * 1024


class OutOfMemoryOnKV(Exception):
    """Raised when the KV tensor for a request does not fit.

    The daemon turns this into `error{req, code="oom"}` and drops the request.
    The head answers the client with HTTP 503 (PRD 6.5, 11.1).
    """

    def __init__(self, req: str, requested_bytes: int, available_bytes: int) -> None:
        super().__init__(f"kv alloc {requested_bytes} B for {req}, {available_bytes} B free")
        self.req = req
        self.requested_bytes = requested_bytes
        self.available_bytes = available_bytes


class ShardTooLarge(Exception):
    """The shard does not fit the memory budget. The daemon reports `error{code="oom"}`."""


def load_shard(
    spec: ModelSpec,
    first_layer: int,
    last_layer: int,
    *,
    embed: bool,
    head: bool,
    source: ByteSource,
    index: Mapping[str, str],
    ctx_max: int,
    device: str,
    dtype: torch.dtype,
    budget_bytes: int | None = None,
    on_progress: Callable[[int, int, int], None] | None = None,
) -> tuple[DecoderStack, int]:
    """Fetch one layer range and build the resident stack (PRD 5.3, 6.4 steps 2-3).

    `[first_layer, last_layer]` is inclusive, as it travels in `load`. One
    tensor at a time: each is fetched, checked, copied into its place and
    dropped, so peak memory is the stack plus the largest tensor. Returns the
    stack and its resident bytes.

    `on_progress(done, total, bytes)` is called from this thread after every
    tensor. The size is checked against `budget_bytes` before anything is
    allocated, because a laptop that swaps is worse than one that says no.

    ponytail: dense weights in the compute dtype, fetched on every load. The
    quantized tiers are BAT-10; the shard cache and the memmapped embedding
    table are BAT-11.
    """
    from baton.model import safetensors_io as sio
    from baton.model.layers import DecoderStack

    names = spec.range_names(first_layer, last_layer + 1, embed=embed, head=head)
    missing = [name for name in names if name not in index]
    if missing:
        raise KeyError(f"index names no file for {missing[0]!r} ({len(missing)} missing)")
    headers = {file: sio.fetch_header(source, file) for file in sorted({index[n] for n in names})}
    refs = sio.resolve(index, headers, names)

    need = sum(ref.numel() for ref in refs) * torch.empty((), dtype=dtype).element_size()
    if budget_bytes is not None and need > budget_bytes:
        raise ShardTooLarge(f"shard needs {need} bytes, the budget is {budget_bytes}")

    stack = DecoderStack(
        spec,
        first_layer,
        last_layer + 1,
        embed=embed,
        head=head,
        max_ctx=ctx_max,
        device=device,
        dtype=dtype,
    )
    state = stack.state_dict()
    slots: dict[str, list[torch.Tensor]] = {}
    for key, name in stack.checkpoint_keys().items():
        slots.setdefault(name, []).append(state[key])

    done = fetched = 0
    with torch.no_grad():
        # `max_gap=-1` turns coalescing off. Tensors sit back to back in a shard
        # file, so any gap allowance merges the whole layer range into one
        # multi-GB read held beside the stack (PRD 5.3 step 5).
        for name, tensor in sio.iter_tensors(refs, source, max_gap=-1):
            for slot in slots[name]:
                # `copy_` broadcasts, so a wrong-shaped tensor would load silently.
                if slot.shape != tensor.shape:
                    raise ValueError(
                        f"{name}: checkpoint shape {tuple(tensor.shape)}, "
                        f"model wants {tuple(slot.shape)}"
                    )
                slot.copy_(tensor)
            done += 1
            fetched += tensor.numel() * tensor.element_size()
            if on_progress is not None:
                on_progress(done, len(refs), fetched)

    resident = sum(t.numel() * t.element_size() for t in (*stack.parameters(), *stack.buffers()))
    return stack.eval(), resident


_WIRE = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def to_wire(x: torch.Tensor, wire: str) -> bytes:
    return x.to("cpu", _WIRE[wire]).contiguous().view(torch.uint8).numpy().tobytes()


def from_wire(payload: bytes, n: int, hidden: int, wire: str) -> torch.Tensor:
    # `bytearray` because `torch.frombuffer` wants a writable buffer.
    return torch.frombuffer(bytearray(payload), dtype=_WIRE[wire]).view(n, hidden)


class LayerKV:
    """One layer's K and V rows. Satisfies `layers.LayerKVCache`."""

    def __init__(self, max_len: int, n_kv_heads: int, head_dim: int, device: str, dtype: Any):
        shape = (max_len, n_kv_heads, head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)

    def append(self, k: torch.Tensor, v: torch.Tensor, pos_start: int):
        end = pos_start + k.shape[0]
        self.k[pos_start:end] = k
        self.v[pos_start:end] = v
        return self.k[:end], self.v[:end]


@dataclass(slots=True)
class KVEntry:
    """The KV rows of one request on this worker, allocated whole on its first frame
    so that a later frame can never fail midway through a generation."""

    req: str
    layers: list[LayerKV]
    max_len: int
    charged_bytes: int


class KVPool:
    """Charges KV memory to a per-worker budget (PRD 6.5).

    Every request is charged for its whole `max_len` at allocation and credited
    on release. Charging late would let two admitted requests both fit at
    admission and both fail at token 2000.

    `bytes_per_token` counts one layer, K and V together.
    """

    def __init__(self, budget_bytes: int, n_local_layers: int, bytes_per_token: int) -> None:
        self._budget_bytes = budget_bytes
        self._n_local_layers = n_local_layers
        self._bytes_per_token = bytes_per_token
        self._entries: dict[str, KVEntry] = {}

    @property
    def used_bytes(self) -> int:
        return sum(e.charged_bytes for e in self._entries.values())

    @property
    def free_bytes(self) -> int:
        return self._budget_bytes - self.used_bytes

    def __len__(self) -> int:
        return len(self._entries)

    def cost_bytes(self, max_len: int) -> int:
        return self._n_local_layers * self._bytes_per_token * max_len

    def allocate(
        self, req: str, max_len: int, n_kv_heads: int, head_dim: int, device: str, dtype: Any
    ) -> KVEntry:
        """Allocate on the first frame of a request. Idempotent for a known `req`.

        Raises `OutOfMemoryOnKV` when the charge exceeds the remaining budget or
        the device allocation itself fails. A failure charges nothing.
        """
        known = self._entries.get(req)
        if known is not None:
            return known
        cost = self.cost_bytes(max_len)
        if cost > self.free_bytes:
            raise OutOfMemoryOnKV(req, cost, self.free_bytes)
        try:
            layers = [
                LayerKV(max_len, n_kv_heads, head_dim, device, dtype)
                for _ in range(self._n_local_layers)
            ]
        except (RuntimeError, MemoryError) as exc:
            raise OutOfMemoryOnKV(req, cost, self.free_bytes) from exc
        entry = self._entries[req] = KVEntry(req, layers, max_len, cost)
        return entry

    def get(self, req: str) -> KVEntry | None:
        return self._entries.get(req)

    def release(self, req: str) -> int:
        """Credit the charge. Zero for an unknown request: `release` rings the
        whole cluster, and a node that already aborted sees it a second time."""
        entry = self._entries.pop(req, None)
        return entry.charged_bytes if entry else 0


@dataclass(slots=True)
class Out:
    """One frame the engine wants sent. `to` is `next` (the ring link) or `head`."""

    to: Literal["next", "head"]
    meta: dict[str, Any]
    payload: bytes = b""


@dataclass(slots=True)
class _Gen:
    """Per-request state on Nk (PRD 10.3)."""

    prompt_ids: list[int]
    max_tokens: int
    stop_ids: set[int]
    sampler: Sampler
    penalised: bool
    gen: list[int] = field(default_factory=list)


class ForwardEngine:
    """Runs this worker's layers for one frame at a time (PRD 6.6, 10).

    Frames in: `prompt`, `act`, `next`, `release`, `abort`. Frames out: `act`
    to the next node, `token` and `release_ack` to the head, `next` to N1.
    With one node the ring is this engine, and the caller feeds `next` back.
    """

    def __init__(
        self,
        stack: DecoderStack,
        kv: KVPool,
        *,
        device: str,
        dtype: Any,
        wire: str = "bf16",
        ctx_max: int,
    ) -> None:
        self.stack = stack
        self.kv = kv
        self.device = device
        self.dtype = dtype
        self.wire = wire
        self.ctx_max = ctx_max
        self.is_first = stack.embed_tokens is not None
        self.is_last = stack.lm_head is not None
        self._gens: dict[str, _Gen] = {}

    @property
    def active_reqs(self) -> int:
        return len(self.kv)

    def drop(self, req: str) -> None:
        """Forget a request after a failure. Safe for an unknown request."""
        self.kv.release(req)
        self._gens.pop(req, None)

    def handle(self, meta: dict[str, Any], payload: bytes = b"") -> list[Out]:
        kind, req = meta["t"], meta["req"]
        with torch.inference_mode():
            if kind == "prompt":
                return self._prompt(req, meta)
            if kind == "next":
                if self.kv.get(req) is None:  # aborted while the frame was in flight
                    return []
                x = self._embed([int(meta["id"])])
                return self._run(req, x, int(meta["pos"]), True)
            if kind == "act":
                if self.kv.get(req) is None:
                    return []
                x = from_wire(payload, int(meta["n"]), self.stack.spec.hidden, meta["dtype"])
                return self._run(req, x.to(self.device, self.dtype), int(meta["pos"]), meta["last"])
            if kind in ("release", "abort"):
                self.drop(req)
                if self.is_last:
                    return [Out("head", {"t": "release_ack", "req": req})]
                return [Out("next", meta)]
        raise ValueError(f"unexpected frame {kind!r}")

    def _embed(self, ids: list[int]) -> torch.Tensor:
        return self.stack.embed(torch.tensor(ids, device=self.device)).to(self.dtype)

    def _prompt(self, req: str, meta: dict[str, Any]) -> list[Out]:
        """Open the request here and pass the frame on, so that Nk learns
        `max_len`, `sampling` and `stop_ids` before the first `act` arrives. The
        link is FIFO. Nk does not forward: the ring would bring it back to N1."""
        spec = self.stack.spec
        ids = [int(i) for i in meta["ids"]]
        max_len = min(int(meta["max_len"]), self.ctx_max)
        self.kv.allocate(req, max_len, spec.n_kv_heads, spec.head_dim, self.device, self.dtype)
        outs: list[Out] = []
        if self.is_last:
            params = SamplingParams(**meta.get("sampling", {}))
            self._gens[req] = _Gen(
                prompt_ids=ids,
                max_tokens=max_len - len(ids),
                stop_ids={int(i) for i in meta.get("stop_ids", [])},
                sampler=Sampler(params, device=self.device),
                penalised=params.repetition_penalty != 1.0,
            )
        else:
            outs.append(Out("next", meta))
        if self.is_first:
            for start in range(0, len(ids), CHUNK_TOKENS):
                chunk = ids[start : start + CHUNK_TOKENS]
                last = start + len(chunk) == len(ids)
                outs += self._run(req, self._embed(chunk), start, last)
        return outs

    def _run(self, req: str, x: torch.Tensor, pos: int, last: bool) -> list[Out]:
        entry = self.kv.get(req)
        assert entry is not None
        # `last_only` matters on Nk, where a 256-row chunk needs one logits row.
        y = self.stack(x, pos_start=pos, cache=entry.layers, last_only=True)
        if not self.is_last:
            meta = {
                "t": "act",
                "req": req,
                "pos": pos,
                "n": x.shape[0],
                "dtype": self.wire,
                "last": last,
            }
            return [Out("next", meta, to_wire(y, self.wire))]
        return self._sample(req, y, pos + x.shape[0]) if last else []

    def _sample(self, req: str, logits: torch.Tensor, pos: int) -> list[Out]:
        """Nk: pick the token for position `pos`, then send it both ways (PRD 10.2)."""
        r = self._gens[req]
        seen = [*r.prompt_ids, *r.gen] if r.penalised else ()
        tid = r.sampler.sample(logits, seen=seen)
        r.gen.append(tid)
        stopped = tid in r.stop_ids
        final = stopped or len(r.gen) >= r.max_tokens
        token: dict[str, Any] = {"t": "token", "req": req, "id": tid, "pos": pos, "final": final}
        if final:
            token["reason"] = "stop" if stopped else "length"
            return [Out("head", token), Out("next", {"t": "release", "req": req})]
        return [Out("head", token), Out("next", {"t": "next", "req": req, "id": tid, "pos": pos})]
