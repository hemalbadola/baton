"""Request lifecycle and admission control on the head. PRD 7.3 and 7.5.

The head is off the token loop (PRD D4). It sends one `prompt` frame to N1,
then reads `token` frames from Nk and turns them into SSE chunks. No tensor
ever reaches the head.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from baton.common.net import send_frame
from baton.head.detok import IncrementalDetokenizer, single_token_stop_ids
from baton.head.planner import Plan
from baton.head.registry import Registry

QUEUE_WAIT_CAP_S = 120.0
RELEASE_GRACE_S = 5.0
ACTIVATION_WORKSPACE_ROWS = 256
KV_BYTES_PER_TOKEN_PER_LAYER = 4096


@dataclass
class Sampling:
    """Sampling parameters. They reach Nk once, in the `prompt` frame (PRD 10.3)."""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    seed: int | None = None
    repetition_penalty: float = 1.0


@dataclass
class Request:
    """One in-flight request (PRD 7.3).

    `req` is `uuid4().hex[:12]`. `max_len` is `len(ids) + max_tokens`, capped at
    the context length.
    """

    req: str
    ids: list[int]
    max_len: int
    sampling: Sampling
    stop_ids: set[int]
    stop_strings: Sequence[str] = ()
    detok: IncrementalDetokenizer | None = None
    generated: list[int] = field(default_factory=list)
    finish_reason: str | None = None
    t_submit: float = 0.0
    t_first: float = 0.0
    t_last: float = 0.0


@dataclass
class Chunk:
    """One SSE chunk. The API lane (PRD 13) renders it in OpenAI shape."""

    req: str
    text: str
    finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    ttft_ms: float = 0.0
    decode_tok_s: float = 0.0


class Rejected(Exception):
    """The request cannot run. Carries the HTTP status the API lane must send.

    `code` is one of the `ErrorCode` names of PRD 13.4, or `invalid_request`.
    """

    def __init__(self, message: str, status: int, code: str = "invalid_request") -> None:
        super().__init__(message)
        self.status = status
        self.code = code


@dataclass
class Admission:
    """Per-node KV budget and the requests that hold it (PRD 7.5).

    `per_token` is the KV bytes of one layer for one token on each node. It
    differs by node because a CPU worker may compute in fp32. A request needs
    `layers_i * per_token_i * max_len` on node i.
    """

    plan: Plan
    kv_budget_bytes: dict[str, int] = field(default_factory=dict)
    held_bytes: dict[str, dict[str, int]] = field(default_factory=dict)
    per_token: dict[str, int] = field(default_factory=dict)

    def cost_bytes(self, max_len: int) -> dict[str, int]:
        """KV bytes that a request of `max_len` tokens needs on every node."""
        return {
            a.name: a.layers * self.per_token.get(a.name, KV_BYTES_PER_TOKEN_PER_LAYER) * max_len
            for a in self.plan.assignments
        }

    def fits(self, max_len: int) -> bool:
        """True when the request fits on every node at once (PRD 7.5)."""
        used = self.usage()
        return all(
            used[name][0] + cost <= used[name][1] for name, cost in self.cost_bytes(max_len).items()
        )

    def hold(self, req: str, max_len: int) -> None:
        """Charge the request to every node. The caller checks `fits` first."""
        self.held_bytes[req] = self.cost_bytes(max_len)

    def release(self, req: str) -> None:
        """Free the request on every node.

        The driver calls this on `release_ack`, or `RELEASE_GRACE_S` after an
        abort, whichever comes first (PRD 7.3).
        """
        self.held_bytes.pop(req, None)

    def usage(self) -> dict[str, tuple[int, int]]:
        """`(kv_used, kv_budget)` per node, for the dashboard (PRD 7.5)."""
        return {
            a.name: (
                sum(held.get(a.name, 0) for held in self.held_bytes.values()),
                self.kv_budget_bytes.get(a.name, 0),
            )
            for a in self.plan.assignments
        }


@dataclass
class Driver:
    """Drives every request around the ring (PRD 7.3).

    The `tokenizer` annotation is Any because the model lane owns the loading of
    it. `registry` gives the driver N1 and Nk. `admission` guards the KV budget.
    `eos_ids` stop every request, besides the ones that its own `stop` names.
    """

    registry: Registry
    admission: Admission
    tokenizer: Any
    ctx: int
    eos_ids: frozenset[int] = frozenset()
    active: dict[str, Request] = field(default_factory=dict)
    last_ttft_ms: float = 0.0
    last_tok_s: float = 0.0
    queued: int = 0

    def __post_init__(self) -> None:
        self._inbox: dict[str, asyncio.Queue[tuple[Any, ...]]] = {}
        self._room = asyncio.Condition()
        self._aborted: set[str] = set()

    def build_request(
        self,
        messages: list[dict[str, str]] | None,
        prompt: str | None,
        max_tokens: int,
        sampling: Sampling,
        stop: Sequence[str] = (),
    ) -> Request:
        """Steps 1 to 3 of PRD 7.3.

        Applies the chat template for `/v1/chat/completions`, or encodes the raw
        prompt for `/v1/completions`. Raises Rejected with status 400 when the
        prompt is longer than `ctx - 1`.
        """
        tok = self.tokenizer
        if messages is not None:
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            ids = list(tok.encode(text, add_special_tokens=False))
        else:
            ids = list(tok.encode(prompt or ""))
        if not ids or len(ids) > self.ctx - 1:
            raise Rejected(f"prompt has {len(ids)} tokens, the context is {self.ctx}", 400)
        return Request(
            req=uuid.uuid4().hex[:12],
            ids=ids,
            max_len=min(len(ids) + max_tokens, self.ctx),
            sampling=sampling,
            stop_ids=set(self.eos_ids) | single_token_stop_ids(tok, stop),
            stop_strings=tuple(stop),
            detok=IncrementalDetokenizer(tok, tuple(stop)),
        )

    async def submit(self, request: Request) -> None:
        """Admit the request and send the `prompt` frame to N1 (steps 4 and 5).

        Queues FIFO when the KV budget is full, and waits at most
        `QUEUE_WAIT_CAP_S`. Raises Rejected with status 503 on that timeout.
        """
        ring = self.admission.plan.ring
        n1 = self.registry.get(ring[0]) if ring else None
        if n1 is None or n1.state != "loaded":
            raise Rejected("the cluster is not ready", 503, "not_ready")
        request.t_submit = time.perf_counter()
        self.queued += 1
        try:
            async with asyncio.timeout(QUEUE_WAIT_CAP_S):
                async with self._room:
                    await self._room.wait_for(lambda: self.admission.fits(request.max_len))
                    # Held under the lock: two woken waiters must not both fit.
                    self.admission.hold(request.req, request.max_len)
        except TimeoutError:
            raise Rejected("no KV room within the queue limit", 503, "queue_timeout") from None
        finally:
            self.queued -= 1
        self.active[request.req] = request
        self._inbox[request.req] = asyncio.Queue()
        sampling = {
            "temperature": request.sampling.temperature,
            "top_p": request.sampling.top_p,
            "top_k": request.sampling.top_k,
            "repetition_penalty": request.sampling.repetition_penalty,
        }
        if request.sampling.seed is not None:
            sampling["seed"] = request.sampling.seed
        frame = {
            "t": "prompt",
            "req": request.req,
            "ids": request.ids,
            "max_len": request.max_len,
            "sampling": sampling,
            "stop_ids": sorted(request.stop_ids),
            "trace": [],
        }
        try:
            await send_frame(n1.control, frame)
        except OSError:
            self._finish(request.req)
            raise Rejected("the first node is gone", 503, "worker_lost") from None

    async def stream(self, request: Request) -> AsyncIterator[Chunk]:
        """Yield one chunk per token frame from Nk (steps 6 and 7).

        Detokenizes incrementally (PRD 7.4). On a multi-token stop string, sends
        `abort{req}` to N1 and yields a final chunk with `finish_reason="stop"`.
        It emits no text after the stop string.
        """
        inbox = self._inbox[request.req]
        detok = request.detok
        assert detok is not None
        done = False
        try:
            while True:
                kind, *rest = await inbox.get()
                if kind == "error":
                    code, message = rest
                    done = True
                    await self.abort(request.req, code)
                    raise Rejected(message, 503, code)
                token_id, _pos, final, reason = rest
                now = time.perf_counter()
                if not request.generated:
                    request.t_first = now
                request.t_last = now
                request.generated.append(token_id)
                text = ""
                stopped = reason == "stop"
                emission_stopped = False
                if not stopped:  # a stop id is not text
                    emission = detok.push(token_id)
                    text = emission.text
                    if emission.finished:
                        stopped = emission_stopped = True
                        await self.abort(request.req, "stop")
                if stopped or final:
                    done = True
                    if not emission_stopped:
                        text += detok.flush()
                    request.finish_reason = "stop" if stopped else "length"
                    if not self._aborted & {request.req}:  # `abort` set its own timer
                        loop = asyncio.get_running_loop()
                        loop.call_later(RELEASE_GRACE_S, self._finish, request.req)
                    yield self._chunk(request, text, request.finish_reason)
                    return
                yield self._chunk(request, text)
        finally:
            if not done and request.req in self.active:  # client left, or cancelled
                await self.abort(request.req, "abort")

    def _chunk(self, request: Request, text: str, finish: str | None = None) -> Chunk:
        n = len(request.generated)
        ttft = (request.t_first - request.t_submit) * 1000
        span = request.t_last - request.t_first
        rate = (n - 1) / span if span > 0 else 0.0
        if finish is not None:
            self.last_ttft_ms, self.last_tok_s = ttft, rate
        return Chunk(request.req, text, finish, len(request.ids), n, ttft, rate)

    async def abort(self, req: str, reason: str) -> None:
        """Send `abort{req}` to N1 and start the release grace timer.

        The head aborts on a stop-string hit, a client disconnect, or a
        wall-clock timeout (PRD 10.4).
        """
        if req in self._aborted or req not in self.active:
            return
        self._aborted.add(req)
        ring = self.admission.plan.ring
        n1 = self.registry.get(ring[0]) if ring else None
        if n1 is not None and n1.state != "lost":
            try:
                await send_frame(n1.control, {"t": "abort", "req": req})
            except OSError:
                self._finish(req)
                return
        asyncio.get_running_loop().call_later(RELEASE_GRACE_S, self._finish, req)

    def on_token(self, req: str, token_id: int, pos: int, final: bool, reason: str) -> None:
        """Handle one `token` frame from Nk (PRD 10.2)."""
        inbox = self._inbox.get(req)
        if inbox is not None and req not in self._aborted:
            inbox.put_nowait(("token", token_id, pos, final, reason))

    def on_error(self, req: str, code: str, message: str) -> None:
        """A worker failed this request. The stream raises, then the ring is aborted."""
        inbox = self._inbox.get(req)
        if inbox is not None and req not in self._aborted:
            inbox.put_nowait(("error", "oom" if code == "oom" else "worker_lost", message))

    def fail_all(self, code: str, message: str) -> None:
        """The ring broke. Nothing will answer, so every stream ends with an error."""
        for req in list(self._inbox):
            self.on_error(req, code, message)
        # The workers that are left hear nothing, so their KV frees on re-plan.
        for req in list(self.active):
            self._finish(req)

    def on_release_ack(self, req: str) -> None:
        """Handle `release_ack` from Nk: every node freed the KV (PRD 7.3).

        `release` travels the full ring, Nk to N1 to N2 and on to Nk. When Nk
        sees its own `release` return, it sends this over its data socket.
        """
        self._finish(req)

    def _finish(self, req: str) -> None:
        self.active.pop(req, None)
        self._inbox.pop(req, None)
        self._aborted.discard(req)
        self.admission.release(req)
        asyncio.ensure_future(self._wake())

    async def _wake(self) -> None:
        async with self._room:
            self._room.notify_all()
