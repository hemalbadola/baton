"""Request lifecycle and admission control on the head. PRD 7.3 and 7.5.

M1 interface only. Every body raises NotImplementedError.

The head is off the token loop (PRD D4). It sends one `prompt` frame to N1,
then reads `token` frames from Nk and turns them into SSE chunks. No tensor
ever reaches the head.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from baton.head.detok import IncrementalDetokenizer
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


@dataclass
class Chunk:
    """One SSE chunk. The API lane (PRD 13) renders it in OpenAI shape."""

    req: str
    text: str
    finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


class Rejected(Exception):
    """The request cannot run. Carries the HTTP status the API lane must send."""

    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class Admission:
    """Per-node KV budget and the requests that hold it (PRD 7.5).

    The budget of node i is
    `usable_bytes - resident_weight_bytes - activation_workspace`, where
    `activation_workspace = 256 * intermediate * 2 B * 3`. A request needs
    `n_local_layers_i * 4 KB * max_len` on node i.
    """

    plan: Plan
    kv_budget_bytes: dict[str, int] = field(default_factory=dict)
    held_bytes: dict[str, dict[str, int]] = field(default_factory=dict)

    def cost_bytes(self, max_len: int) -> dict[str, int]:
        """KV bytes that a request of `max_len` tokens needs on every node."""
        raise NotImplementedError

    def fits(self, max_len: int) -> bool:
        """True when the request fits on every node at once (PRD 7.5)."""
        raise NotImplementedError

    def hold(self, req: str, max_len: int) -> None:
        """Charge the request to every node. The caller checks `fits` first."""
        raise NotImplementedError

    def release(self, req: str) -> None:
        """Free the request on every node.

        The driver calls this on `release_ack`, or `RELEASE_GRACE_S` after an
        abort, whichever comes first (PRD 7.3).
        """
        raise NotImplementedError

    def usage(self) -> dict[str, tuple[int, int]]:
        """`(kv_used, kv_budget)` per node, for the dashboard (PRD 7.5)."""
        raise NotImplementedError


@dataclass
class Driver:
    """Drives every request around the ring (PRD 7.3).

    The `tokenizer` annotation is Any because the model lane owns the loading of
    it. `registry` gives the driver N1 and Nk. `admission` guards the KV budget.
    """

    registry: Registry
    admission: Admission
    tokenizer: Any
    ctx: int
    active: dict[str, Request] = field(default_factory=dict)

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
        raise NotImplementedError

    async def submit(self, request: Request) -> None:
        """Admit the request and send the `prompt` frame to N1 (steps 4 and 5).

        Queues FIFO when the KV budget is full, and waits at most
        `QUEUE_WAIT_CAP_S`. Raises Rejected with status 503 on that timeout.
        """
        raise NotImplementedError

    async def stream(self, request: Request) -> AsyncIterator[Chunk]:
        """Yield one chunk per token frame from Nk (steps 6 and 7).

        Detokenizes incrementally (PRD 7.4). On a multi-token stop string, sends
        `abort{req}` to N1 and yields a final chunk with `finish_reason="stop"`.
        It emits no text after the stop string.
        """
        raise NotImplementedError

    async def abort(self, req: str, reason: str) -> None:
        """Send `abort{req}` to N1 and start the release grace timer.

        The head aborts on a stop-string hit, a client disconnect, or a
        wall-clock timeout (PRD 10.4).
        """
        raise NotImplementedError

    def on_token(self, req: str, token_id: int, pos: int, final: bool, reason: str) -> None:
        """Handle one `token` frame from Nk (PRD 10.2)."""
        raise NotImplementedError

    def on_release_ack(self, req: str) -> None:
        """Handle `release_ack` from Nk: every node freed the KV (PRD 7.3).

        `release` travels the full ring, Nk to N1 to N2 and on to Nk. When Nk
        sees its own `release` return, it sends this over its data socket.
        """
        raise NotImplementedError
