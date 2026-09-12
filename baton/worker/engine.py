"""Forward engine: KV cache and the compute thread (PRD 6.5, 6.6).

One instance per worker process. It owns the device, the resident shard, and
every KV tensor. It runs exactly one compute thread, which handles one frame at
a time in arrival order, so no lock protects the model itself.

The control loop never calls into the model. It only puts jobs on `inbox` and
takes finished frames off `outbox`. See the design note in memory.md.
"""

from __future__ import annotations

import queue
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol

if TYPE_CHECKING:  # transport lane (PRD 8); imported for types only
    import torch

    from baton.common.messages import Frame

JobKind = Literal["prompt", "next", "act", "release", "abort", "bench"]

#: Prefill chunk width in tokens, constant by decision D10 (PRD 10.1).
CHUNK_TOKENS = 256

#: Bytes per token per layer of KV cache, both K and V, in a 2-byte compute
#: dtype: 2 tensors * n_kv_heads * head_dim * 2 B. 4 KB is the PRD 6.5 figure
#: for the 70B shape and is used only for the pre-allocation estimate.
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


@dataclass(slots=True)
class Job:
    """One unit of work for the compute thread.

    A job is either a data-plane frame to run through the local layers, or a
    lifecycle instruction. `frame` is None only for `bench`.
    """

    kind: JobKind
    req: str
    frame: Frame | None = None
    t_recv_ns: int = 0
    """`time.perf_counter_ns()` at receive, for the trace (PRD 6.6, 17.2)."""


@dataclass(slots=True)
class KVEntry:
    """The one KV tensor for one request on this worker.

    Shape `[n_local_layers, 2, max_len, n_kv_heads, head_dim]` in compute
    dtype, allocated whole on the first frame of the request so a later frame
    can never fail midway through a generation.
    """

    req: str
    tensor: torch.Tensor
    max_len: int
    charged_bytes: int
    """Bytes debited from the engine budget when this entry was allocated."""

    seq_len: int = 0
    """Rows written so far. The next frame writes at this offset."""


class ShardRunner(Protocol):
    """What the engine needs from the model lane (PRD 5.2), and nothing more.

    Declared as a Protocol so this module does not import the kernels lane and
    the two can be written in parallel.
    """

    n_local_layers: int
    is_first: bool
    """True when this worker holds layer 0 and therefore embeds (N1)."""

    is_last: bool
    """True when this worker holds the final layer and therefore samples (Nk)."""

    def embed(self, ids: torch.Tensor) -> torch.Tensor: ...

    def run_layers(self, x: torch.Tensor, kv: torch.Tensor, pos: int) -> torch.Tensor: ...

    def sample(self, x: torch.Tensor, sampling: dict[str, object]) -> int: ...


class KVPool:
    """Charges KV cache memory to a per-worker budget (PRD 6.5).

    Every request is charged at allocation, for its whole `max_len`, and
    credited on release. Nothing is charged incrementally as a sequence grows,
    because admission control on the head (PRD 7.5) reserves against the same
    worst case. Charging late would let two admitted requests both fit at
    admission and both fail at token 2000.
    """

    def __init__(self, budget_bytes: int, n_local_layers: int, bytes_per_token: int) -> None:
        self._budget_bytes = budget_bytes
        self._n_local_layers = n_local_layers
        self._bytes_per_token = bytes_per_token
        self._entries: dict[str, KVEntry] = {}

    @property
    def used_bytes(self) -> int:
        """Sum of charges for live requests. Reported as `kv_used` in health."""
        raise NotImplementedError

    @property
    def free_bytes(self) -> int:
        """Budget minus charges. Not a device query: the caching allocator
        keeps freed blocks in its pool, so the device would under-report."""
        raise NotImplementedError

    def cost_bytes(self, max_len: int) -> int:
        """Bytes this worker will charge for a request of `max_len` tokens."""
        raise NotImplementedError

    def allocate(self, req: str, prompt_len: int, max_tokens: int, ctx_max: int) -> KVEntry:
        """Allocate the KV tensor on the first frame of a request.

        `max_len = min(prompt_len + max_tokens, ctx_max)`. Raises
        `OutOfMemoryOnKV` when the charge exceeds the remaining budget, or when
        the device allocation itself fails. Idempotent for a `req` already
        present: a retried first frame must not double charge.
        """
        raise NotImplementedError

    def get(self, req: str) -> KVEntry | None:
        """Return the live entry, or None when the request is unknown."""
        raise NotImplementedError

    def release(self, req: str) -> int:
        """Delete the tensor and credit the charge. Returns bytes credited.

        Zero for an unknown request: `release` rings the whole cluster and a
        node that already aborted must not treat the second visit as an error.
        The memory returns to the allocator pool, not to the OS (PRD 6.5).
        """
        raise NotImplementedError


class ForwardEngine:
    """The compute thread and its two queues (PRD 6.6).

    `inbox` is a plain `queue.Queue` written by the control loop. `outbox`
    carries finished frames back. The control loop is asyncio and the compute
    thread is not, so the thread hands each frame over with
    `loop.call_soon_threadsafe`, set up by `attach_loop`.
    """

    def __init__(self, runner: ShardRunner, kv: KVPool, device: str) -> None:
        self._runner = runner
        self._kv = kv
        self._device = device
        self.inbox: queue.Queue[Job | None] = queue.Queue()
        self._stopping = False

    def attach_loop(self, on_frame: object) -> None:
        """Register the control loop's thread-safe frame sink.

        `on_frame` is called from the compute thread for every outbound frame.
        It must not block: the implementation wraps
        `loop.call_soon_threadsafe(queue.put_nowait, frame)`.
        """
        raise NotImplementedError

    def start(self) -> None:
        """Start the single compute thread. Idempotent."""
        raise NotImplementedError

    def stop(self, timeout_s: float = 5.0) -> None:
        """Drain `inbox`, stop the thread, free every KV entry.

        Puts the `None` sentinel rather than setting a flag, so the thread
        wakes from a blocking `get()` without a poll timeout.
        """
        raise NotImplementedError

    def submit(self, job: Job) -> None:
        """Hand one job to the compute thread. Called from the control loop."""
        raise NotImplementedError

    @property
    def queue_depth(self) -> int:
        """Jobs waiting. Reported in health and used by head admission."""
        raise NotImplementedError

    @property
    def active_reqs(self) -> int:
        """Requests with a live KV entry on this worker."""
        raise NotImplementedError

    # --- compute thread internals; never called from the control loop ---

    def _run(self) -> None:
        """The compute thread body: one blocking `get`, one frame, repeat."""
        raise NotImplementedError

    def _handle(self, job: Job) -> None:
        """Dispatch one job by kind (PRD 6.6).

        - `prompt` (N1): ids to 256-token chunks, embed each, run layers, emit
          one `act` frame per chunk.
        - `next` (N1): embed one id, run layers, emit one `act` frame.
        - `act`: view the payload as compute dtype, run layers, emit `act` to
          the next node. On Nk instead: norm, lm_head on the last row, sample,
          emit `token` to the head and `next` to N1.
        - `release` / `abort`: free the KV entry, then forward the frame on.
        """
        raise NotImplementedError

    def _stamp_trace(self, job: Job, frame: Frame) -> None:
        """Append `{node, t_recv}` and add `t_send` to that entry.

        Timestamps are `perf_counter_ns()` on this node only. Absolute clocks
        are never compared across nodes (PRD 17.2).
        """
        raise NotImplementedError
