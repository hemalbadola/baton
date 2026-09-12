"""One TypedDict per wire message (PRD 8.3).

Every JSON meta carries `t`, the message type. Request-scoped messages also
carry `req`, the request id. Control-plane messages have no payload. On the data
plane only `act` has one: `n x hidden` values in the wire dtype (PRD 8.4).

These types are static only. They cost nothing at runtime, because a TypedDict
is a plain `dict`. For runtime work use `message_type()` and `MESSAGE_TYPES`.
"""

# No `from __future__ import annotations` here, on purpose. PEP 563 turns every
# annotation into a string, and a TypedDict then cannot see `NotRequired`, so
# `__required_keys__` comes out wrong. Runtime code reads those sets.
from typing import Any, Literal, NotRequired, TypedDict

__all__ = [
    "CONTROL_DOWN_TYPES",
    "CONTROL_UP_TYPES",
    "DATA_TYPES",
    "MESSAGE_TYPES",
    "Abort",
    "Act",
    "Bench",
    "BenchResult",
    "ControlDown",
    "ControlUp",
    "DataPlane",
    "Error",
    "Health",
    "Hello",
    "LinkDown",
    "Load",
    "LoadProgress",
    "Loaded",
    "Message",
    "Next",
    "PingPeer",
    "PongPeer",
    "Prompt",
    "Release",
    "ReleaseAck",
    "Sampling",
    "Token",
    "Unload",
    "Welcome",
    "message_type",
]


class Sampling(TypedDict, total=False):
    """Sampling knobs carried on `prompt`. Absent keys take the head default."""

    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    seed: int


# --- control plane: worker -> head -------------------------------------------


class Hello(TypedDict):
    """Sent once after the control socket opens. `token` must match the head's."""

    t: Literal["hello"]
    name: str
    token: str
    version: str
    caps: dict[str, Any]  # PRD 6.2 capability report; the worker lane fills it.
    data_addr: str  # "ip:port" the peer ring dials.


class Health(TypedDict):
    """Every 2 s. Six seconds of silence marks the worker lost (PRD 8.6)."""

    t: Literal["health"]
    mem_free: int
    kv_used: int
    queue_depth: int
    active_reqs: int
    loaded_rev: NotRequired[int | None]


class BenchResult(TypedDict):
    """Reply to `bench`. Feeds the planner's cost model."""

    t: Literal["bench_result"]
    t_dec_ms: float
    t_pre_ms: float
    quant: str
    dtype: str


class LoadProgress(TypedDict):
    """Emitted while a shard loads, about every 500 ms, for the dashboard bar."""

    t: Literal["load_progress"]
    done: int  # tensors finished
    total: int  # tensors in this shard
    bytes: int  # bytes fetched so far


class Loaded(TypedDict):
    """Shard is resident and the data links are open. Reply to `load`."""

    t: Literal["loaded"]
    rev: int  # plan revision this load belongs to
    resident_bytes: int
    seconds: float
    from_cache: bool


class LinkDown(TypedDict):
    """The data socket to `peer` closed. The head re-plans (PRD 11)."""

    t: Literal["link_down"]
    peer: str


class Error(TypedDict):
    """Any failure. `req` is present when the failure belongs to one request."""

    t: Literal["error"]
    code: str
    message: str
    req: NotRequired[str]


class Token(TypedDict):
    """Nk -> head on the data socket. One sampled token."""

    t: Literal["token"]
    req: str
    id: int
    pos: int
    final: bool
    reason: NotRequired[str]  # "stop" | "length" | "abort" on the final token


class ReleaseAck(TypedDict):
    """Nk -> head on the data socket, after its own `release` returns."""

    t: Literal["release_ack"]
    req: str


class PongPeer(TypedDict):
    """Reply to `ping_peer`. One cell of the planner's RTT matrix."""

    t: Literal["pong_peer"]
    peer: str
    rtt_ms: float


# --- control plane: head -> worker -------------------------------------------


class Welcome(TypedDict):
    """Reply to `hello`. The worker learns the name the head will use for it."""

    t: Literal["welcome"]
    cluster_id: str
    your_name: str
    plan_rev: NotRequired[int]


class Bench(TypedDict):
    """Run the micro-benchmark of PRD 6.3. Sent before planning."""

    t: Literal["bench"]
    spec: dict[str, Any]
    quant: str
    dtype: str


class Load(TypedDict):
    """Load one shard (PRD 6.4), fetched by byte range from Hugging Face.

    `index` is this node's position in the ring, zero-based: index 0 is N1.
    `headers` are the HTTP headers for the range fetch. `hf_token` is sent apart
    from `headers` so the head can rotate it without rebuilding the request.
    """

    t: Literal["load"]
    plan_rev: int
    model: str
    range: list[int]  # [first_layer, last_layer], inclusive
    quant: str
    roles: dict[str, bool]  # {"embed": bool, "head": bool}
    ctx_max: int
    kv_budget_bytes: int
    next_node: str  # "ip:port" of Ni+1, or "" for Nk
    head_data_addr: str
    index: int
    headers: dict[str, str]
    hf_token: NotRequired[str]


class Unload(TypedDict):
    """Free the resident shard. Sent on re-plan and on shutdown."""

    t: Literal["unload"]


class PingPeer(TypedDict):
    """Measure the round trip to `peer_addr`. The worker replies `pong_peer`."""

    t: Literal["ping_peer"]
    peer_addr: str


# --- data plane: the ring -----------------------------------------------------


class Prompt(TypedDict):
    """head -> N1. Starts a request. No payload."""

    t: Literal["prompt"]
    req: str
    ids: list[int]
    max_len: int
    sampling: Sampling
    stop_ids: list[int]
    trace: list[Any]


class Next(TypedDict):
    """Nk -> N1. Carries the sampled token back for the next step. No payload."""

    t: Literal["next"]
    req: str
    id: int
    pos: int
    trace: list[Any]


class Act(TypedDict):
    """Ni -> Ni+1. The only message with a payload: `n x hidden` in `dtype`."""

    t: Literal["act"]
    req: str
    pos: int
    n: int
    dtype: str  # wire dtype, "bf16" by default (PRD 8.4)
    trace: list[Any]


class Release(TypedDict):
    """Nk -> N1 -> ... -> Nk. Frees the KV block for `req` around the whole ring."""

    t: Literal["release"]
    req: str
    trace: list[Any]


class Abort(TypedDict):
    """Cancel `req`. The head sends it to N1 only, and it rings around."""

    t: Literal["abort"]
    req: str


ControlUp = (
    Hello
    | Health
    | BenchResult
    | LoadProgress
    | Loaded
    | LinkDown
    | Error
    | Token
    | ReleaseAck
    | PongPeer
)
ControlDown = Welcome | Bench | Load | Unload | PingPeer | Abort
DataPlane = Prompt | Next | Act | Release | Abort
Message = ControlUp | ControlDown | DataPlane

CONTROL_UP_TYPES: frozenset[str] = frozenset(
    {
        "hello",
        "health",
        "bench_result",
        "load_progress",
        "loaded",
        "link_down",
        "error",
        "token",
        "release_ack",
        "pong_peer",
    }
)
CONTROL_DOWN_TYPES: frozenset[str] = frozenset(
    {"welcome", "bench", "load", "unload", "ping_peer", "abort"}
)
DATA_TYPES: frozenset[str] = frozenset({"prompt", "next", "act", "release", "abort"})
MESSAGE_TYPES: frozenset[str] = CONTROL_UP_TYPES | CONTROL_DOWN_TYPES | DATA_TYPES


def message_type(meta: dict[str, Any]) -> str:
    """Return the `t` of a decoded meta, or raise `ValueError` if it is unknown.

    Call this on every frame that arrives. An unknown `t` is a version skew or a
    stray client, and the caller must answer with `error`, not guess.

    Raise `TypeError` if `t` is absent or is not a string, `ValueError` if it is
    a string this build does not know.
    """
    t = meta.get("t")
    if not isinstance(t, str):
        raise TypeError(f"meta has no string 't', got {t!r}")
    if t not in MESSAGE_TYPES:
        raise ValueError(f"unknown message type {t!r}")
    return t
