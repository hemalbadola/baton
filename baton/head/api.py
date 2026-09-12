"""FastAPI routes, SSE and WebSocket for the head (PRD 13, 14.3).

Interfaces only. The request and response models are real, because they are the
contract that the CLI, the dashboard and every OpenAI client build against. The
route bodies raise NotImplementedError until the driver (7.3) lands; each route
delegates to a `Driver` that `baton.head.serve` attaches to `app.state.driver`.
"""

from __future__ import annotations

import time
from typing import Annotated, Any, Literal, Protocol

from fastapi import APIRouter, FastAPI, WebSocket
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "ChatCompletionRequest",
    "ClusterSnapshot",
    "CompletionRequest",
    "Driver",
    "ErrorCode",
    "create_app",
    "router",
]

# 13.4. The HTTP status each error code maps to.
ErrorCode = Literal["worker_lost", "not_ready", "oom", "queue_timeout", "timeout"]
ERROR_STATUS: dict[str, int] = {
    "not_ready": 503,
    "worker_lost": 503,
    "oom": 503,
    "queue_timeout": 503,
    "timeout": 504,
}

# 13.1. Fields a client can send that we reject outright with 400.
UNSUPPORTED_FIELDS = ("functions", "response_format", "logprobs")


class BatonError(Exception):
    """Raised by the driver. The exception handler turns it into 13.4 JSON."""

    def __init__(self, message: str, code: ErrorCode) -> None:
        super().__init__(message)
        self.message = message
        self.code = code

    @property
    def status_code(self) -> int:
        return ERROR_STATUS[self.code]


# --------------------------------------------------------------------------- #
# 13.2  Request models
# --------------------------------------------------------------------------- #


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class _SamplingRequest(BaseModel):
    """Fields shared by /v1/chat/completions and /v1/completions (13.2)."""

    model_config = ConfigDict(extra="allow")

    model: str  # Ignored, but must be present for client compatibility.
    max_tokens: Annotated[int, Field(ge=1)] = 1024
    temperature: Annotated[float, Field(ge=0.0, le=2.0)] = 0.7
    top_p: Annotated[float, Field(gt=0.0, le=1.0)] = 1.0
    top_k: Annotated[int, Field(ge=0)] = 0  # Extension.
    repetition_penalty: Annotated[float, Field(gt=0.0)] = 1.0  # Extension.
    seed: int | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    n: Annotated[int, Field(ge=1, le=1)] = 1  # n > 1 is 400 (13.2).
    tools: list[Any] | None = None  # Accepted only when empty.

    @field_validator("stop")
    @classmethod
    def _at_most_four_stops(cls, v: str | list[str] | None) -> str | list[str] | None:
        if isinstance(v, list) and len(v) > 4:
            raise ValueError("stop accepts at most 4 sequences")
        return v

    @field_validator("tools")
    @classmethod
    def _tools_must_be_empty(cls, v: list[Any] | None) -> list[Any] | None:
        if v:
            raise ValueError("tools are not supported")
        return v


class ChatCompletionRequest(_SamplingRequest):
    messages: Annotated[list[ChatMessage], Field(min_length=1)]


class CompletionRequest(_SamplingRequest):
    prompt: str


# --------------------------------------------------------------------------- #
# 14.3  Cluster snapshot. GET /cluster and every /ws message share this schema.
# --------------------------------------------------------------------------- #


class ModelInfo(BaseModel):
    id: str
    quant: Literal["none", "int8", "int4"]
    ctx: int
    n_layers: int


class NodeMemory(BaseModel):
    total: float
    weights: float
    kv_used: float
    kv_budget: float


class StageMs(BaseModel):
    p50: float
    p95: float


class NodeSnapshot(BaseModel):
    name: str
    role: str  # "N1", "N2", ... in layer order.
    backend: Literal["cuda", "mps", "cpu"]
    link: Literal["wired", "wifi"]
    layers: tuple[int, int]
    mem: NodeMemory
    stage_ms: StageMs
    queue_depth: int
    state: Literal["joining", "loading", "loaded", "standby", "lost"]


class LiveStats(BaseModel):
    tok_s: float
    ttft_ms: float
    active: int
    queued: int


class ClusterEvent(BaseModel):
    t: float
    kind: Literal["join", "loss", "replan", "error"]
    msg: str


class ClusterSnapshot(BaseModel):
    """The one object the dashboard renders. The dashboard does no math."""

    state: Literal["INIT", "PLANNING", "LOADING", "READY", "SERVING", "DEGRADED"]
    model: ModelInfo | None
    plan_rev: int
    nodes: list[NodeSnapshot]
    hops_ms: list[float]  # len(nodes) + 1: head->N1, each hop, Nk->head.
    live: LiveStats
    events: list[ClusterEvent]  # Newest last, capped at 200 (14.2).


class HealthResponse(BaseModel):
    state: str
    model: str | None
    plan_rev: int


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "baton"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]


# --------------------------------------------------------------------------- #
# Driver seam
# --------------------------------------------------------------------------- #


class Driver(Protocol):
    """What `baton.head.driver` must provide. Attached as `app.state.driver`."""

    def snapshot(self) -> ClusterSnapshot:
        """Current cluster state (14.3)."""

    async def generate(self, req: ChatCompletionRequest | CompletionRequest) -> dict:
        """Run one request to completion. Raises BatonError on failure."""

    async def stream(self, req: ChatCompletionRequest | CompletionRequest):
        """Yield OpenAI SSE chunk dicts, one per token (13.3).

        The final chunk carries `finish_reason`, a `usage` object with
        `prompt_tokens` and `completion_tokens`, and the extension `x_baton`
        with `ttft_ms`, `decode_tok_s` and `req`.
        """

    async def subscribe(self):
        """Yield a ClusterSnapshot at 2 Hz for /ws (14.1)."""


router = APIRouter()


@router.get("/v1/models", response_model=ModelList)
async def list_models() -> ModelList:
    """One entry: the loaded model id. Empty list when not READY (13.1)."""
    raise NotImplementedError("api: /v1/models")


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(req: ChatCompletionRequest) -> dict | StreamingResponse:
    """Chat template applied. `stream: true` returns SSE (13.1, 13.3)."""
    raise NotImplementedError("api: /v1/chat/completions")


@router.post("/v1/completions", response_model=None)
async def completions(req: CompletionRequest) -> dict | StreamingResponse:
    """Raw prompt. Same streaming rules as chat (13.1)."""
    raise NotImplementedError("api: /v1/completions")


@router.get("/healthz", response_model=HealthResponse)
async def healthz() -> HealthResponse:
    """200 when READY or SERVING, 503 otherwise (13.1)."""
    raise NotImplementedError("api: /healthz")


@router.get("/cluster", response_model=ClusterSnapshot)
async def cluster() -> ClusterSnapshot:
    """The snapshot the dashboard reads on load (14.3)."""
    raise NotImplementedError("api: /cluster")


@router.websocket("/ws")
async def ws(socket: WebSocket) -> None:
    """Push one ClusterSnapshot at 2 Hz. No polling, no client messages (14.1)."""
    raise NotImplementedError("api: /ws")


def sse_chunk(payload: dict) -> str:
    """Encode one SSE frame. `data: [DONE]` closes the stream (13.3)."""
    raise NotImplementedError("api: SSE encoder")


def error_response(message: str, code: ErrorCode) -> dict:
    """Build the 13.4 error body."""
    return {"error": {"message": message, "type": "server_error", "code": code}}


def create_app(driver: Driver | None = None, dashboard_dist: str | None = None) -> FastAPI:
    """Build the head's ASGI app.

    `dashboard_dist` mounts the Vite build at `/` (13.1). `baton.head.serve`
    passes `dashboard/dist` unless `--no-dashboard` was given.
    """
    raise NotImplementedError("api: create_app")
