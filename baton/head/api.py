"""FastAPI routes, SSE and WebSocket for the head (PRD 13, 14.3).

The request and response models are the contract that the CLI, the dashboard
and every OpenAI client build against. Each route delegates to a `HeadSeam`
that `create_app` attaches to `app.state.driver`. `ClusterApi` is the seam that
`baton serve` uses: it reads the live `Head`.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Annotated, Any, Literal, Protocol

from fastapi import APIRouter, FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from baton.head.driver import Rejected, Sampling

if TYPE_CHECKING:
    from baton.head.serve import Head

__all__ = [
    "ChatCompletionRequest",
    "ClusterSnapshot",
    "CompletionRequest",
    "ErrorCode",
    "HeadSeam",
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
    compute_ms: list[float] = []  # Extension: the last decode steps on this node.


class LiveStats(BaseModel):
    tok_s: float
    ttft_ms: float
    active: int
    queued: int


class ClusterEvent(BaseModel):
    t: float
    kind: Literal["join", "loss", "replan", "error"]
    msg: str


class PlanRow(BaseModel):
    role: str
    name: str
    layers: tuple[int, int]
    weights: float
    kv_budget: float
    predicted_ms: float
    measured_ms: float | None


class PlanInfo(BaseModel):
    """The planner table (PRD 9.9). An agreed extension of the PRD 14.3 snapshot."""

    objective: str
    kv_fraction: float
    guaranteed_ctx: int
    predicted_ttft_ms: float
    predicted_tok_s: float
    rows: list[PlanRow]


class ClusterSnapshot(BaseModel):
    """The one object the dashboard renders. The dashboard does no math."""

    state: Literal["INIT", "PLANNING", "LOADING", "READY", "SERVING", "DEGRADED"]
    model: ModelInfo | None
    plan_rev: int
    nodes: list[NodeSnapshot]
    hops_ms: list[float]  # len(nodes) + 1: head->N1, each hop, Nk->head.
    live: LiveStats
    events: list[ClusterEvent]  # Newest last, capped at 200 (14.2).
    plan: PlanInfo | None = None


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


class HeadSeam(Protocol):
    """What the HTTP layer needs from the head. Attached as `app.state.driver`.

    This is a seam, not a duplicate of `baton.head.driver.Driver`. It exists so the
    surface lane (M3) can be built and tested before the head lane (M2) is finished.
    The concrete backing for each method, which M2 must supply:

    - `generate` and `stream`  -> `driver.Driver.submit` plus `driver.Driver.stream`
    - `snapshot` and `subscribe` -> `registry.Registry.roster` plus `.ring`

    Those concrete methods do NOT exist yet. Wiring them is the first task of M3.
    """

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


def _seam(request: Request) -> HeadSeam:
    return request.app.state.driver


def _reject_unsupported(req: _SamplingRequest) -> None:
    extra = req.model_extra or {}
    for name in UNSUPPORTED_FIELDS:
        if extra.get(name) not in (None, False):
            raise Rejected(f"{name} is not supported", 400)


@router.get("/v1/models", response_model=ModelList)
async def list_models(request: Request) -> ModelList:
    """One entry: the loaded model id. Empty list when not READY (13.1)."""
    snap = _seam(request).snapshot()
    ready = snap.state in ("READY", "SERVING") and snap.model is not None
    return ModelList(data=[ModelCard(id=snap.model.id)] if ready and snap.model else [])


@router.post("/v1/chat/completions", response_model=None)
async def chat_completions(
    req: ChatCompletionRequest, request: Request
) -> dict | StreamingResponse:
    """Chat template applied. `stream: true` returns SSE (13.1, 13.3)."""
    return await _complete(req, _seam(request))


@router.post("/v1/completions", response_model=None)
async def completions(req: CompletionRequest, request: Request) -> dict | StreamingResponse:
    """Raw prompt. Same streaming rules as chat (13.1)."""
    return await _complete(req, _seam(request))


async def _complete(
    req: ChatCompletionRequest | CompletionRequest, seam: HeadSeam
) -> dict | StreamingResponse:
    _reject_unsupported(req)
    if not req.stream:
        return await seam.generate(req)
    chunks = seam.stream(req)
    # The first chunk comes after admission, so a 503 is still a plain HTTP error.
    first = await chunks.__anext__()

    async def events() -> AsyncIterator[str]:
        try:
            yield sse_chunk(first)
            async for chunk in chunks:
                yield sse_chunk(chunk)
        except Rejected as exc:
            yield sse_chunk(error_response(str(exc), exc.code))  # type: ignore[arg-type]
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@router.get("/healthz", response_model=HealthResponse)
async def healthz(request: Request) -> JSONResponse:
    """200 when READY or SERVING, 503 otherwise (13.1)."""
    snap = _seam(request).snapshot()
    ok = snap.state in ("READY", "SERVING")
    body = HealthResponse(
        state=snap.state, model=snap.model.id if snap.model else None, plan_rev=snap.plan_rev
    )
    return JSONResponse(body.model_dump(), status_code=200 if ok else 503)


@router.get("/cluster", response_model=ClusterSnapshot)
async def cluster(request: Request) -> ClusterSnapshot:
    """The snapshot the dashboard reads on load (14.3)."""
    return _seam(request).snapshot()


@router.websocket("/ws")
async def ws(socket: WebSocket) -> None:
    """Push one ClusterSnapshot at 2 Hz. No polling, no client messages (14.1)."""
    await socket.accept()
    try:
        async for snap in socket.app.state.driver.subscribe():
            await socket.send_text(snap.model_dump_json())
    except (WebSocketDisconnect, RuntimeError):
        pass


def sse_chunk(payload: dict) -> str:
    """Encode one SSE frame. `data: [DONE]` closes the stream (13.3)."""
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


def error_response(message: str, code: ErrorCode) -> dict:
    """Build the 13.4 error body."""
    return {"error": {"message": message, "type": "server_error", "code": code}}


class ClusterApi:
    """The seam over a live `Head` (PRD 13, 14.3)."""

    def __init__(self, head: Head) -> None:
        self.head = head

    # --- requests ---

    async def _open(self, req: ChatCompletionRequest | CompletionRequest):
        h = self.head
        driver = h.driver
        if h.state != "ready" or driver is None:
            raise Rejected("the cluster is not ready", 503, "not_ready")
        stop = [req.stop] if isinstance(req.stop, str) else list(req.stop or [])
        sampling = Sampling(
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            seed=req.seed,
            repetition_penalty=req.repetition_penalty,
        )
        chat = isinstance(req, ChatCompletionRequest)
        request = driver.build_request(
            [m.model_dump() for m in req.messages]
            if isinstance(req, ChatCompletionRequest)
            else None,
            None if chat else req.prompt,  # type: ignore[union-attr]
            req.max_tokens,
            sampling,
            stop,
        )
        await driver.submit(request)
        return driver, request, chat

    def _body(self, request, chat: bool, delta: str, finish: str | None, role: bool = False):
        base = {
            "id": ("chatcmpl-" if chat else "cmpl-") + request.req,
            "object": "chat.completion.chunk" if chat else "text_completion",
            "created": int(time.time()),
            "model": self.head.options.model,
        }
        if chat:
            d = {"role": "assistant", "content": ""} if role else {"content": delta}
            choice = {"index": 0, "delta": d, "finish_reason": finish}
        else:
            choice = {"index": 0, "text": delta, "finish_reason": finish}
        return base | {"choices": [choice]}

    async def stream(self, req: ChatCompletionRequest | CompletionRequest):
        driver, request, chat = await self._open(req)
        yield self._body(request, chat, "", None, role=True)
        async for chunk in driver.stream(request):
            body = self._body(request, chat, chunk.text, chunk.finish_reason)
            if chunk.finish_reason is not None:
                body["usage"] = {
                    "prompt_tokens": chunk.prompt_tokens,
                    "completion_tokens": chunk.completion_tokens,
                    "total_tokens": chunk.prompt_tokens + chunk.completion_tokens,
                }
                body["x_baton"] = {
                    "ttft_ms": chunk.ttft_ms,
                    "decode_tok_s": chunk.decode_tok_s,
                    "req": chunk.req,
                }
            yield body

    async def generate(self, req: ChatCompletionRequest | CompletionRequest) -> dict:
        text, last = "", None
        async for body in self.stream(req):
            choice = body["choices"][0]
            text += choice.get("text") or choice.get("delta", {}).get("content", "")
            last = body
        assert last is not None
        chat = isinstance(req, ChatCompletionRequest)
        choice = {"index": 0, "finish_reason": last["choices"][0]["finish_reason"]}
        if chat:
            choice["message"] = {"role": "assistant", "content": text}
        else:
            choice["text"] = text
        return {
            "id": last["id"],
            "object": "chat.completion" if chat else "text_completion",
            "created": last["created"],
            "model": last["model"],
            "choices": [choice],
            "usage": last.get("usage", {}),
            "x_baton": last.get("x_baton", {}),
        }

    # --- cluster ---

    def snapshot(self) -> ClusterSnapshot:
        h = self.head
        driver = h.driver
        state = {"starting": "INIT", "idle": "PLANNING", "loading": "LOADING"}.get(h.state)
        if h.state == "idle" and h.plan_rev:
            state = "DEGRADED"
        if h.state == "ready":
            state = "SERVING" if driver is not None and driver.active else "READY"
        model = None
        per_layer = 0
        if h.metadata is not None:
            spec = h.metadata.spec
            quant = {"bf16": "none"}.get(h.options.quant, h.options.quant)
            model = ModelInfo(
                id=h.options.model, quant=quant, ctx=h.options.ctx, n_layers=spec.n_layers
            )
            per_layer = h.metadata.planner_model(h.options.quant).weight_bytes_per_layer
        plan = h.plan if h.plan is not None and h.plan.feasible else None
        stage = {a.name: a for a in plan.assignments} if plan else {}
        kv_fraction = plan.kv_fraction if plan else 0.0
        usage = driver.admission.usage() if driver is not None else {}

        nodes: list[NodeSnapshot] = []
        ordered = sorted(h.registry, key=lambda w: (w.first_layer is None, w.first_layer or 0))
        ring = 0
        for w in ordered:
            on_ring = w.name in stage and w.state != "lost"
            ring += on_ring
            caps = w.capabilities
            health = w.health
            predicted = stage[w.name].stage_ms if w.name in stage else 0.0
            times = sorted(health.compute_ms) if health else []
            # Measured when the node has run decode steps, else the planner's figure.
            p50 = times[len(times) // 2] if times else predicted
            p95 = times[min(len(times) - 1, int(len(times) * 0.95))] if times else predicted
            nodes.append(
                NodeSnapshot(
                    name=w.name,
                    role=f"N{ring}" if on_ring else "-",
                    backend=caps.backend if caps.backend in ("cuda", "mps", "cpu") else "cpu",
                    link=caps.link if caps.link in ("wired", "wifi") else "wifi",
                    layers=(w.first_layer or 0, w.last_layer or 0),
                    mem=NodeMemory(
                        total=caps.usable_bytes,
                        weights=w.layers * per_layer,
                        kv_used=usage.get(w.name, (health.kv_used_bytes if health else 0, 0))[0],
                        kv_budget=caps.usable_bytes * kv_fraction,
                    ),
                    stage_ms=StageMs(p50=p50, p95=p95),
                    queue_depth=health.queue_depth if health else 0,
                    state=w.state,
                    compute_ms=health.compute_ms if health else [],
                )
            )
        live = LiveStats(
            tok_s=driver.last_tok_s if driver else 0.0,
            ttft_ms=driver.last_ttft_ms if driver else 0.0,
            active=len(driver.active) if driver else 0,
            queued=driver.queued if driver else 0,
        )
        plan_info = None
        if plan is not None and h.metadata is not None:
            plan_info = PlanInfo(
                objective=plan.objective,
                kv_fraction=plan.kv_fraction,
                guaranteed_ctx=plan.guaranteed_ctx,
                predicted_ttft_ms=0.0,
                predicted_tok_s=1000.0 / plan.token_ms if plan.token_ms > 0 else 0.0,
                rows=[
                    PlanRow(
                        role=a.role,
                        name=a.name,
                        layers=(a.first_layer, a.last_layer),
                        weights=a.layers * per_layer,
                        kv_budget=usage.get(a.name, (0, 0))[1],
                        predicted_ms=a.stage_ms,
                        measured_ms=None,
                    )
                    for a in plan.assignments
                ],
            )
        return ClusterSnapshot(
            state=state or "INIT",
            model=model,
            plan_rev=h.plan_rev,
            nodes=nodes,
            hops_ms=[plan.hop_ms if plan else 0.0] * (len(nodes) + 1),
            live=live,
            events=[ClusterEvent(**e) for e in h.events],
            plan=plan_info,
        )

    async def subscribe(self):
        while True:
            yield self.snapshot()
            await asyncio.sleep(0.5)


def create_app(driver: HeadSeam | None = None, dashboard_dist: str | None = None) -> FastAPI:
    """Build the head's ASGI app.

    `dashboard_dist` mounts the Vite build at `/` (13.1). `baton.head.serve`
    passes `dashboard/dist` when it exists.
    """
    app = FastAPI(title="baton", docs_url=None, redoc_url=None)
    app.state.driver = driver
    app.include_router(router)

    @app.exception_handler(BatonError)
    async def _baton_error(_: Request, exc: BatonError) -> JSONResponse:
        return JSONResponse(error_response(exc.message, exc.code), status_code=exc.status_code)

    @app.exception_handler(Rejected)
    async def _rejected(_: Request, exc: Rejected) -> JSONResponse:
        return JSONResponse(error_response(str(exc), exc.code), status_code=exc.status)  # type: ignore[arg-type]

    if dashboard_dist:
        app.mount("/", StaticFiles(directory=dashboard_dist, html=True), name="dashboard")
    return app
