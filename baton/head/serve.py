"""Head node startup sequence. PRD 7.1.

M1 interface only. Every body raises NotImplementedError.

    baton serve --model REPO_ID [--revision main] [--quant bf16|int8|int4]
                [--ctx 8192] [--objective latency|throughput] [--port 7700]
                [--no-local-worker] [--token SECRET] [--hf-token TOKEN]
                [--kv-fraction 0.2]

The nine steps of 7.1 are one method each, so the CLI lane and the tests can
drive them one at a time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from baton.head.driver import Driver
from baton.head.planner import Fleet, Model, Plan
from baton.head.registry import Registry

CONTROL_PORT = 7711
DEFAULT_HTTP_PORT = 7700
MDNS_SERVICE = "_baton._tcp.local."
JOIN_QUIET_S = 10.0
JOIN_DEADLINE_S = 30.0
BENCH_TIMEOUT_S = 60.0

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
    no_local_worker: bool = False
    token: str | None = None
    hf_token: str | None = None
    kv_fraction: float = 0.2
    plan_override: str | None = None


@dataclass
class ModelMetadata:
    """What step 4 fetches and caches.

    `shard_headers` maps a shard file name to the JSON header that the head read
    with an HTTP Range request over the first `8 + header_len` bytes. The head
    never downloads a weight: each worker range-fetches its own layers
    (PRD D9, 12.1).
    """

    config: dict[str, Any]
    tokenizer_config: dict[str, Any]
    index: dict[str, Any]
    shard_headers: dict[str, dict[str, Any]]
    tokenizer: Any = None

    def planner_model(self, quant: Quant) -> Model:
        """The planner's view of this model at the chosen tier (PRD 9.1)."""
        raise NotImplementedError


@dataclass
class Head:
    """The head node process (PRD 7.1).

    `state` follows the CLI status line: starting, idle, loading, ready. An
    infeasible plan leaves the head in `idle` with the shortfall printed.
    """

    options: ServeOptions
    registry: Registry = field(default_factory=Registry)
    state: State = "starting"
    metadata: ModelMetadata | None = None
    plan: Plan | None = None
    driver: Driver | None = None
    cluster_id: str = ""

    async def run(self) -> None:
        """Run steps 1 to 9 in order, then serve until shutdown."""
        raise NotImplementedError

    async def start_servers(self) -> None:
        """Step 1. Control server on 7711, HTTP server on `--port`."""
        raise NotImplementedError

    async def advertise(self) -> None:
        """Step 2. Announce `_baton._tcp.local.` over mDNS.

        The TXT record carries `{version, control_port, cluster_id}`. Workers
        join the first head they see. Two heads on one LAN hold no election in
        v1: each dashboard warns about the other record (PRD 7.2).
        """
        raise NotImplementedError

    async def spawn_local_worker(self) -> None:
        """Step 3. Start a worker subprocess with `--head 127.0.0.1`.

        The head skips this step when `--no-local-worker` is set.
        """
        raise NotImplementedError

    async def fetch_metadata(self) -> ModelMetadata:
        """Step 4. Fetch and cache the model metadata (PRD 7.1, 12.1)."""
        raise NotImplementedError

    async def wait_for_workers(self) -> None:
        """Step 5. Print the roster as workers join.

        Planning starts on the first of three events: every worker seen in the
        last 10 s of mDNS joined, the user pressed Enter, or 30 s passed with at
        least one worker.
        """
        raise NotImplementedError

    async def benchmark(self) -> Fleet:
        """Step 6. Send `bench` to every worker and wait up to 60 s.

        This step also collects the RTT matrix through `ping_peer`: each worker
        pings every other worker once, five samples, median (PRD 9.1).
        """
        raise NotImplementedError

    def make_plan(self, fleet: Fleet) -> Plan:
        """Step 7. Run the planner and print the plan (PRD 9).

        An infeasible fleet prints the shortfall and leaves the head in `idle`.
        `--plan` skips the solver but keeps feasibility, ring order and roles.
        """
        raise NotImplementedError

    async def load_plan(self, plan: Plan) -> None:
        """Step 8. Send `load` to every node in the plan and wait for `loaded`."""
        raise NotImplementedError

    def print_urls(self) -> None:
        """Step 9. Print the API URL and the dashboard URL."""
        raise NotImplementedError

    async def replan(self, lost: str) -> Plan:
        """Re-plan after a worker loss (PRD 7.6, 11).

        A device whose new range overlaps its old one fetches only the missing
        layers (PRD 9.7, 5.4).
        """
        raise NotImplementedError


async def serve(options: ServeOptions) -> None:
    """Entry point for `baton serve`. Builds a Head and runs it."""
    raise NotImplementedError
