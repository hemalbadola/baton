"""`baton app`: one window per laptop (BAT-41).

Every laptop runs this small agent and opens its page. The agents find each
other over mDNS and show a list of nearby laptops. One person creates a room and
invites laptops. Each invited person clicks Accept. The host clicks Start, and
the agents run `baton serve` and `baton worker` themselves. Nobody types a
command after the install.

The agent never runs a command it received. A peer can only ask for three
things, each guarded: an invite (shown to the user), an accept (answers an invite
that this laptop sent), and a join (only for an invite that the user accepted,
and only from the laptop that sent it).
"""

from __future__ import annotations

import asyncio
import contextlib
import secrets
import socket
import sys
from collections import deque
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from baton.common.net import CONTROL_PORT, lan_addresses

AGENT_PORT = 7800
HEAD_HTTP_PORT = 7700
NODE_SERVICE = "_baton-node._tcp.local."
PEER_TIMEOUT_S = 0.8

#: Models the page offers. Sizes are bf16 on disk. Only `--quant none` loads (BAT-10).
MODELS = [
    {"id": "Qwen/Qwen2.5-0.5B-Instruct", "name": "Qwen 2.5 0.5B", "gb": 1.0},
    {"id": "unsloth/Llama-3.2-1B-Instruct", "name": "Llama 3.2 1B", "gb": 2.5},
    {"id": "Qwen/Qwen2.5-1.5B-Instruct", "name": "Qwen 2.5 1.5B", "gb": 3.1},
    {"id": "Qwen/Qwen2.5-3B-Instruct", "name": "Qwen 2.5 3B", "gb": 6.2},
    {"id": "unsloth/Llama-3.2-3B-Instruct", "name": "Llama 3.2 3B", "gb": 6.4},
]

Spawn = Callable[[list[str]], Coroutine[Any, Any, Any]]


@dataclass
class Invite:
    """An invite this laptop received."""

    id: str
    from_name: str
    from_addr: str  # "ip:port" of the host's agent
    host_ip: str
    control_port: int
    model: str
    status: str = "pending"  # pending | accepted | declined


@dataclass
class Room:
    """The room this laptop hosts."""

    model: str
    invited: dict[str, str] = field(default_factory=dict)  # peer name -> invite id
    accepted: dict[str, dict[str, str]] = field(default_factory=dict)  # name -> {addr, id}


class Agent:
    """State: `idle`, `lobby` (host, before Start), `hosting`, `joined`."""

    def __init__(
        self,
        name: str,
        port: int = AGENT_PORT,
        *,
        spawn: Spawn | None = None,
        head_http: int = HEAD_HTTP_PORT,
        control_port: int = CONTROL_PORT,
    ) -> None:
        self.name = name
        self.port = port
        self.head_http = head_http
        self.control_port = control_port
        self.state = "idle"
        self.room: Room | None = None
        self.invites: dict[str, Invite] = {}
        self.joined_invite: Invite | None = None
        self.peers: dict[str, dict[str, Any]] = {}  # mDNS service name -> {name, addrs}
        self.log: deque[str] = deque(maxlen=300)
        self._spawn = spawn or self._spawn_process
        self._proc: Any = None
        self._watch: asyncio.Task[None] | None = None
        self._zc: Any = None
        self._browser: Any = None
        self._tasks: set[asyncio.Task[None]] = set()

    # --- subprocess -------------------------------------------------------------

    async def _spawn_process(self, argv: list[str]) -> asyncio.subprocess.Process:
        return await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "baton.cli",
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

    async def _run(self, argv: list[str], state: str) -> None:
        await self._stop_process()
        self.log.append(f"$ baton {' '.join(argv)}")
        self._proc = await self._spawn(argv)
        self.state = state
        self._watch = asyncio.create_task(self._pump(self._proc))

    async def _pump(self, proc: Any) -> None:
        """Keep the child's output for the page, and fall back to idle when it ends."""
        stream = getattr(proc, "stdout", None)
        if stream is not None:
            async for line in stream:
                self.log.append(line.decode(errors="replace").rstrip())
        code = await proc.wait()
        if self._proc is proc:
            self._proc = None
            self.log.append(f"stopped (exit {code})")
            if self.state in ("hosting", "joined"):
                self.state, self.room, self.joined_invite = "idle", None, None

    async def _stop_process(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None or proc.returncode is not None:
            return
        proc.terminate()
        try:
            async with asyncio.timeout(10):
                await proc.wait()
        except TimeoutError:
            proc.kill()
            await proc.wait()

    # --- nearby laptops ---------------------------------------------------------

    async def start_discovery(self) -> None:
        from zeroconf import ServiceInfo, ServiceStateChange
        from zeroconf.asyncio import AsyncServiceBrowser, AsyncServiceInfo, AsyncZeroconf

        self._zc = AsyncZeroconf()
        info = ServiceInfo(
            NODE_SERVICE,
            f"{self.name}.{NODE_SERVICE}",
            addresses=[socket.inet_aton(a) for a in lan_addresses()],
            port=self.port,
            properties={"name": self.name},
            server=f"baton-node-{self.name}.local.",
        )
        await self._zc.async_register_service(info, allow_name_change=True)

        async def resolve(service: str, change: Any) -> None:
            if change is ServiceStateChange.Removed:
                self.peers.pop(service, None)
                return
            found = AsyncServiceInfo(NODE_SERVICE, service)
            if not await found.async_request(self._zc.zeroconf, 3000) or found.port is None:
                return
            name = (found.properties.get(b"name") or b"").decode()
            if name and name != self.name:
                hosts = found.parsed_addresses()
                self.peers[service] = {"name": name, "addrs": [f"{h}:{found.port}" for h in hosts]}

        def on_change(zeroconf: Any, service_type: str, name: str, state_change: Any) -> None:
            task = asyncio.ensure_future(resolve(name, state_change))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

        self._browser = AsyncServiceBrowser(self._zc.zeroconf, NODE_SERVICE, handlers=[on_change])

    async def nearby(self) -> list[dict[str, Any]]:
        """Ask every known peer for its state. A peer that does not answer is dropped."""
        async with httpx.AsyncClient(timeout=PEER_TIMEOUT_S) as client:

            async def ask(peer: dict[str, Any]) -> dict[str, Any] | None:
                for addr in peer["addrs"]:
                    try:
                        r = await client.get(f"http://{addr}/api/me")
                        me = r.json()
                    except (httpx.HTTPError, ValueError):
                        continue
                    return {"name": me["name"], "addr": addr, "state": me["state"]}
                return None

            found = await asyncio.gather(*(ask(p) for p in list(self.peers.values())))
        unique = {p["name"]: p for p in found if p}  # one laptop, many interfaces
        return sorted(unique.values(), key=lambda p: p["name"])

    async def stop_discovery(self) -> None:
        if self._browser is not None:
            await self._browser.async_cancel()
        if self._zc is not None:
            with contextlib.suppress(Exception):
                await self._zc.async_unregister_all_services()
            await self._zc.async_close()

    # --- the page's API ---------------------------------------------------------

    def me(self) -> dict[str, Any]:
        from baton.worker.probe import memory_budget, pick_device

        backend = pick_device("auto")
        free = memory_budget(backend, None).usable_bytes
        room = self.room
        return {
            "name": self.name,
            "state": self.state,
            "backend": backend,
            "usable_gb": round(free / 1024**3, 1),
            "models": MODELS,
            "room": room
            and {
                "model": room.model,
                "invited": sorted(set(room.invited) - set(room.accepted)),
                "accepted": sorted(room.accepted),
            },
            "invites": [
                {"id": i.id, "from": i.from_name, "model": i.model}
                for i in self.invites.values()
                if i.status == "pending"
            ],
            "joined": self.joined_invite and {"host": self.joined_invite.from_name},
            "log": list(self.log)[-40:],
        }

    def _head_base(self) -> str | None:
        if self.state == "hosting":
            return f"http://127.0.0.1:{self.head_http}"
        if self.state == "joined" and self.joined_invite is not None:
            return f"http://{self.joined_invite.host_ip}:{self.head_http}"
        return None

    def app(self) -> FastAPI:
        api = FastAPI(docs_url=None, redoc_url=None)
        page = Path(__file__).with_name("agent_ui.html")

        def bad(message: str, status: int = 400) -> JSONResponse:
            return JSONResponse({"error": message}, status_code=status)

        @api.get("/", response_class=HTMLResponse)
        async def index() -> str:
            return page.read_text()

        @api.get("/api/me")
        async def me() -> dict[str, Any]:
            return self.me()

        @api.get("/api/peers")
        async def peers() -> list[dict[str, Any]]:
            return await self.nearby()

        # -- host --

        @api.post("/api/room")
        async def create_room(body: dict[str, Any]) -> Any:
            if self.state not in ("idle", "lobby"):
                return bad(f"this laptop is {self.state}")
            model = str(body.get("model", "")).strip()
            if not model:
                return bad("pick a model")
            self.room = Room(model)
            self.state = "lobby"
            return {"ok": True}

        @api.post("/api/invite")
        async def invite(body: dict[str, Any]) -> Any:
            if self.state != "lobby" or self.room is None:
                return bad("create a room first")
            peers = {p["name"]: p for p in await self.nearby()}
            peer = peers.get(str(body.get("peer")))
            if peer is None:
                return bad("that laptop is not nearby any more", 404)
            invite_id = secrets.token_hex(6)
            payload = {
                "id": invite_id,
                "from_name": self.name,
                "from_port": self.port,
                "control_port": self.control_port,
                "model": self.room.model,
            }
            try:
                async with httpx.AsyncClient(timeout=3) as client:
                    r = await client.post(f"http://{peer['addr']}/api/invited", json=payload)
                    r.raise_for_status()
            except httpx.HTTPError as exc:
                return bad(f"could not reach {peer['name']}: {exc}", 502)
            self.room.invited[peer["name"]] = invite_id
            self.room.accepted.pop(peer["name"], None)
            return {"ok": True}

        @api.post("/api/accepted")
        async def accepted(body: dict[str, Any], request: Request) -> Any:
            room = self.room
            name = str(body.get("name"))
            if room is None or room.invited.get(name) != body.get("id"):
                return bad("no such invite", 404)
            host = request.client.host if request.client else ""
            room.accepted[name] = {"addr": f"{host}", "id": str(body["id"])}
            return {"ok": True}

        @api.post("/api/start")
        async def start() -> Any:
            if self.state != "lobby" or self.room is None:
                return bad("create a room first")
            room = self.room
            await self._run(
                [
                    "serve",
                    "--model",
                    room.model,
                    "--quant",
                    "none",
                    "--min-workers",
                    str(1 + len(room.accepted)),
                    "--wait",
                    "180",
                    "--port",
                    str(self.head_http),
                    "--control-port",
                    str(self.control_port),
                ],
                "hosting",
            )
            async with httpx.AsyncClient(timeout=5) as client:
                for name, who in room.accepted.items():
                    peer = {p["name"]: p for p in await self.nearby()}.get(name)
                    if peer is None:
                        self.log.append(f"{name} is not reachable; the cluster starts without it")
                        continue
                    try:
                        await client.post(f"http://{peer['addr']}/api/join", json={"id": who["id"]})
                    except httpx.HTTPError as exc:
                        self.log.append(f"could not tell {name} to join: {exc}")
            return {"ok": True}

        @api.post("/api/stop")
        async def stop() -> Any:
            room, self.room = self.room, None
            invite, self.joined_invite = self.joined_invite, None
            await self._stop_process()
            self.state = "idle"
            if room is not None:
                peers = {p["name"]: p for p in await self.nearby()}
                async with httpx.AsyncClient(timeout=3) as client:
                    for name, who in room.invited.items():
                        if name in peers:
                            with contextlib.suppress(httpx.HTTPError):
                                await client.post(
                                    f"http://{peers[name]['addr']}/api/leave", json={"id": who}
                                )
            if invite is not None:
                invite.status = "declined"
            return {"ok": True}

        # -- invited laptop --

        @api.post("/api/invited")
        async def invited(body: dict[str, Any], request: Request) -> Any:
            host = request.client.host if request.client else ""
            invite = Invite(
                id=str(body["id"]),
                from_name=str(body["from_name"]),
                from_addr=f"{host}:{int(body['from_port'])}",
                host_ip=host,
                control_port=int(body["control_port"]),
                model=str(body["model"]),
            )
            self.invites[invite.id] = invite
            return {"ok": True}

        @api.post("/api/accept")
        async def accept(body: dict[str, Any]) -> Any:
            invite = self.invites.get(str(body.get("id")))
            if invite is None or invite.status != "pending":
                return bad("no such invite", 404)
            if self.state != "idle":
                return bad(f"this laptop is {self.state}")
            try:
                async with httpx.AsyncClient(timeout=3) as client:
                    r = await client.post(
                        f"http://{invite.from_addr}/api/accepted",
                        json={"id": invite.id, "name": self.name},
                    )
                    r.raise_for_status()
            except httpx.HTTPError as exc:
                return bad(f"could not reach {invite.from_name}: {exc}", 502)
            invite.status = "accepted"
            return {"ok": True}

        @api.post("/api/decline")
        async def decline(body: dict[str, Any]) -> Any:
            invite = self.invites.get(str(body.get("id")))
            if invite is not None:
                invite.status = "declined"
            return {"ok": True}

        @api.post("/api/join")
        async def join(body: dict[str, Any], request: Request) -> Any:
            """The host says "start". Only for an invite that the user accepted."""
            invite = self.invites.get(str(body.get("id")))
            host = request.client.host if request.client else ""
            if invite is None or invite.status != "accepted" or invite.host_ip != host:
                return bad("no accepted invite from you", 403)
            self.joined_invite = invite
            await self._run(
                [
                    "worker",
                    "--head",
                    f"{invite.host_ip}:{invite.control_port}",
                    "--name",
                    self.name,
                ],
                "joined",
            )
            return {"ok": True}

        @api.post("/api/leave")
        async def leave(body: dict[str, Any], request: Request) -> Any:
            invite = self.joined_invite
            host = request.client.host if request.client else ""
            if invite is None or invite.id != body.get("id") or invite.host_ip != host:
                return bad("not joined to you", 403)
            self.joined_invite = None
            await self._stop_process()
            self.state = "idle"
            return {"ok": True}

        # -- the cluster's own API, so the page needs no second port --

        async def proxy(request: Request, path: str) -> Any:
            base = self._head_base()
            if base is None:
                return JSONResponse(
                    {"error": {"message": "no cluster yet", "code": "not_ready"}}, 503
                )
            client = httpx.AsyncClient(timeout=None)
            upstream = client.build_request(
                request.method,
                f"{base}{path}",
                content=await request.body(),
                headers={"content-type": request.headers.get("content-type", "application/json")},
            )
            try:
                response = await client.send(upstream, stream=True)
            except httpx.HTTPError:
                await client.aclose()
                return JSONResponse(
                    {"error": {"message": "the head does not answer", "code": "not_ready"}}, 503
                )

            async def body() -> Any:
                try:
                    async for chunk in response.aiter_raw():
                        yield chunk
                finally:  # a client that leaves closes the upstream and aborts the ring
                    await response.aclose()
                    await client.aclose()

            return StreamingResponse(
                body(),
                status_code=response.status_code,
                media_type=response.headers.get("content-type"),
            )

        @api.get("/cluster")
        async def cluster(request: Request) -> Any:
            return await proxy(request, "/cluster")

        @api.api_route("/v1/{rest:path}", methods=["GET", "POST"])
        async def v1(rest: str, request: Request) -> Any:
            return await proxy(request, f"/v1/{rest}")

        return api


async def run_agent(
    name: str | None = None, port: int = AGENT_PORT, open_browser: bool = True
) -> None:
    """Serve the page, advertise this laptop, and run until cancelled."""
    import uvicorn

    agent = Agent(name or socket.gethostname().split(".")[0], port)
    server = uvicorn.Server(
        uvicorn.Config(agent.app(), host="0.0.0.0", port=port, log_level="warning")
    )
    server.install_signal_handlers = lambda: None
    serving = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)
    await agent.start_discovery()
    url = f"http://127.0.0.1:{port}"
    print(f"baton: {agent.name} is ready. Open {url}", flush=True)
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    try:
        await serving
    finally:
        await agent._stop_process()
        await agent.stop_discovery()
