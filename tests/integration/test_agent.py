"""Two laptops, one room, no terminal (BAT-41).

Two agents on one machine find each other over mDNS. One hosts and invites, the
other accepts, and the host starts. The real `baton serve` and `baton worker`
are replaced by a recorder, so the test checks what the agents would run.

Sockets and multicast need the sandbox off.
"""

import asyncio
import contextlib
import socket

import httpx
import uvicorn

from baton.agent import Agent


class FakeProc:
    returncode = None
    stdout = None

    def __init__(self) -> None:
        self._done = asyncio.Event()

    def terminate(self) -> None:
        self.returncode = 0
        self._done.set()

    kill = terminate

    async def wait(self) -> int:
        await self._done.wait()
        return 0


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def until(done, timeout=20.0) -> None:
    async with asyncio.timeout(timeout):
        while not await done():
            await asyncio.sleep(0.1)


@contextlib.asynccontextmanager
async def laptop(name: str):
    ran: list[list[str]] = []

    async def spawn(argv):
        ran.append(argv)
        return FakeProc()

    port = free_port()
    agent = Agent(name, port, spawn=spawn)
    server = uvicorn.Server(
        uvicorn.Config(agent.app(), host="0.0.0.0", port=port, log_level="error")
    )
    server.install_signal_handlers = lambda: None
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)
    await agent.start_discovery()
    try:
        yield agent, ran, f"http://127.0.0.1:{port}"
    finally:
        await agent._stop_process()
        await agent.stop_discovery()
        server.should_exit = True
        await task


async def test_a_room_forms_from_nearby_laptops_with_no_command() -> None:
    async with (
        laptop("alpha") as (a, a_ran, a_url),
        laptop("beta") as (b, b_ran, b_url),
        httpx.AsyncClient(timeout=10) as http,
    ):

        async def sees_beta() -> bool:
            return "beta" in [p["name"] for p in (await http.get(f"{a_url}/api/peers")).json()]

        await until(sees_beta)

        assert (await http.post(f"{a_url}/api/room", json={"model": "Qwen/x"})).status_code == 200
        assert (await http.post(f"{a_url}/api/invite", json={"peer": "beta"})).status_code == 200

        # beta sees the invite and has not started anything.
        invites = (await http.get(f"{b_url}/api/me")).json()["invites"]
        assert [(i["from"], i["model"]) for i in invites] == [("alpha", "Qwen/x")]
        assert b.state == "idle" and not b_ran

        # A join with no accepted invite is refused, and so is a guessed id.
        refused = await http.post(f"{b_url}/api/join", json={"id": invites[0]["id"]})
        assert refused.status_code == 403
        assert (await http.post(f"{b_url}/api/join", json={"id": "guess"})).status_code == 403

        assert (
            await http.post(f"{b_url}/api/accept", json={"id": invites[0]["id"]})
        ).status_code == 200
        assert (await http.get(f"{a_url}/api/me")).json()["room"]["accepted"] == ["beta"]

        assert (await http.post(f"{a_url}/api/start")).status_code == 200
        serve = a_ran[0]
        assert serve[0] == "serve" and serve[serve.index("--model") + 1] == "Qwen/x"
        assert serve[serve.index("--min-workers") + 1] == "2"
        assert serve[serve.index("--objective") + 1] == "balance"  # spread is the default
        assert b.state == "joined"
        assert b_ran[0][:2] == ["worker", "--head"] and b_ran[0][2].endswith(":7711")
        assert b_ran[0][-2:] == ["--name", "beta"]
        assert a.state == "hosting"

        # Stop frees both laptops.
        assert (await http.post(f"{a_url}/api/stop")).status_code == 200
        assert a.state == "idle" and b.state == "idle"


async def test_an_invite_only_comes_from_the_room_that_created_it() -> None:
    async with (
        laptop("alpha") as (a, _, a_url),
        laptop("beta"),
        httpx.AsyncClient(timeout=10) as http,
    ):
        # beta never invited alpha, so alpha's accept is refused.
        r = await http.post(f"{a_url}/api/accepted", json={"id": "x", "name": "beta"})
        assert r.status_code == 404
        r = await http.post(f"{a_url}/api/invite", json={"peer": "beta"})
        assert r.status_code == 400  # no room yet
        assert a.state == "idle"
