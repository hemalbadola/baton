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
async def laptop(name: str, head_http: int = 7700):
    ran: list[list[str]] = []

    async def spawn(argv):
        ran.append(argv)
        return FakeProc()

    port = free_port()
    agent = Agent(name, port, spawn=spawn, head_http=head_http)
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


@contextlib.asynccontextmanager
async def fake_head():
    """Answers a chat the way the head does: three SSE chunks, then the timing chunk."""
    from fastapi import FastAPI
    from fastapi.responses import StreamingResponse

    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat() -> StreamingResponse:
        async def events():
            for word in ("Par", "is", "."):
                yield f'data: {{"choices":[{{"delta":{{"content":"{word}"}}}}]}}\n\n'
            yield 'data: {"choices":[{"delta":{"content":""}}],"x_baton":{"decode_tok_s":20.0,"ttft_ms":100}}\n\n'
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    server.install_signal_handlers = lambda: None
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)
    try:
        yield port
    finally:
        server.should_exit = True
        await task


async def test_every_laptop_shows_the_same_conversation() -> None:
    ask = {"model": "x", "stream": True, "messages": [{"role": "user", "content": "Capital?"}]}
    async with (
        fake_head() as head_port,
        laptop("alpha", head_port) as (a, _, a_url),
        laptop("beta") as (_b, _, b_url),
        httpx.AsyncClient(timeout=10) as http,
    ):

        async def sees_beta() -> bool:
            return "beta" in [p["name"] for p in (await http.get(f"{a_url}/api/peers")).json()]

        await until(sees_beta)
        await http.post(f"{a_url}/api/room", json={"model": "Qwen/x"})
        await http.post(f"{a_url}/api/invite", json={"peer": "beta"})
        invite = (await http.get(f"{b_url}/api/me")).json()["invites"][0]["id"]
        await http.post(f"{b_url}/api/accept", json={"id": invite})
        await http.post(f"{a_url}/api/start")

        # The guest asks. Its request goes through the host, which keeps the conversation.
        reply = await http.post(f"{b_url}/v1/chat/completions", json=ask)
        assert reply.status_code == 200 and "Par" in reply.text

        want = [
            {"role": "user", "content": "Capital?", "by": "beta"},
            {
                "role": "assistant",
                "content": "Paris.",
                "done": True,
                "tok_s": 20.0,
                "ttft_ms": 100,
            },
        ]
        assert a.transcript == want
        assert (await http.get(f"{a_url}/api/transcript")).json() == want
        assert (await http.get(f"{b_url}/api/transcript")).json() == want

        await http.post(f"{a_url}/api/stop")
        assert a.transcript == []


async def test_another_laptop_cannot_click_for_the_user() -> None:
    """Accept, Start and Stop are the user's clicks. They must come from the page
    on this laptop, never from the network."""
    from baton.common.net import lan_addresses

    lan = lan_addresses()[0]
    if lan.startswith("127."):
        return  # no network: every address is loopback
    async with laptop("alpha") as (a, ran, a_url), httpx.AsyncClient(timeout=10) as http:
        remote = a_url.replace("127.0.0.1", lan)
        for path in ("room", "invite", "start", "stop", "accept", "decline"):
            r = await http.post(f"{remote}/api/{path}", json={"model": "x", "id": "y", "peer": "z"})
            assert r.status_code == 403, path
        assert a.state == "idle" and not ran
        assert (await http.get(f"{remote}/api/me")).status_code == 200  # peers may look


async def test_the_public_page_may_ask_only_whether_baton_runs_here() -> None:
    web = "https://baton-plum.vercel.app"
    async with laptop("alpha") as (_a, _, url), httpx.AsyncClient(timeout=10) as http:
        ok = await http.get(f"{url}/api/hello", headers={"origin": web})
        assert ok.json()["baton"] is True and ok.json()["version"]
        assert ok.headers["access-control-allow-origin"] == web
        assert ok.headers["access-control-allow-private-network"] == "true"

        other = await http.get(f"{url}/api/hello", headers={"origin": "https://evil.example"})
        assert "access-control-allow-origin" not in other.headers

        # Nothing else is open to a web page: no log, no actions.
        me = await http.get(f"{url}/api/me", headers={"origin": web})
        assert "access-control-allow-origin" not in me.headers
