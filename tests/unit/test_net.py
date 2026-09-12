"""Unit tests for the socket helpers (PRD 8.1, 8.5, 8.6).

The tests run real loopback TCP. That is the only way to prove `TCP_NODELAY`,
`writev` and the preamble handshake actually work together.
"""

from __future__ import annotations

import asyncio
import socket
import time

import pytest

from baton.common.framing import FrameError
from baton.common.net import (
    DATA_CONNECT_TIMEOUT,
    HEALTH_SILENCE_TIMEOUT,
    PREAMBLE_SIZE,
    AuthError,
    LinkClosed,
    accept_data_link,
    build_preamble,
    check_preamble,
    close_writer,
    connect_with_retry,
    format_addr,
    open_data_link,
    parse_addr,
    read_frame,
    send_frame,
    set_nodelay,
)

CLUSTER = "baton-7f3a"
TOKEN = "hunter2"

# --- addresses ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("192.168.1.7:9001", ("192.168.1.7", 9001)),
        ("localhost:80", ("localhost", 80)),
        ("[::1]:9000", ("::1", 9000)),
    ],
)
def test_parse_addr(text, expected):
    assert parse_addr(text) == expected


@pytest.mark.parametrize("text", ["9001", ":9001", "host:port", "host:"])
def test_parse_addr_rejects_junk(text):
    with pytest.raises(ValueError):
        parse_addr(text)


def test_format_addr_round_trips_including_ipv6():
    for host, port in [("10.0.0.2", 9001), ("::1", 9000)]:
        assert parse_addr(format_addr(host, port)) == (host, port)


# --- preamble auth (PRD 8.5) ---------------------------------------------------


def test_preamble_is_exactly_32_bytes():
    assert PREAMBLE_SIZE == 32
    assert len(build_preamble(CLUSTER, TOKEN)) == 32


def test_preamble_does_not_carry_the_token_in_the_clear():
    assert TOKEN.encode() not in build_preamble(CLUSTER, TOKEN)
    assert CLUSTER.encode() not in build_preamble(CLUSTER, TOKEN)


def test_matching_preamble_passes():
    check_preamble(build_preamble(CLUSTER, TOKEN), CLUSTER, TOKEN)


def test_wrong_token_is_refused():
    with pytest.raises(AuthError, match="token hash"):
        check_preamble(build_preamble(CLUSTER, "wrong"), CLUSTER, TOKEN)


def test_wrong_cluster_is_refused_even_with_the_right_token():
    with pytest.raises(AuthError, match="another cluster"):
        check_preamble(build_preamble("other-cluster", TOKEN), CLUSTER, TOKEN)


def test_no_token_accepts_any_token_hash_but_still_checks_the_cluster():
    """PRD 8.5: no token means any LAN device can join."""
    check_preamble(build_preamble(CLUSTER, "whatever"), CLUSTER, "")
    with pytest.raises(AuthError, match="another cluster"):
        check_preamble(build_preamble("other", "whatever"), CLUSTER, "")


@pytest.mark.parametrize("size", [0, 31, 33])
def test_wrong_preamble_length_is_refused(size):
    with pytest.raises(AuthError, match="must be 32 bytes"):
        check_preamble(b"\x00" * size, CLUSTER, TOKEN)


# --- helpers for the socket tests ----------------------------------------------


class Server:
    """A loopback listener that runs one handler per connection."""

    def __init__(self, handler):
        self.handler = handler
        self.server = None
        self.port = 0
        self.errors: list[BaseException] = []

    async def __aenter__(self):
        self.server = await asyncio.start_server(self._wrap, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]
        return self

    async def _wrap(self, reader, writer):
        try:
            await self.handler(reader, writer)
        except BaseException as exc:  # noqa: BLE001 - recorded so a test can assert on it
            self.errors.append(exc)
        finally:
            await close_writer(writer)

    async def __aexit__(self, *exc):
        self.server.close()
        await self.server.wait_closed()


# --- frame I/O over a real socket ----------------------------------------------


async def test_frame_round_trip_over_loopback():
    received: list[tuple[dict, bytes]] = []

    async def handler(reader, writer):
        while True:
            try:
                received.append(await read_frame(reader))
            except LinkClosed:
                return

    async with Server(handler) as srv:
        _, writer = await connect_with_retry("127.0.0.1", srv.port, total_timeout=5.0)
        await send_frame(writer, {"t": "hello", "name": "n1", "token": TOKEN})
        await send_frame(writer, {"t": "act", "req": "r", "pos": 0, "n": 4}, b"\xbe\xef" * 32)
        await close_writer(writer)
        for _ in range(200):
            if len(received) == 2:
                break
            await asyncio.sleep(0.005)

    assert received[0] == ({"t": "hello", "name": "n1", "token": TOKEN}, b"")
    assert received[1][0]["t"] == "act"
    assert received[1][1] == b"\xbe\xef" * 32
    assert not srv.errors


async def test_connect_sets_tcp_nodelay():
    async def handler(reader, writer):
        await reader.read(1)

    async with Server(handler) as srv:
        _, writer = await connect_with_retry("127.0.0.1", srv.port, total_timeout=5.0)
        sock = writer.get_extra_info("socket")
        # macOS returns a non-zero flag value, not literally 1.
        assert sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY) != 0
        # Prove set_nodelay does the work and does not rely on the asyncio default.
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 0)
        assert sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY) == 0
        set_nodelay(writer)
        assert sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY) != 0
        await close_writer(writer)


async def test_read_frame_raises_link_closed_on_clean_eof():
    async def handler(reader, writer):
        writer.close()

    async with Server(handler) as srv:
        reader, writer = await connect_with_retry("127.0.0.1", srv.port, total_timeout=5.0)
        with pytest.raises(LinkClosed, match="peer closed"):
            await read_frame(reader)
        await close_writer(writer)


async def test_read_frame_raises_link_closed_on_a_truncated_frame():
    async def handler(reader, writer):
        from baton.common.framing import encode_bytes

        writer.write(encode_bytes({"t": "act", "req": "r", "pos": 0, "n": 1}, b"\x00" * 64)[:-10])
        await writer.drain()
        writer.close()

    async with Server(handler) as srv:
        reader, writer = await connect_with_retry("127.0.0.1", srv.port, total_timeout=5.0)
        with pytest.raises(LinkClosed, match="mid-frame"):
            await read_frame(reader)
        await close_writer(writer)


async def test_read_frame_raises_frame_error_on_bad_magic():
    async def handler(reader, writer):
        writer.write(b"HTTP/1.1 200 OK\r\n\r\n")
        await writer.drain()
        await asyncio.sleep(0.2)

    async with Server(handler) as srv:
        reader, writer = await connect_with_retry("127.0.0.1", srv.port, total_timeout=5.0)
        with pytest.raises(FrameError, match="bad magic"):
            await read_frame(reader)
        await close_writer(writer)


# --- data link handshake --------------------------------------------------------


async def test_open_and_accept_data_link():
    accepted = asyncio.Event()

    async def handler(reader, writer):
        await accept_data_link(reader, writer, CLUSTER, TOKEN)
        accepted.set()
        meta, _ = await read_frame(reader)
        assert meta["t"] == "prompt"

    async with Server(handler) as srv:
        _, writer = await open_data_link(
            format_addr("127.0.0.1", srv.port), CLUSTER, TOKEN, total_timeout=5.0
        )
        await send_frame(writer, {"t": "prompt", "req": "r1", "ids": [1]})
        await asyncio.wait_for(accepted.wait(), 2.0)
        await close_writer(writer)
    assert not srv.errors


async def test_accept_data_link_refuses_a_bad_token():
    async def handler(reader, writer):
        await accept_data_link(reader, writer, CLUSTER, TOKEN)

    async with Server(handler) as srv:
        reader, writer = await open_data_link(
            format_addr("127.0.0.1", srv.port), CLUSTER, "wrong", total_timeout=5.0
        )
        # The server drops us. The read must end, not hang.
        with pytest.raises(LinkClosed):
            await asyncio.wait_for(read_frame(reader), 2.0)
        await close_writer(writer)

    assert srv.errors and isinstance(srv.errors[0], AuthError)


async def test_recv_preamble_times_out_on_a_silent_peer():
    async def handler(reader, writer):
        await accept_data_link(reader, writer, CLUSTER, TOKEN, timeout=0.2)

    async with Server(handler) as srv:
        _, writer = await connect_with_retry("127.0.0.1", srv.port, total_timeout=5.0)
        await asyncio.sleep(0.5)  # say nothing
        await close_writer(writer)

    assert srv.errors and isinstance(srv.errors[0], (TimeoutError, LinkClosed))


# --- connect with retry (PRD 8.6) -------------------------------------------------


async def test_connect_with_retry_gives_up_at_the_total_timeout():
    port = _free_port()
    start = time.monotonic()
    with pytest.raises(TimeoutError, match="no connection to next node"):
        await connect_with_retry(
            "127.0.0.1",
            port,
            total_timeout=0.3,
            retry_interval=0.05,
            label="next node",
        )
    assert time.monotonic() - start < 3.0


async def test_connect_with_retry_waits_for_a_late_listener():
    """The control plane dials before the head is up. It must not fail (PRD 8.6)."""
    port = _free_port()
    server: list = []

    async def start_later():
        await asyncio.sleep(0.25)
        srv = await asyncio.start_server(lambda r, w: None, "127.0.0.1", port)
        server.append(srv)

    task = asyncio.create_task(start_later())
    _, writer = await connect_with_retry(
        "127.0.0.1", port, total_timeout=10.0, retry_interval=0.05, label="head"
    )
    await task
    assert writer.get_extra_info("socket") is not None
    await close_writer(writer)
    server[0].close()
    await server[0].wait_closed()


def test_timeout_constants_match_the_prd():
    assert DATA_CONNECT_TIMEOUT == 30.0
    assert HEALTH_SILENCE_TIMEOUT == 6.0


def _free_port() -> int:
    """A port that nothing listens on right now."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
