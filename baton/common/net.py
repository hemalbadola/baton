"""Socket helpers for both Baton planes: connect-with-retry, frame I/O, preamble auth.

PRD 8.1 sets three rules that this module enforces so no caller has to remember
them:

- `TCP_NODELAY` on every socket. A 16 KB activation must not wait for Nagle.
- One `writev` per frame. `send_frame` hands the three buffers of a frame to
  `StreamWriter.writelines`, so header, meta and payload leave in one syscall.
- Every write is followed by `await writer.drain()`.

Timeouts come from PRD 8.6. There is no per-frame deadline: a slow CPU prefill
chunk is legitimate, and liveness comes from `health`, not from frame timing.

Authentication is PRD 8.5. The control plane authenticates with `hello.token`.
Data sockets authenticate with a 32-byte preamble, because the ring carries no
`hello`. The LAN is trusted, so everything else is plaintext.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import socket
import time
from typing import Any

from baton.common.framing import HEADER_SIZE, decode_header, decode_meta, encode

__all__ = [
    "ADMISSION_QUEUE_WAIT",
    "CONTROL_CONNECT_TIMEOUT",
    "DATA_CONNECT_TIMEOUT",
    "HEALTH_INTERVAL",
    "HEALTH_SILENCE_TIMEOUT",
    "PREAMBLE_SIZE",
    "REQUEST_WALL_CLOCK",
    "RETRY_INTERVAL",
    "AuthError",
    "LinkClosed",
    "accept_data_link",
    "build_preamble",
    "check_preamble",
    "close_writer",
    "connect_with_retry",
    "format_addr",
    "open_data_link",
    "parse_addr",
    "read_frame",
    "recv_preamble",
    "send_frame",
    "send_preamble",
    "set_nodelay",
]

log = logging.getLogger("baton.net")

# PRD 8.6, all in seconds.
CONTROL_CONNECT_TIMEOUT = 5.0  # per attempt; retry forever
DATA_CONNECT_TIMEOUT = 30.0  # total, after load; then `loaded` fails
HEALTH_INTERVAL = 2.0
HEALTH_SILENCE_TIMEOUT = 6.0
REQUEST_WALL_CLOCK = 600.0
ADMISSION_QUEUE_WAIT = 120.0

RETRY_INTERVAL = 1.0  # PRD 6.4: retry every second
_LOG_INTERVAL = 60.0  # PRD 8.6: print once per minute while retrying

PREAMBLE_SIZE = 32
_DIGEST_SIZE = 16  # two 16-byte digests fill the preamble exactly
_CLUSTER_PERSON = b"baton-clu"
_TOKEN_PERSON = b"baton-tok"


class AuthError(Exception):
    """A data socket presented a preamble for another cluster or another token."""


class LinkClosed(ConnectionError):
    """The peer closed the connection. Clean at a frame boundary, truncated inside one."""


# --- addresses ----------------------------------------------------------------


def parse_addr(addr: str) -> tuple[str, int]:
    """Split an `"ip:port"` string, as carried in `data_addr` and `next_node`.

    Accepts a bracketed IPv6 literal, `"[::1]:9000"`.
    """
    host, sep, port = addr.rpartition(":")
    if not sep or not host:
        raise ValueError(f"address {addr!r} is not 'ip:port'")
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    try:
        return host, int(port)
    except ValueError:
        raise ValueError(f"address {addr!r} has a non-numeric port") from None


def format_addr(host: str, port: int) -> str:
    """Inverse of `parse_addr`. Brackets an IPv6 literal."""
    if ":" in host:
        return f"[{host}]:{port}"
    return f"{host}:{port}"


# --- sockets ------------------------------------------------------------------


def set_nodelay(writer: asyncio.StreamWriter) -> None:
    """Turn Nagle off. Safe to call on a transport that has no socket."""
    sock = writer.get_extra_info("socket")
    if sock is None:
        return
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except OSError as exc:  # a UNIX socket in a test, for example
        log.debug("TCP_NODELAY not set: %s", exc)


# --- preamble auth (PRD 8.5) ---------------------------------------------------


def _digest(value: str, person: bytes) -> bytes:
    return hashlib.blake2b(value.encode(), digest_size=_DIGEST_SIZE, person=person).digest()


def build_preamble(cluster_id: str, token: str = "") -> bytes:
    """Return the 32 bytes a data socket sends first: cluster digest, token hash.

    Both halves are keyed BLAKE2b digests, so the token never crosses the wire
    in the clear even though the rest of the stream is plaintext.
    """
    return _digest(cluster_id, _CLUSTER_PERSON) + _digest(token, _TOKEN_PERSON)


def check_preamble(data: bytes, cluster_id: str, token: str = "") -> None:
    """Verify a received preamble. Raise `AuthError` if it does not match.

    An empty `token` means the head runs without one, and any token hash is
    accepted (PRD 8.5). The cluster digest is always checked, because two Baton
    clusters on one LAN must not cross-connect.
    """
    if len(data) != PREAMBLE_SIZE:
        raise AuthError(f"preamble must be {PREAMBLE_SIZE} bytes, got {len(data)}")
    if not hmac.compare_digest(data[:_DIGEST_SIZE], _digest(cluster_id, _CLUSTER_PERSON)):
        raise AuthError("preamble is for another cluster")
    if token and not hmac.compare_digest(data[_DIGEST_SIZE:], _digest(token, _TOKEN_PERSON)):
        raise AuthError("preamble token hash does not match")


async def send_preamble(writer: asyncio.StreamWriter, cluster_id: str, token: str = "") -> None:
    """Write the preamble and drain. Call it before the first frame on a data socket."""
    writer.write(build_preamble(cluster_id, token))
    await writer.drain()


async def recv_preamble(
    reader: asyncio.StreamReader,
    cluster_id: str,
    token: str = "",
    *,
    timeout: float = CONTROL_CONNECT_TIMEOUT,
) -> None:
    """Read and verify a preamble. Raise `AuthError`, `LinkClosed` or `TimeoutError`.

    The timeout stops a peer that connects and then says nothing from holding a
    server coroutine open forever.
    """
    try:
        async with asyncio.timeout(timeout):
            data = await reader.readexactly(PREAMBLE_SIZE)
    except asyncio.IncompleteReadError as exc:
        raise LinkClosed(f"closed during preamble after {len(exc.partial)} bytes") from None
    check_preamble(data, cluster_id, token)


# --- frame I/O ----------------------------------------------------------------


async def send_frame(
    writer: asyncio.StreamWriter, meta: dict[str, Any], payload: bytes = b""
) -> None:
    """Write one frame with a single `writev`, then drain (PRD 8.1)."""
    writer.writelines(encode(meta, payload))
    await writer.drain()


async def read_frame(reader: asyncio.StreamReader) -> tuple[dict[str, Any], bytes]:
    """Read exactly one frame.

    Raise `LinkClosed` on EOF, `FrameError` on a malformed header. There is no
    timeout by design (PRD 8.6): wrap the call in `asyncio.timeout` if the
    caller has a deadline of its own.
    """
    try:
        header = await reader.readexactly(HEADER_SIZE)
    except asyncio.IncompleteReadError as exc:
        if not exc.partial:
            raise LinkClosed("peer closed the connection") from None
        raise LinkClosed(f"closed mid-header after {len(exc.partial)} bytes") from None

    meta_len, payload_len = decode_header(header)
    try:
        meta_bytes = await reader.readexactly(meta_len)
        payload = await reader.readexactly(payload_len) if payload_len else b""
    except asyncio.IncompleteReadError as exc:
        raise LinkClosed(f"closed mid-frame after {len(exc.partial)} bytes") from None

    return decode_meta(meta_bytes), payload


async def close_writer(writer: asyncio.StreamWriter) -> None:
    """Close a stream and wait for it, swallowing the errors of an already dead peer."""
    try:
        writer.close()
        await writer.wait_closed()
    except (OSError, ConnectionError) as exc:
        log.debug("error while closing: %s", exc)


# --- connect with retry (PRD 8.6) ----------------------------------------------


async def connect_with_retry(
    host: str,
    port: int,
    *,
    attempt_timeout: float = CONTROL_CONNECT_TIMEOUT,
    total_timeout: float | None = None,
    retry_interval: float = RETRY_INTERVAL,
    label: str = "peer",
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Dial `host:port` until it answers. Return the connected stream, nodelay set.

    `total_timeout` of None retries forever and logs once per minute, which is
    the control-plane behavior. Pass `DATA_CONNECT_TIMEOUT` for a data link, and
    a `TimeoutError` is raised when the budget runs out so `loaded` can fail and
    the head can re-plan.
    """
    start = time.monotonic()
    last_log = 0.0
    attempts = 0
    last_error: BaseException | None = None

    while True:
        attempts += 1
        remaining = None if total_timeout is None else total_timeout - (time.monotonic() - start)
        if remaining is not None and remaining <= 0:
            raise TimeoutError(
                f"no connection to {label} {format_addr(host, port)} in {total_timeout:g} s "
                f"after {attempts - 1} attempts: {last_error}"
            )
        budget = attempt_timeout if remaining is None else min(attempt_timeout, remaining)
        try:
            async with asyncio.timeout(budget):
                reader, writer = await asyncio.open_connection(host, port)
        except (OSError, TimeoutError) as exc:
            last_error = exc
            elapsed = time.monotonic() - start
            if elapsed - last_log >= _LOG_INTERVAL or last_log == 0.0:
                last_log = elapsed
                log.warning(
                    "waiting for %s at %s (%d attempts, %.0f s): %s",
                    label,
                    format_addr(host, port),
                    attempts,
                    elapsed,
                    exc or type(exc).__name__,
                )
        else:
            set_nodelay(writer)
            if attempts > 1:
                log.info(
                    "connected to %s at %s after %d attempts",
                    label,
                    format_addr(host, port),
                    attempts,
                )
            return reader, writer

        remaining = None if total_timeout is None else total_timeout - (time.monotonic() - start)
        if remaining is not None and remaining <= 0:
            raise TimeoutError(
                f"no connection to {label} {format_addr(host, port)} in {total_timeout:g} s "
                f"after {attempts} attempts: {last_error}"
            )
        delay = retry_interval if remaining is None else min(retry_interval, remaining)
        await asyncio.sleep(delay)


async def open_data_link(
    addr: str,
    cluster_id: str,
    token: str = "",
    *,
    total_timeout: float | None = DATA_CONNECT_TIMEOUT,
    label: str = "next node",
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Dial a ring peer by `"ip:port"` and send the preamble. Ready for frames."""
    host, port = parse_addr(addr)
    reader, writer = await connect_with_retry(host, port, total_timeout=total_timeout, label=label)
    try:
        await send_preamble(writer, cluster_id, token)
    except BaseException:
        await close_writer(writer)
        raise
    return reader, writer


async def accept_data_link(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    cluster_id: str,
    token: str = "",
    *,
    timeout: float = CONTROL_CONNECT_TIMEOUT,
) -> None:
    """Server side of `open_data_link`: set nodelay, then verify the preamble.

    The caller closes the socket if this raises. It does not close it here,
    because the caller usually wants to log the peer address first.
    """
    set_nodelay(writer)
    await recv_preamble(reader, cluster_id, token, timeout=timeout)
