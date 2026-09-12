"""Frame encode and decode for the Baton wire protocol (PRD 8.2).

Layout, big-endian:

    offset  size         field
    0       4            magic = b"BTN1"
    4       4            meta_len    (u32)
    8       4            payload_len (u32)
    12      meta_len     UTF-8 JSON object
    12+m    payload_len  raw bytes (may be zero-length)

One framing carries both planes. The JSON meta holds the message, the payload
holds tensor bytes. This module is synchronous and does no socket work, so it
stays cheap to test and to benchmark. `baton.common.net` puts frames on sockets.
"""

from __future__ import annotations

import json
import struct
from typing import Any

__all__ = [
    "HEADER_SIZE",
    "MAGIC",
    "MAX_META",
    "MAX_PAYLOAD",
    "FrameDecoder",
    "FrameError",
    "decode",
    "decode_header",
    "decode_meta",
    "encode",
    "encode_bytes",
]

MAGIC = b"BTN1"
_HEADER = struct.Struct("!4sII")
HEADER_SIZE = _HEADER.size  # 12

# u32 caps both lengths. PRD 8.2 states payload max 4 GiB.
MAX_PAYLOAD = 0xFFFFFFFF
# Meta is a control message of ~150 bytes. A megabyte is already absurd, so a
# larger value on the wire means a desynchronised stream, not a real frame.
MAX_META = 1 << 20

# Bound methods, resolved once. The hot path calls them per frame.
_pack_header = _HEADER.pack
_unpack_header = _HEADER.unpack_from
_dump_meta = json.JSONEncoder(separators=(",", ":"), check_circular=False).encode
_load_meta = json.JSONDecoder().decode


class FrameError(Exception):
    """A frame on the wire is malformed. The socket must be closed."""


def encode(meta: dict[str, Any], payload: bytes = b"") -> tuple[bytes, bytes, bytes]:
    """Return the three buffers of one frame: header, JSON meta, payload.

    The parts stay separate so the caller can hand them to `writer.writelines()`,
    which writes them with one `writev` and never copies the payload (PRD 8.1).
    """
    meta_bytes = _dump_meta(meta).encode()
    meta_len = len(meta_bytes)
    payload_len = len(payload)
    if meta_len > MAX_META:
        raise FrameError(f"meta of {meta_len} bytes is over the {MAX_META} byte limit")
    if payload_len > MAX_PAYLOAD:
        raise FrameError(f"payload of {payload_len} bytes is over the {MAX_PAYLOAD} byte limit")
    return _pack_header(MAGIC, meta_len, payload_len), meta_bytes, payload


def encode_bytes(meta: dict[str, Any], payload: bytes = b"") -> bytes:
    """Return one frame as a single `bytes`. This copies the payload.

    Use `encode()` on any hot path. This exists for tests, for files, and for
    callers that own a plain socket instead of an asyncio stream.
    """
    return b"".join(encode(meta, payload))


def decode_header(buf: bytes | bytearray | memoryview) -> tuple[int, int]:
    """Return `(meta_len, payload_len)` from the first 12 bytes of `buf`."""
    if len(buf) < HEADER_SIZE:
        raise FrameError(f"header needs {HEADER_SIZE} bytes, got {len(buf)}")
    magic, meta_len, payload_len = _unpack_header(buf, 0)
    if magic != MAGIC:
        raise FrameError(f"bad magic {bytes(magic)!r}, expected {MAGIC!r}")
    if meta_len > MAX_META:
        raise FrameError(f"meta_len {meta_len} is over the {MAX_META} byte limit")
    if meta_len == 0:
        raise FrameError("meta_len is zero, every frame carries a JSON object")
    return meta_len, payload_len


def decode_meta(meta_bytes: bytes | bytearray | memoryview) -> dict[str, Any]:
    """Decode the JSON meta of a frame whose header is already parsed.

    `net.read_frame` uses this: it reads the exact lengths off the stream, so it
    must not pay for a second header check.
    """
    meta = _load_meta(bytes(meta_bytes).decode())
    if not isinstance(meta, dict):
        raise FrameError(f"meta must be a JSON object, got {type(meta).__name__}")
    return meta


def decode(buf: bytes | bytearray | memoryview) -> tuple[dict[str, Any], bytes]:
    """Decode one complete frame from `buf`. Trailing bytes are an error.

    For a stream of frames use `FrameDecoder` or `baton.common.net.read_frame`.
    """
    meta_len, payload_len = decode_header(buf)
    end = HEADER_SIZE + meta_len + payload_len
    if len(buf) != end:
        raise FrameError(f"frame needs exactly {end} bytes, got {len(buf)}")
    view = memoryview(buf)
    meta = decode_meta(view[HEADER_SIZE : HEADER_SIZE + meta_len])
    return meta, bytes(view[HEADER_SIZE + meta_len : end])


class FrameDecoder:
    """Incremental decoder for a byte stream that carries many frames.

    Feed it whatever a `recv()` returned, then drain complete frames::

        dec = FrameDecoder()
        dec.feed(sock.recv(65536))
        for meta, payload in dec:
            handle(meta, payload)

    An asyncio caller does not need this: `net.read_frame` reads exact lengths
    off a `StreamReader`. This covers plain sockets and replayed capture files.
    """

    __slots__ = ("_buf", "_meta_len", "_payload_len")

    def __init__(self) -> None:
        self._buf = bytearray()
        self._meta_len = -1
        self._payload_len = -1

    def feed(self, data: bytes) -> None:
        self._buf += data

    def pending(self) -> int:
        """Bytes held that do not yet make a whole frame."""
        return len(self._buf)

    def next_frame(self) -> tuple[dict[str, Any], bytes] | None:
        """Return the next complete frame, or None if more bytes are needed."""
        buf = self._buf
        if self._meta_len < 0:
            if len(buf) < HEADER_SIZE:
                return None
            self._meta_len, self._payload_len = decode_header(buf)
        end = HEADER_SIZE + self._meta_len + self._payload_len
        if len(buf) < end:
            return None
        frame = decode(memoryview(buf)[:end])
        del buf[:end]
        self._meta_len = -1
        self._payload_len = -1
        return frame

    def __iter__(self):
        while (frame := self.next_frame()) is not None:
            yield frame
