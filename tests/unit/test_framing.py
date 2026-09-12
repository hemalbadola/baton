"""Unit tests for the BTN1 frame codec (PRD 8.2)."""

from __future__ import annotations

import json
import struct

import pytest

from baton.common.framing import (
    HEADER_SIZE,
    MAGIC,
    MAX_META,
    FrameDecoder,
    FrameError,
    decode,
    decode_header,
    decode_meta,
    encode,
    encode_bytes,
)


def test_header_is_twelve_bytes_big_endian():
    header, meta, payload = encode({"t": "unload"}, b"")
    assert len(header) == HEADER_SIZE == 12
    magic, meta_len, payload_len = struct.unpack("!4sII", header)
    assert magic == MAGIC == b"BTN1"
    assert meta_len == len(meta)
    assert payload_len == len(payload) == 0


def test_encode_returns_three_parts_for_writev():
    payload = b"\x01\x02\x03\x04"
    parts = encode({"t": "act", "req": "r1", "pos": 0, "n": 1}, payload)
    assert len(parts) == 3
    # The payload part is the very object handed in. writev must not copy it.
    assert parts[2] is payload


def test_round_trip_control_frame():
    meta = {"t": "health", "mem_free": 1 << 33, "kv_used": 0, "queue_depth": 2, "active_reqs": 1}
    got_meta, got_payload = decode(encode_bytes(meta))
    assert got_meta == meta
    assert got_payload == b""


def test_round_trip_payload_frame():
    payload = bytes(range(256)) * 64  # 16 KiB, one activation hop
    meta = {"t": "act", "req": "abc", "pos": 7, "n": 8, "dtype": "bf16", "trace": []}
    got_meta, got_payload = decode(encode_bytes(meta, payload))
    assert got_meta == meta
    assert got_payload == payload


def test_meta_json_is_compact():
    _, meta, _ = encode({"t": "next", "req": "r", "id": 5})
    assert b", " not in meta
    assert b'": ' not in meta
    assert json.loads(meta) == {"t": "next", "req": "r", "id": 5}


def test_unicode_meta_round_trips():
    meta = {"t": "error", "code": "load", "message": "no space — ужас"}
    assert decode(encode_bytes(meta))[0] == meta


@pytest.mark.parametrize(
    "buf",
    [
        b"",
        b"BTN1",
        b"BTN1\x00\x00\x00",
    ],
)
def test_short_header_raises(buf):
    with pytest.raises(FrameError, match="header needs"):
        decode_header(buf)


def test_bad_magic_raises():
    bad = b"XXXX" + struct.pack("!II", 2, 0) + b"{}"
    with pytest.raises(FrameError, match="bad magic"):
        decode(bad)


def test_zero_meta_len_raises():
    with pytest.raises(FrameError, match="meta_len is zero"):
        decode_header(b"BTN1" + struct.pack("!II", 0, 0))


def test_absurd_meta_len_raises_before_allocation():
    with pytest.raises(FrameError, match="over the"):
        decode_header(b"BTN1" + struct.pack("!II", MAX_META + 1, 0))


def test_truncated_frame_raises():
    frame = encode_bytes({"t": "unload"})
    with pytest.raises(FrameError, match="needs exactly"):
        decode(frame[:-1])


def test_trailing_bytes_raise():
    with pytest.raises(FrameError, match="needs exactly"):
        decode(encode_bytes({"t": "unload"}) + b"x")


def test_meta_must_be_an_object():
    body = b"[1,2]"
    bad = b"BTN1" + struct.pack("!II", len(body), 0) + body
    with pytest.raises(FrameError, match="must be a JSON object"):
        decode(bad)
    with pytest.raises(FrameError, match="must be a JSON object"):
        decode_meta(body)


def test_oversize_meta_is_refused_on_encode():
    with pytest.raises(FrameError, match="over the"):
        encode({"t": "error", "message": "x" * (MAX_META + 1)})


def test_decoder_splits_a_stream_of_frames():
    frames = [
        ({"t": "prompt", "req": "r1", "ids": [1, 2, 3]}, b""),
        ({"t": "act", "req": "r1", "pos": 0, "n": 3}, b"\xaa" * 48),
        ({"t": "next", "req": "r1", "id": 9, "pos": 1}, b""),
    ]
    stream = b"".join(encode_bytes(m, p) for m, p in frames)
    dec = FrameDecoder()
    dec.feed(stream)
    assert list(dec) == frames
    assert dec.pending() == 0


def test_decoder_handles_byte_at_a_time_delivery():
    """A TCP segment boundary can fall anywhere, including inside the header."""
    meta = {"t": "act", "req": "r", "pos": 0, "n": 2, "dtype": "bf16"}
    payload = b"\x10\x20\x30\x40"
    stream = encode_bytes(meta, payload)
    dec = FrameDecoder()
    for i, byte in enumerate(stream):
        dec.feed(bytes([byte]))
        got = dec.next_frame()
        if i < len(stream) - 1:
            assert got is None
        else:
            assert got == (meta, payload)


def test_decoder_keeps_a_partial_second_frame():
    first = encode_bytes({"t": "unload"})
    second = encode_bytes({"t": "bench", "quant": "int4"})
    dec = FrameDecoder()
    dec.feed(first + second[:5])
    assert dec.next_frame() == ({"t": "unload"}, b"")
    assert dec.next_frame() is None
    assert dec.pending() == 5
    dec.feed(second[5:])
    assert dec.next_frame() == ({"t": "bench", "quant": "int4"}, b"")


def test_codec_overhead_is_under_the_m0_gate():
    """M0 exit gate: encode plus decode of one frame under 0.1 ms.

    The bound is ten times the measured cost on an M-series laptop, so this
    catches a real regression (a copy added to the hot path) and not jitter.
    Run `python bench/frame_bench.py` for the full table.
    """
    import time

    meta = {
        "t": "act",
        "req": "8f14e45fea",
        "pos": 412,
        "n": 1,
        "dtype": "bf16",
        "trace": [["n1", 1234567.8], ["n2", 1234568.1]],
    }
    payload = b"\xab" * (8192 * 2)  # 16 KiB, one decode hop
    frame = encode_bytes(meta, payload)
    meta_len = len(frame) - HEADER_SIZE - len(payload)
    header = memoryview(frame)[:HEADER_SIZE]
    meta_view = memoryview(frame)[HEADER_SIZE : HEADER_SIZE + meta_len]

    iters = 5000
    best = float("inf")
    for _ in range(3):
        start = time.perf_counter_ns()
        for _ in range(iters):
            encode(meta, payload)
            decode_header(header)
            decode_meta(meta_view)
        best = min(best, (time.perf_counter_ns() - start) / iters / 1e6)
    assert best < 0.1, f"{best:.4f} ms per frame exceeds the 0.1 ms M0 gate"
