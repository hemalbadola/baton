"""Per-frame overhead, the second half of the M0 exit test (PRD 19, 17.2).

`frame_overhead` in PRD 17.2 is the encode plus decode cost of one frame.
`bench/frame_bench.py` prints the full table; this pins the gate in pytest on
the frames the 1B ring actually sends. The node-side figure (queue hop, bytes
to tensor and back, meta build) is measured from the ring traces in
`test_ring_llama.py` and reported, not gated: the PRD gates framing alone.
"""

from __future__ import annotations

import statistics
import time

import pytest

from baton.common.framing import HEADER_SIZE, decode_header, decode_meta, encode, encode_bytes

GATE_S = 0.1e-3
HIDDEN = 2048  # Llama-3.2-1B
TRACE = [{"node": "n1", "t_recv": 9837.836639, "t_send": 9837.837270708, "compute": 0.0004821}]


def median_seconds(fn, iters: int, batches: int = 5) -> float:
    """Median over batches, so one GC pause cannot become the headline number."""
    out = []
    for _ in range(batches):
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        out.append((time.perf_counter() - start) / iters)
    return statistics.median(out)


@pytest.mark.parametrize(
    "n,wire",
    [(1, "fp32"), (1, "bf16"), (256, "fp32"), (256, "bf16")],
    ids=["decode-fp32", "decode-bf16", "prefill-fp32", "prefill-bf16"],
)
def test_encode_plus_decode_under_the_gate(n, wire):
    meta = {"t": "act", "req": "r0", "pos": 0, "n": n, "dtype": wire, "last": True, "trace": TRACE}
    payload = b"\xab" * (n * HIDDEN * (4 if wire == "fp32" else 2))
    frame = encode_bytes(meta, payload)
    meta_len = len(frame) - HEADER_SIZE - len(payload)
    header, meta_view = frame[:HEADER_SIZE], memoryview(frame)[HEADER_SIZE : HEADER_SIZE + meta_len]

    iters = 5000 if n == 1 else 200
    enc = median_seconds(lambda: encode(meta, payload), iters)
    dec = median_seconds(lambda: (decode_header(header), decode_meta(meta_view)), iters)
    total = enc + dec
    print(
        f"\n{n}x{HIDDEN} {wire}: encode {enc * 1e6:.1f} us + decode {dec * 1e6:.1f} us = {total * 1e6:.1f} us"
    )
    assert total < GATE_S, f"encode + decode took {total * 1e3:.4f} ms, gate is {GATE_S * 1e3} ms"
