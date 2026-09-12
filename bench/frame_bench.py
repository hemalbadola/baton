"""Measure per-frame encode plus decode overhead (PRD 8.2, M0 exit gate).

The gate: encode plus decode of one frame must cost under 0.1 ms. PRD 8.2 gives
the expectation as under 50 us per frame in CPython 3.11.

What is measured, per frame:

    encode  -> the three buffers of the frame (header, JSON meta, payload)
    decode  -> struct.unpack of the header, json.loads of the meta

The payload is deliberately not copied in the reported number. On the real path
`net.read_frame` reads the payload straight off the socket into its own buffer,
so a memcpy of the activation is not framing overhead. The last column reports
`decode()`, which does copy, so the cost of that copy stays visible.

Run:  python bench/frame_bench.py [--iters N] [--csv out.csv]
"""

from __future__ import annotations

import argparse
import csv
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from baton.common.framing import (
    HEADER_SIZE,
    decode,
    decode_header,
    decode_meta,
    encode,
    encode_bytes,
)

GATE_MS = 0.1  # M0 exit test
PRD_EXPECTATION_US = 50.0  # PRD 8.2


def cases() -> list[tuple[str, dict, bytes]]:
    """Frames that the fleet actually sends, smallest to largest."""
    trace = [["n1", 1234567.8], ["n2", 1234568.1]]
    return [
        (
            "health (control, no payload)",
            {
                "t": "health",
                "mem_free": 12_884_901_888,
                "kv_used": 268_435_456,
                "queue_depth": 0,
                "active_reqs": 1,
                "loaded_rev": 3,
            },
            b"",
        ),
        (
            "next (data, no payload)",
            {
                "t": "next",
                "req": "8f14e45fea",
                "id": 128009,
                "pos": 412,
                "trace": trace,
            },
            b"",
        ),
        (
            "prompt (data, no payload)",
            {
                "t": "prompt",
                "req": "8f14e45fea",
                "ids": list(range(64)),
                "max_len": 512,
                "sampling": {"temperature": 0.7, "top_p": 0.95, "top_k": 40},
                "stop_ids": [128001, 128009],
                "trace": trace,
            },
            b"",
        ),
        # One decode step: 1 token x hidden 8192 in bf16 = 16 KiB (PRD D1).
        (
            "act, 1 x 8192 bf16 (16 KiB)",
            {
                "t": "act",
                "req": "8f14e45fea",
                "pos": 412,
                "n": 1,
                "dtype": "bf16",
                "trace": trace,
            },
            b"\xab" * (8192 * 2),
        ),
        # One prefill chunk: 256 tokens x hidden 8192 in bf16 = 4 MiB (PRD D10).
        (
            "act, 256 x 8192 bf16 (4 MiB)",
            {
                "t": "act",
                "req": "8f14e45fea",
                "pos": 0,
                "n": 256,
                "dtype": "bf16",
                "trace": trace,
            },
            b"\xab" * (256 * 8192 * 2),
        ),
    ]


def time_ns(fn, iters: int) -> float:
    """Median nanoseconds per call over five batches. Median beats mean here:
    one GC pause must not become the headline number."""
    batches = []
    for _ in range(5):
        start = time.perf_counter_ns()
        for _ in range(iters):
            fn()
        batches.append((time.perf_counter_ns() - start) / iters)
    return statistics.median(batches)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=0, help="calls per batch (0 = pick by size)")
    ap.add_argument("--csv", type=Path, default=None, help="append the results to this CSV")
    args = ap.parse_args()

    print(f"python {platform.python_version()} on {platform.machine()} / {platform.system()}")
    print(f"gate: encode + decode under {GATE_MS} ms per frame\n")
    header = (
        f"{'frame':<32}{'meta B':>8}{'payload B':>11}"
        f"{'enc us':>9}{'dec us':>9}{'sum us':>9}{'+copy us':>10}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    worst = 0.0
    for name, meta, payload in cases():
        frame = encode_bytes(meta, payload)
        meta_len = len(frame) - HEADER_SIZE - len(payload)
        iters = args.iters or (20_000 if len(payload) < 1 << 20 else 200)

        enc_us = time_ns(lambda m=meta, p=payload: encode(m, p), iters) / 1000
        whole = memoryview(frame)
        meta_view = whole[HEADER_SIZE : HEADER_SIZE + meta_len]

        def dec(h=whole[:HEADER_SIZE], m=meta_view):
            decode_header(h)
            decode_meta(m)

        dec_us = time_ns(dec, iters) / 1000
        copy_us = time_ns(lambda f=frame: decode(f), iters) / 1000
        total = enc_us + dec_us
        worst = max(worst, total)
        rows.append((name, meta_len, len(payload), enc_us, dec_us, total, copy_us))
        print(
            f"{name:<32}{meta_len:>8}{len(payload):>11}"
            f"{enc_us:>9.2f}{dec_us:>9.2f}{total:>9.2f}{copy_us:>10.2f}"
        )

    worst_ms = worst / 1000
    print(f"\nworst frame: {worst:.2f} us = {worst_ms:.4f} ms encode + decode")
    print(f"PRD 8.2 expectation: under {PRD_EXPECTATION_US:.0f} us per frame")
    ok = worst_ms < GATE_MS
    print(f"M0 gate (< {GATE_MS} ms): {'PASS' if ok else 'FAIL'}")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        new = not args.csv.exists()
        with args.csv.open("a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(
                    [
                        "frame",
                        "meta_bytes",
                        "payload_bytes",
                        "encode_us",
                        "decode_us",
                        "sum_us",
                        "decode_with_copy_us",
                    ]
                )
            w.writerows(rows)
        print(f"wrote {args.csv}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
