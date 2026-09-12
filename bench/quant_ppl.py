"""Measure bytes on disk and perplexity drift for bf16, int8 and int4 (PRD 5.5, 16).

The PRD expects RTN int4 g128 to cost 0.1-0.3 perplexity. This script measures it
instead of assuming it. Every decoder Linear is quantized, then the model runs on
wikitext-2 raw test.

Usage::

    python bench/quant_ppl.py --model Qwen/Qwen2.5-0.5B --windows 8

Writes ``bench/quant_ppl.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import httpx
import torch
from torch import nn

from baton.model.quant import linear, quantize, relative_error, tier_bytes

ROWS_URL = "https://datasets-server.huggingface.co/rows"
CORPUS = Path(__file__).parent / "wikitext2-test.txt"


# --------------------------------------------------------------------------- #
# Corpus
# --------------------------------------------------------------------------- #


def fetch_wikitext(rows: int = 1500) -> str:
    """wikitext-2 raw test, through the datasets server. Cached beside this script."""
    if CORPUS.exists():
        return CORPUS.read_text()
    out: list[str] = []
    with httpx.Client(timeout=60) as client:
        for offset in range(0, rows, 100):
            resp = client.get(
                ROWS_URL,
                params={
                    "dataset": "Salesforce/wikitext",
                    "config": "wikitext-2-raw-v1",
                    "split": "test",
                    "offset": offset,
                    "length": min(100, rows - offset),
                },
            )
            resp.raise_for_status()
            out += [r["row"]["text"] for r in resp.json()["rows"]]
    text = "".join(out)
    CORPUS.write_text(text)
    return text


# --------------------------------------------------------------------------- #
# A Linear that runs the tier's real matmul path
# --------------------------------------------------------------------------- #


class QuantLinear(nn.Module):
    """Drop-in for ``nn.Linear`` that goes through :func:`baton.model.quant.linear`."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None, tier: str, fast: bool):
        super().__init__()
        self.qw = quantize(weight, tier)
        self.bias = bias
        self.fast = fast if tier == "int4" else None
        self.kernel_dtype = torch.bfloat16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, x.shape[-1])
        if self.fast:
            # The packed-int4 kernel takes bf16 activations. A bf16 worker is the
            # real case; here the rest of the net runs in fp32, so cast in and out.
            out = linear(flat.to(self.kernel_dtype), self.qw, fast=True).to(flat.dtype)
            if self.bias is not None:
                out = out + self.bias
        else:
            out = linear(flat, self.qw, self.bias, fast=False)
        return out.reshape(*x.shape[:-1], out.shape[-1])


def decoder_linears(model) -> dict[str, nn.Linear]:
    """Every Linear inside the decoder layers. The embedding stays bf16 (decision D5)."""
    return {
        name: mod
        for name, mod in model.named_modules()
        if isinstance(mod, nn.Linear) and ".layers." in name
    }


def swap(model, originals: dict[str, torch.Tensor], tier: str, fast: bool) -> dict[str, float]:
    """Replace every decoder Linear with the quantized tier. Returns size and error stats."""
    stats = {"bytes": 0, "bf16_bytes": 0, "params": 0, "worst_rel_err": 0.0}
    for name, weight in originals.items():
        parent = model.get_submodule(name.rsplit(".", 1)[0])
        leaf = name.rsplit(".", 1)[1]
        old = getattr(parent, leaf)
        bias = old.bias.data if getattr(old, "bias", None) is not None else None

        ql = QuantLinear(weight, bias, tier, fast)
        setattr(parent, leaf, ql)

        shape = (weight.shape[0], weight.shape[1])
        stats["bytes"] += tier_bytes(shape, tier)
        stats["bf16_bytes"] += tier_bytes(shape, "bf16")
        stats["params"] += weight.numel()
        stats["worst_rel_err"] = max(
            stats["worst_rel_err"], relative_error(ql.qw.dequantize(torch.float32), weight.float())
        )
    return stats


# --------------------------------------------------------------------------- #
# Perplexity
# --------------------------------------------------------------------------- #


@torch.no_grad()
def perplexity(model, ids: torch.Tensor, window: int, windows: int) -> tuple[float, int]:
    """Mean negative log likelihood over non-overlapping windows, then exp."""
    total, counted = 0.0, 0
    for i in range(windows):
        chunk = ids[:, i * window : (i + 1) * window]
        if chunk.shape[1] < 2:
            break
        logits = model(chunk).logits.float()
        loss = nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]), chunk[:, 1:].reshape(-1)
        )
        total += loss.item() * (chunk.shape[1] - 1)
        counted += chunk.shape[1] - 1
    return math.exp(total / counted), counted


# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--rows", type=int, default=1500)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    model.eval()

    ids = tok(fetch_wikitext(args.rows), return_tensors="pt").input_ids
    originals = {
        n: m.weight.data.clone().to(torch.bfloat16) for n, m in decoder_linears(model).items()
    }
    print(f"{args.model}: {len(originals)} decoder Linears, corpus {ids.shape[1]} tokens")

    runs = [("bf16", False), ("int8", False), ("int4", False), ("int4", True)]
    rows = []
    for tier, fast in runs:
        label = f"{tier}-fast" if fast else tier
        t0 = time.time()
        stats = swap(model, originals, tier, fast)
        ppl, counted = perplexity(model, ids, args.window, args.windows)
        rows.append(
            {
                "tier": label,
                "decoder_weight_bytes": stats["bytes"],
                "vs_bf16": round(stats["bytes"] / stats["bf16_bytes"], 4),
                "worst_weight_rel_err": round(stats["worst_rel_err"], 5),
                "ppl": round(ppl, 4),
                "tokens": counted,
                "seconds": round(time.time() - t0, 1),
            }
        )
        print(json.dumps(rows[-1]))

    base = rows[0]["ppl"]
    for r in rows:
        r["ppl_drift"] = round(r["ppl"] - base, 4)

    out = Path(__file__).parent / "quant_ppl.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
