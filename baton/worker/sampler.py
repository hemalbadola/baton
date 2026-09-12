"""Token sampling on Nk (PRD 10.3).

The last node in the ring holds `lm_head`, so it samples. The head node never
sees logits (decision D6), which keeps 256 KB per token off the wire and saves
one hop.

Sampling parameters arrive once in the `prompt` frame and live per request on
Nk. `Sampler` is that per-request state: the parameters and the generator.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = ["Sampler", "SamplingParams", "apply_repetition_penalty", "top_k_filter", "top_p_filter"]

NEG_INF = float("-inf")


@dataclass(frozen=True)
class SamplingParams:
    """One request's sampling settings, as the OpenAI-shaped API names them."""

    temperature: float = 1.0
    top_k: int = 0  # 0 disables
    top_p: float = 1.0  # 1.0 disables
    repetition_penalty: float = 1.0  # 1.0 disables
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be > 0")

    @property
    def greedy(self) -> bool:
        return self.temperature == 0


def apply_repetition_penalty(logits: Tensor, seen: Iterable[int], penalty: float) -> Tensor:
    """Push down every token already in the prompt or the generation.

    Sign aware: a positive logit is divided, a negative one is multiplied. Both
    moves reduce the probability, which a plain division would not do for a
    negative logit.
    """
    if penalty == 1.0:
        return logits
    ids = torch.tensor(sorted(set(seen)), dtype=torch.long, device=logits.device)
    if ids.numel() == 0:
        return logits
    chosen = logits[ids]
    logits[ids] = torch.where(chosen > 0, chosen / penalty, chosen * penalty)
    return logits


def top_k_filter(logits: Tensor, k: int) -> Tensor:
    """Keep the `k` largest logits, mask the rest."""
    if k <= 0 or k >= logits.numel():
        return logits
    threshold = torch.topk(logits, k).values[-1]
    return logits.masked_fill(logits < threshold, NEG_INF)


def top_p_filter(logits: Tensor, p: float) -> Tensor:
    """Keep the smallest set of tokens whose probabilities reach `p` (nucleus).

    The token that crosses the threshold stays in, so the most likely token
    survives any `p > 0`.
    """
    if p >= 1.0:
        return logits
    ordered, index = torch.sort(logits, descending=True)
    cumulative = ordered.softmax(-1).cumsum(-1)
    drop = cumulative > p
    drop[1:] = drop[:-1].clone()
    drop[0] = False
    # `drop` is in sorted order; `index` maps it back to vocab order.
    return logits.index_fill(0, index[drop], NEG_INF)


class Sampler:
    """Per-request sampling state on Nk.

    A seeded run repeats on the same backend and torch version. Bit equality
    across backends is not promised (PRD 16.5).
    """

    def __init__(
        self,
        params: SamplingParams | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.params = params or SamplingParams()
        self.device = torch.device(device)
        seed = self.params.seed
        if seed is None:
            seed = int.from_bytes(os.urandom(8), "little")
        self.seed = seed
        # `torch.multinomial` needs a generator on the tensor's own device.
        self.generator = torch.Generator(device=self.device).manual_seed(seed)

    def sample(self, logits: Tensor, seen: Iterable[int] = ()) -> int:
        """Pick one token id from the logits of a single position.

        `logits` is `[vocab]` or `[1, vocab]`. `seen` is every token id in the
        prompt and the generation so far; N1 forwards the prompt ids inside the
        first `act` frame so Nk can apply the repetition penalty.
        """
        logits = logits.reshape(-1).to(torch.float32)
        p = self.params

        if p.repetition_penalty != 1.0:
            logits = apply_repetition_penalty(logits.clone(), seen, p.repetition_penalty)
        if p.greedy:
            return int(torch.argmax(logits))

        logits = logits / p.temperature
        logits = top_k_filter(logits, p.top_k)
        logits = top_p_filter(logits, p.top_p)

        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, 1, generator=self.generator))
