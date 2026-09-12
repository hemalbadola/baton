"""The decoder stack (PRD 5.2).

One `DecoderLayer` class covers Llama 3.x and Qwen 2.5. Per-model differences
arrive as flags on `ModelSpec`, never as subclasses.

Two rules run through the whole file:

* Reductions happen in fp32 and the result is cast back. RMSNorm, the RoPE
  tables and the attention softmax all follow this, so cross-backend drift comes
  from matmul alone (PRD 16.5).
* Submodules carry the Hugging Face names (`self_attn.q_proj`, `mlp.gate_proj`,
  ...). A checkpoint tensor then loads with no name mapping beyond the layer
  index shift a shard needs.

A worker holds a contiguous layer range, so `DecoderStack` is built for
`[layer_start, layer_end)` and only the last node in the ring carries
`norm` and `lm_head`.
"""

from __future__ import annotations

import math
from typing import Protocol

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from baton.model.spec import ModelSpec

__all__ = [
    "MLP",
    "Attention",
    "DecoderLayer",
    "DecoderStack",
    "LayerKVCache",
    "RMSNorm",
    "apply_rope",
    "build_rope_tables",
    "causal_mask",
    "rms_norm",
    "rotate_half",
]


class LayerKVCache(Protocol):
    """What `Attention` needs from a per-layer KV cache.

    `model/cache.py` is another lane's module, so the decoder depends on this
    shape and nothing more.
    """

    def append(self, k: Tensor, v: Tensor, pos_start: int) -> tuple[Tensor, Tensor]:
        """Write `k`, `v` at absolute positions `pos_start …` and return the whole
        cache so far.

        `k` and `v` arrive as `[n, n_kv_heads, head_dim]`. The return is
        `[pos_start + n, n_kv_heads, head_dim]` for each.
        """
        ...


def _probe_enable_gqa() -> bool:
    """True when this torch build takes `enable_gqa` (PyTorch >= 2.5)."""
    q = torch.zeros(1, 2, 1, 2)
    k = torch.zeros(1, 1, 1, 2)
    try:
        F.scaled_dot_product_attention(q, k, k, enable_gqa=True)
    except (TypeError, RuntimeError):
        return False
    return True


ENABLE_GQA = _probe_enable_gqa()


# --- normalization -------------------------------------------------------


def rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """Root-mean-square norm with the reduction in fp32 (PRD 5.2 step 1)."""
    dtype = x.dtype
    x32 = x.to(torch.float32)
    variance = x32.pow(2).mean(-1, keepdim=True)
    x32 = x32 * torch.rsqrt(variance + eps)
    return weight * x32.to(dtype)


class RMSNorm(nn.Module):
    def __init__(self, hidden: int, eps: float, **factory: object) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden, **factory))  # type: ignore[arg-type]

    def forward(self, x: Tensor) -> Tensor:
        return rms_norm(x, self.weight, self.eps)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


# --- rotary position embedding -------------------------------------------


def _inv_freq(spec: ModelSpec, device: torch.device | None = None) -> Tensor:
    """Inverse frequencies, with the llama3 rescale applied when the config asks
    for it.

    The rescale is the single easiest constant to get wrong, and a wrong one
    still produces fluent text. `tests/numerics/test_rope.py` is the only thing
    that catches it.
    """
    dim = spec.head_dim
    base = spec.rope_theta
    steps = torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float32)
    inv_freq = 1.0 / (base ** (steps / dim))

    if spec.rope_type == "default":
        return inv_freq
    if spec.rope_type != "llama3":
        raise NotImplementedError(f"rope_scaling type {spec.rope_type!r} is not supported")

    scaling = spec.rope_scaling or {}
    factor = float(scaling["factor"])
    low_freq_factor = float(scaling["low_freq_factor"])
    high_freq_factor = float(scaling["high_freq_factor"])
    old_ctx = float(scaling["original_max_position_embeddings"])

    low_freq_wavelen = old_ctx / low_freq_factor
    high_freq_wavelen = old_ctx / high_freq_factor
    wavelen = 2 * math.pi / inv_freq

    # Long wavelengths (low frequency) are stretched by the full factor.
    out = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    # The band between the two wavelengths blends the stretched and raw values.
    smooth = (old_ctx / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed = (1 - smooth) * out / factor + smooth * out
    is_medium = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    return torch.where(is_medium, smoothed, out)


def build_rope_tables(
    spec: ModelSpec, max_ctx: int, device: torch.device | None = None
) -> tuple[Tensor, Tensor]:
    """Precompute `cos` and `sin` for positions `0 … max_ctx-1` (PRD 5.2 step 3).

    Both come back as fp32 `[max_ctx, head_dim]`. Built once per worker and
    indexed by absolute position, so decode costs a slice.
    """
    inv_freq = _inv_freq(spec, device=device)
    positions = torch.arange(max_ctx, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    # Each frequency appears twice because `rotate_half` splits the head in two.
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x: Tensor) -> Tensor:
    """Rotate the two halves of the last dimension: `[a, b] -> [-b, a]`."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate `x: [n, heads, head_dim]` by the angles in `cos`/`sin: [n, head_dim]`."""
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    return x * cos + rotate_half(x) * sin


def causal_mask(
    n: int, kv_len: int, pos_start: int, device: torch.device | None = None
) -> Tensor | None:
    """Boolean `[1, 1, n, kv_len]` mask, True where the query may attend.

    Query `i` sits at absolute position `pos_start + i` and sees every key up to
    that position. Returns None for a single-token decode step, where every key
    in the cache is visible and a mask would only cost memory.
    """
    if n == 1:
        return None
    q_pos = torch.arange(pos_start, pos_start + n, device=device)
    k_pos = torch.arange(kv_len, device=device)
    return (k_pos.unsqueeze(0) <= q_pos.unsqueeze(1))[None, None]


# --- attention -----------------------------------------------------------


class Attention(nn.Module):
    """Grouped-query attention over a KV cache (PRD 5.2 steps 2-6)."""

    def __init__(self, spec: ModelSpec, **factory: object) -> None:
        super().__init__()
        self.spec = spec
        self.n_heads = spec.n_heads
        self.n_kv_heads = spec.n_kv_heads
        self.head_dim = spec.head_dim
        self.scale = spec.head_dim**-0.5
        bias = spec.attn_bias
        self.q_proj = nn.Linear(spec.hidden, spec.q_dim, bias=bias, **factory)  # type: ignore[arg-type]
        self.k_proj = nn.Linear(spec.hidden, spec.kv_dim, bias=bias, **factory)  # type: ignore[arg-type]
        self.v_proj = nn.Linear(spec.hidden, spec.kv_dim, bias=bias, **factory)  # type: ignore[arg-type]
        self.o_proj = nn.Linear(spec.q_dim, spec.hidden, bias=False, **factory)  # type: ignore[arg-type]

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        pos_start: int = 0,
        cache: LayerKVCache | None = None,
    ) -> Tensor:
        n = x.shape[0]
        q = self.q_proj(x).view(n, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(n, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(n, self.n_kv_heads, self.head_dim)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if cache is not None:
            k, v = cache.append(k, v, pos_start)
        kv_len = k.shape[0]

        # SDPA wants [batch, heads, seq, dim]. One request per call, so batch is 1.
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        mask = causal_mask(n, kv_len, pos_start, device=x.device)
        if ENABLE_GQA:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, scale=self.scale, enable_gqa=True
            )
        else:
            groups = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scale)

        out = out.squeeze(0).transpose(0, 1).reshape(n, self.n_heads * self.head_dim)
        return self.o_proj(out)


# --- feed-forward --------------------------------------------------------


class MLP(nn.Module):
    """SwiGLU feed-forward block (PRD 5.2 step 7)."""

    def __init__(self, spec: ModelSpec, **factory: object) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(spec.hidden, spec.intermediate, bias=False, **factory)  # type: ignore[arg-type]
        self.up_proj = nn.Linear(spec.hidden, spec.intermediate, bias=False, **factory)  # type: ignore[arg-type]
        self.down_proj = nn.Linear(spec.intermediate, spec.hidden, bias=False, **factory)  # type: ignore[arg-type]

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# --- one layer -----------------------------------------------------------


class DecoderLayer(nn.Module):
    """Norm, attention, residual, norm, MLP, residual."""

    def __init__(self, spec: ModelSpec, **factory: object) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(spec.hidden, spec.rms_eps, **factory)
        self.self_attn = Attention(spec, **factory)
        self.post_attention_layernorm = RMSNorm(spec.hidden, spec.rms_eps, **factory)
        self.mlp = MLP(spec, **factory)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        pos_start: int = 0,
        cache: LayerKVCache | None = None,
    ) -> Tensor:
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, pos_start, cache)
        return x + self.mlp(self.post_attention_layernorm(x))


# Our submodule path for each `ModelSpec.tensor_names` key. The checkpoint name
# differs only by the `model.layers.{i}.` prefix a shard strips.
_LOCAL_SUFFIX: dict[str, str] = {
    "input_layernorm": "input_layernorm.weight",
    "q_proj": "self_attn.q_proj.weight",
    "k_proj": "self_attn.k_proj.weight",
    "v_proj": "self_attn.v_proj.weight",
    "o_proj": "self_attn.o_proj.weight",
    "q_bias": "self_attn.q_proj.bias",
    "k_bias": "self_attn.k_proj.bias",
    "v_bias": "self_attn.v_proj.bias",
    "post_attention_layernorm": "post_attention_layernorm.weight",
    "gate_proj": "mlp.gate_proj.weight",
    "up_proj": "mlp.up_proj.weight",
    "down_proj": "mlp.down_proj.weight",
}


# --- the shard -----------------------------------------------------------


class DecoderStack(nn.Module):
    """The layers one worker owns, plus the pieces the ring ends carry.

    `embed_tokens` exists only on N1, `norm` and `lm_head` only on Nk. On a
    single-device run one stack holds all three.
    """

    def __init__(
        self,
        spec: ModelSpec,
        layer_start: int,
        layer_end: int,
        *,
        embed: bool = False,
        head: bool = False,
        max_ctx: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if not 0 <= layer_start <= layer_end <= spec.n_layers:
            raise ValueError(f"layer range [{layer_start}, {layer_end}) outside 0..{spec.n_layers}")
        self.spec = spec
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.max_ctx = max_ctx or spec.max_position
        factory = {"device": device, "dtype": dtype}

        self.embed_tokens = nn.Embedding(spec.vocab, spec.hidden, **factory) if embed else None
        self.layers = nn.ModuleList(
            DecoderLayer(spec, **factory) for _ in range(layer_start, layer_end)
        )
        self.norm = RMSNorm(spec.hidden, spec.rms_eps, **factory) if head else None
        self.lm_head = nn.Linear(spec.hidden, spec.vocab, bias=False, **factory) if head else None
        # Llama 3.2 1B and 3B ship no `lm_head.weight`; the embedding is reused
        # (PRD 5.1). On a single-device run one stack holds both, so share the
        # tensor instead of keeping a second copy.
        if self.lm_head is not None and self.embed_tokens is not None and spec.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # fp32 tables regardless of compute dtype; `apply_rope` casts the slice.
        cos, sin = build_rope_tables(spec, self.max_ctx, device=torch.device(device or "cpu"))
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    @property
    def n_local_layers(self) -> int:
        return self.layer_end - self.layer_start

    def embed(self, ids: Tensor) -> Tensor:
        """Token ids `[n]` to hidden states `[n, hidden]`. N1 only."""
        if self.embed_tokens is None:
            raise RuntimeError("this stack has no embedding table")
        return self.embed_tokens(ids)

    def forward(
        self,
        x: Tensor,
        pos_start: int = 0,
        cache: list[LayerKVCache] | None = None,
        last_only: bool = False,
    ) -> Tensor:
        """Run the local layers over `x: [n, hidden]`.

        Returns hidden states `[n, hidden]`, or logits when this stack carries
        `lm_head`. With `last_only`, the head runs on the final row alone, which
        is the only row a prefill chunk needs (PRD 5.2 step 8).
        """
        if x.shape[0] + pos_start > self.max_ctx:
            raise ValueError(
                f"position {pos_start + x.shape[0]} exceeds rope table of {self.max_ctx}"
            )
        positions = slice(pos_start, pos_start + x.shape[0])
        cos = self.rope_cos[positions]
        sin = self.rope_sin[positions]

        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, pos_start, cache[i] if cache is not None else None)

        if self.lm_head is None or self.norm is None:
            return x
        if last_only:
            x = x[-1:]
        return self.lm_head(self.norm(x))

    # --- weights ---------------------------------------------------------

    def load_hf_weights(self, tensors: dict[str, Tensor]) -> None:
        """Copy Hugging Face checkpoint tensors into this stack.

        `tensors` is keyed by checkpoint name. Only the tensors this shard owns
        need to be present.
        """
        spec = self.spec
        state: dict[str, Tensor] = {}
        for local, absolute in enumerate(range(self.layer_start, self.layer_end)):
            for key, name in spec.layer_names(absolute).items():
                state[f"layers.{local}.{_LOCAL_SUFFIX[key]}"] = tensors[name]
        if self.embed_tokens is not None:
            state["embed_tokens.weight"] = tensors[spec.tensor_names["embed"]]
        if self.norm is not None:
            names = spec.head_names()
            state["norm.weight"] = tensors[names["final_norm"]]
            state["lm_head.weight"] = tensors[names["lm_head"]]
        self.load_state_dict(state, strict=True)
