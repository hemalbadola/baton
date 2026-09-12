"""Shared fixtures for the layer-level numeric gate (PRD 16.1).

Every test here compares our decoder against `transformers` with the same
weights. Reading generated text never catches wrong maths, because wrong maths
still writes fluent sentences. Only this comparison catches it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# Allow `pytest tests/numerics` from the repo root without an editable install.
try:  # pragma: no cover - import side effect
    import baton  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from baton.model.spec import ModelSpec  # noqa: E402

transformers = pytest.importorskip("transformers", reason="the reference implementation")

# --- tolerances ----------------------------------------------------------
#
# TOL_FP32 is the gate the PRD names: max abs diff < 1e-4 for fp32 on CPU.
# TOL_EXACT is used where the two implementations run the same fp32 operations
# in the same order (RoPE tables, RMSNorm). A drift there is a real change, not
# rounding, so the bound is tight enough to see it.
TOL_FP32 = 1e-4
TOL_EXACT = 1e-6
# bf16 carries 8 mantissa bits. A residual stream of this depth lands near 1e-2.
TOL_BF16 = 5e-2

SEED = 1234
N_TOKENS = 64  # PRD 16.1 says a 64-token random input.

# Llama-3.2-1B `config.json`, field for field. The rope block is the part that
# is easy to get wrong and impossible to see in the output.
LLAMA_3_2_1B: dict = {
    "hidden_size": 2048,
    "intermediate_size": 8192,
    "num_hidden_layers": 16,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 64,
    "vocab_size": 128256,
    "rms_norm_eps": 1e-5,
    "rope_theta": 500000.0,
    "rope_scaling": {
        "rope_type": "llama3",
        "factor": 32.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
    },
    "max_position_embeddings": 131072,
    "tie_word_embeddings": True,
    "attention_bias": False,
}

# Llama-3.2-1B geometry costs about 250 MB per layer in fp32, so a whole-stack
# comparison cannot run in CI. TINY keeps every flag that changes the maths --
# llama3 rope, grouped-query attention, the same eps and theta -- and shrinks
# only the sizes. `head_dim` 16 still puts one frequency inside the llama3
# smoothing band, so the branch is exercised.
TINY: dict = {
    **LLAMA_3_2_1B,
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_hidden_layers": 4,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "vocab_size": 512,
}

# Qwen 2.5 differs by attention bias and plain rope at theta 1e6 (PRD 5.1).
QWEN_TINY: dict = {
    **TINY,
    "rope_theta": 1e6,
    "rope_scaling": None,
    "attention_bias": True,
    "tie_word_embeddings": False,
}


def make_spec(config: dict, **overrides) -> ModelSpec:
    return ModelSpec.from_config({**config, **overrides})


def make_hf_config(config: dict, **overrides):
    """A `LlamaConfig` holding exactly the values `make_spec` reads."""
    merged = {**config, **overrides}
    return transformers.LlamaConfig(
        hidden_size=merged["hidden_size"],
        intermediate_size=merged["intermediate_size"],
        num_hidden_layers=merged["num_hidden_layers"],
        num_attention_heads=merged["num_attention_heads"],
        num_key_value_heads=merged["num_key_value_heads"],
        head_dim=merged["head_dim"],
        vocab_size=merged["vocab_size"],
        rms_norm_eps=merged["rms_norm_eps"],
        rope_theta=merged["rope_theta"],
        rope_scaling=merged["rope_scaling"],
        max_position_embeddings=merged["max_position_embeddings"],
        tie_word_embeddings=merged["tie_word_embeddings"],
        attention_bias=merged["attention_bias"],
        attn_implementation="eager",
    )


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.to(torch.float32) - b.to(torch.float32)).abs().max().item()


def randomize(module: torch.nn.Module, seed: int = SEED) -> torch.nn.Module:
    """Small random weights, so the residual stream stays in a normal range.

    Default `nn.Linear` init would do, but a fixed scale keeps the tolerance
    numbers in this file meaningful run to run.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in module.named_parameters():
            if name.endswith("layernorm.weight") or name.endswith("norm.weight"):
                param.copy_(1 + 0.02 * torch.randn(param.shape, generator=generator))
            elif name.endswith(".bias"):
                param.copy_(0.02 * torch.randn(param.shape, generator=generator))
            else:
                param.copy_(0.02 * torch.randn(param.shape, generator=generator))
    return module


class RefKVCache:
    """The smallest cache that satisfies `layers.LayerKVCache`.

    `model/cache.py` belongs to another lane. This exists so the decoder's
    incremental path can be tested without waiting for it.
    """

    def __init__(self, spec: ModelSpec, max_ctx: int, dtype: torch.dtype = torch.float32) -> None:
        shape = (max_ctx, spec.n_kv_heads, spec.head_dim)
        self.k = torch.zeros(shape, dtype=dtype)
        self.v = torch.zeros(shape, dtype=dtype)
        self.length = 0

    def append(self, k, v, pos_start):
        n = k.shape[0]
        self.k[pos_start : pos_start + n] = k
        self.v[pos_start : pos_start + n] = v
        self.length = max(self.length, pos_start + n)
        return self.k[: self.length], self.v[: self.length]


@pytest.fixture(scope="module")
def tiny_spec() -> ModelSpec:
    return make_spec(TINY)


@pytest.fixture(scope="module")
def llama_1b_spec() -> ModelSpec:
    return make_spec(LLAMA_3_2_1B)


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(SEED)
    yield
