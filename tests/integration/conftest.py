"""Fixtures for the ring tests.

Sockets bind on 127.0.0.1, so this directory needs the sandbox off. A
`PermissionError` on bind is the sandbox, not a bug (see god's memory).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from baton.model.spec import ModelSpec

# The geometry of `tests/numerics/conftest.TINY`: every flag that changes the
# maths (llama3 rope, grouped-query attention, eps, theta), only the sizes
# shrunk. Copied rather than imported so a ring test never imports the
# `transformers` reference.
TINY_CONFIG: dict = {
    "hidden_size": 128,
    "intermediate_size": 256,
    "num_hidden_layers": 4,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "vocab_size": 512,
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

SEED = 7
# The PRD 19 exit test: 64 greedy tokens. PRD 16.2 asks for 20 prompts.
M0_MODEL = os.environ.get("BATON_M0_MODEL", "unsloth/Llama-3.2-1B")
M0_PROMPTS = int(os.environ.get("BATON_M0_PROMPTS", "20"))
M0_TOKENS = int(os.environ.get("BATON_M0_TOKENS", "64"))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: needs the real Llama-3.2-1B weights on disk")


@pytest.fixture(scope="session")
def tiny_spec() -> ModelSpec:
    return ModelSpec.from_config(TINY_CONFIG)


@pytest.fixture
def random_prompts(tiny_spec):
    """Eight prompts of mixed length, fixed by seed."""
    gen = torch.Generator().manual_seed(SEED)
    lengths = [5, 12, 1, 33, 20, 8, 40, 17]
    return [torch.randint(0, tiny_spec.vocab, (n,), generator=gen).tolist() for n in lengths]


def llama_dir() -> Path | None:
    """The local snapshot of the M0 model, or None when it is not on disk.

    Nothing here downloads. `BATON_M0_DIR` points at any directory holding
    `config.json`, `model.safetensors` and the tokenizer files.
    """
    override = os.environ.get("BATON_M0_DIR")
    if override:
        return Path(override)
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(M0_MODEL, "model.safetensors", local_files_only=True)
    except Exception:  # noqa: BLE001 - any cache miss means "skip", the reason is printed
        return None
    return Path(path).parent


@pytest.fixture(scope="session")
def llama():
    """`(spec, weights, prompts)` for the real model, or skip."""
    d = llama_dir()
    if d is None or not (d / "config.json").exists():
        pytest.skip(f"{M0_MODEL} is not in the local Hugging Face cache")
    transformers = pytest.importorskip("transformers")
    tok = transformers.AutoTokenizer.from_pretrained(d)
    prompts = [tok(text).input_ids for text in PROMPT_TEXTS[:M0_PROMPTS]]
    spec = ModelSpec.from_pretrained(d)
    return spec, ("safetensors", str(d / "model.safetensors")), prompts


PROMPT_TEXTS = [
    "The capital of France is",
    "Once upon a time, in a small village by the sea,",
    "def fibonacci(n):\n    ",
    "The three laws of thermodynamics are",
    "In 1969, Neil Armstrong",
    "A recipe for a simple tomato soup:",
    "The difference between a list and a tuple in Python is",
    "Water boils at",
    "Dear hiring manager, I am writing to",
    "The mitochondria is",
    "Q: What is the largest planet in the solar system?\nA:",
    "To install the package, run",
    "Shakespeare wrote",
    "The quick brown fox",
    "Photosynthesis converts",
    "1, 1, 2, 3, 5, 8,",
    "My favourite season is autumn because",
    "The Treaty of Westphalia in 1648",
    "import numpy as np\nimport torch\n\n",
    "The theory of relativity states that",
]
