"""Run `async def` tests without pytest-asyncio.

Ten lines of hook beat a plugin dependency. Each coroutine test gets its own
event loop, which is what we want anyway: a leaked socket in one test must not
reach the next one.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    func = pyfuncitem.obj
    if not inspect.iscoroutinefunction(func):
        return None
    kwargs = {name: pyfuncitem.funcargs[name] for name in pyfuncitem._fixtureinfo.argnames}
    asyncio.run(func(**kwargs))
    return True


@pytest.fixture
def tiny_checkpoint(tmp_path):
    """A random 4-layer checkpoint on disk in Hugging Face layout.

    Returns `(directory, spec, stack)`. `stack` is the whole model that wrote
    the file, so a test can compare a split load against it.
    """
    import json

    import torch

    from baton.model import safetensors_io as sio
    from baton.model.layers import DecoderStack
    from baton.model.spec import ModelSpec

    config = {
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "vocab_size": 512,
        "tie_word_embeddings": True,
    }
    spec = ModelSpec.from_config(config)
    torch.manual_seed(0)
    stack = DecoderStack(spec, 0, spec.n_layers, embed=True, head=True, max_ctx=64).eval()
    state = stack.state_dict()
    tensors = {name: state[key] for key, name in stack.checkpoint_keys().items()}
    (tmp_path / "config.json").write_text(json.dumps(config))
    sio.save_safetensors(tmp_path / "model.safetensors", tensors)
    return tmp_path, spec, stack
