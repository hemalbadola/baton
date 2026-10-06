"""Shard load: a layer range fetched by byte range becomes a resident stack (PRD 5.3)."""

import pytest
import torch

from baton.model import safetensors_io as sio
from baton.model.spec import replace_spec
from baton.worker.engine import ShardTooLarge, load_shard


def _index(directory) -> dict[str, str]:
    header, _ = sio.read_header(directory / "model.safetensors")
    return {name: "model.safetensors" for name in header if name != "__metadata__"}


def _load(directory, spec, first, last, **roles):
    return load_shard(
        spec,
        first,
        last,
        source=sio.open_source(str(directory)),
        index=_index(directory),
        ctx_max=64,
        device="cpu",
        dtype=torch.float32,
        **roles,
    )


def test_two_shards_reproduce_the_whole_model(tiny_checkpoint) -> None:
    directory, spec, whole = tiny_checkpoint
    seen: list[tuple[int, int, int]] = []
    n1, n1_bytes = load_shard(
        spec,
        0,
        1,
        embed=True,
        head=False,
        source=sio.open_source(str(directory)),
        index=_index(directory),
        ctx_max=64,
        device="cpu",
        dtype=torch.float32,
        on_progress=lambda *step: seen.append(step),
    )
    nk, nk_bytes = _load(directory, spec, 2, 3, embed=False, head=True)

    ids = torch.tensor([1, 5, 9, 2])
    with torch.no_grad():
        assert torch.equal(nk(n1(n1.embed(ids))), whole(whole.embed(ids)))

    done, total, fetched = seen[-1]
    assert done == total == len(seen) == len(spec.range_names(0, 2, embed=True))
    assert fetched > 0
    assert n1_bytes > 0 and nk_bytes > 0


def test_a_shard_over_budget_is_refused_before_it_allocates(tiny_checkpoint) -> None:
    directory, spec, _ = tiny_checkpoint
    with pytest.raises(ShardTooLarge):
        load_shard(
            spec,
            0,
            3,
            embed=True,
            head=True,
            source=sio.open_source(str(directory)),
            index=_index(directory),
            ctx_max=64,
            device="cpu",
            dtype=torch.float32,
            budget_bytes=1000,
        )


def test_a_wrong_shaped_tensor_is_an_error_not_a_broadcast(tiny_checkpoint) -> None:
    directory, spec, _ = tiny_checkpoint
    with pytest.raises(ValueError, match="checkpoint shape"):
        _load(directory, replace_spec(spec, vocab=spec.vocab + 1), 0, 0, embed=True, head=False)


def test_a_name_missing_from_the_index_is_reported(tiny_checkpoint) -> None:
    directory, spec, _ = tiny_checkpoint
    with pytest.raises(KeyError, match="index names no file"):
        load_shard(
            spec,
            0,
            0,
            embed=False,
            head=False,
            source=sio.open_source(str(directory)),
            index={},
            ctx_max=64,
            device="cpu",
            dtype=torch.float32,
        )


def test_no_read_is_larger_than_one_tensor(tiny_checkpoint) -> None:
    """BAT-22. Tensors sit back to back in the file. One merged read of the whole
    range would sit in memory beside the stack and double the peak."""
    directory, spec, _ = tiny_checkpoint
    reads: list[int] = []

    class Counting(sio.LocalSource):
        def fetch(self, file: str, start: int, end: int) -> bytes:
            reads.append(end - start)
            return super().fetch(file, start, end)

    index = _index(directory)
    header, header_len = sio.read_header(directory / "model.safetensors")
    largest = max(ref.nbytes for ref in sio.refs_from_header("f", header, header_len).values())
    load_shard(
        spec,
        0,
        3,
        embed=True,
        head=True,
        source=Counting(directory),
        index=index,
        ctx_max=64,
        device="cpu",
        dtype=torch.float32,
    )
    assert max(reads) <= max(largest, 8 + header_len)
