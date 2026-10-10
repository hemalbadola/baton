"""The dense tensor cache (BAT-11): a second load fetches nothing."""

import pytest
import torch

from baton.model import safetensors_io as sio
from baton.model.cache import BlobCache
from baton.worker.engine import load_shard


class Net(sio.LocalSource):
    """A local file that plays the network: it counts reads and can go down."""

    commit = "c0ffee"

    def __init__(self, root) -> None:
        super().__init__(root)
        self.reads: list[tuple[int, int]] = []
        self.down = False

    def fetch(self, file: str, start: int, end: int) -> bytes:
        if self.down:
            raise sio.SafetensorsError("no network")
        self.reads.append((start, end))
        return super().fetch(file, start, end)


def load(directory, spec, net, root, first=0, last=3):
    header, _ = sio.read_header(directory / "model.safetensors")
    index = {name: "model.safetensors" for name in header if name != "__metadata__"}
    cache = BlobCache(root, "org/tiny", "main", net)
    stack, _ = load_shard(
        spec,
        first,
        last,
        embed=first == 0,
        head=last == 3,
        source=net,
        index=index,
        ctx_max=64,
        device="cpu",
        dtype=torch.float32,
        cache=cache,
    )
    return stack, cache


def logits(stack):
    with torch.no_grad():
        return stack(stack.embed(torch.tensor([1, 5, 9])))


def test_the_second_load_reads_the_disk_only(tiny_checkpoint, tmp_path) -> None:
    directory, spec, whole = tiny_checkpoint
    net = Net(directory)
    first, cache = load(directory, spec, net, tmp_path / "cache")
    assert cache.fetched_bytes > 0
    assert (tmp_path / "cache" / "models" / "org" / "tiny" / "c0ffee" / "blobs").is_dir()

    before = len(net.reads)
    second, cache = load(directory, spec, net, tmp_path / "cache")
    assert cache.fetched_bytes == 0
    assert len(net.reads) - before == 2  # the header, which also checks the commit
    assert torch.equal(logits(second), logits(first))
    assert torch.equal(logits(second), logits(whole))


def test_a_cached_shard_loads_with_no_network(tiny_checkpoint, tmp_path) -> None:
    directory, spec, whole = tiny_checkpoint
    net = Net(directory)
    load(directory, spec, net, tmp_path / "cache")
    net.down = True
    offline, cache = load(directory, spec, net, tmp_path / "cache")
    assert cache.fetched_bytes == 0
    assert torch.equal(logits(offline), logits(whole))


def test_no_cache_and_no_network_is_an_error(tiny_checkpoint, tmp_path) -> None:
    directory, spec, _ = tiny_checkpoint
    net = Net(directory)
    net.down = True
    with pytest.raises(sio.SafetensorsError):
        load(directory, spec, net, tmp_path / "cache")


def test_a_new_range_fetches_only_what_it_lacks(tiny_checkpoint, tmp_path) -> None:
    directory, spec, _ = tiny_checkpoint
    net = Net(directory)
    _, narrow = load(directory, spec, net, tmp_path / "cache", first=0, last=1)
    _, wide = load(directory, spec, net, tmp_path / "cache", first=0, last=3)
    _, again = load(directory, spec, net, tmp_path / "cache", first=2, last=3)
    assert 0 < wide.fetched_bytes  # layers 2 and 3, and the final norm
    assert again.fetched_bytes == 0  # a re-plan onto cached layers downloads nothing
    assert narrow.fetched_bytes > 0


def test_a_truncated_file_is_fetched_again(tiny_checkpoint, tmp_path) -> None:
    directory, spec, whole = tiny_checkpoint
    net = Net(directory)
    load(directory, spec, net, tmp_path / "cache")
    blobs = sorted((tmp_path / "cache").rglob("blobs/*/*"))
    blobs[0].write_bytes(blobs[0].read_bytes()[:-4])
    healed, cache = load(directory, spec, net, tmp_path / "cache")
    assert cache.fetched_bytes > 0
    assert torch.equal(logits(healed), logits(whole))


def test_a_new_commit_does_not_reuse_old_bytes(tiny_checkpoint, tmp_path) -> None:
    directory, spec, _ = tiny_checkpoint
    net = Net(directory)
    load(directory, spec, net, tmp_path / "cache")
    net.commit = "beef01"
    _, cache = load(directory, spec, net, tmp_path / "cache")
    assert cache.fetched_bytes > 0
