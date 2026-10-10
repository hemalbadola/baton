"""Unit tests for baton.model.safetensors_io (PRD 5.3, 12.1)."""

from __future__ import annotations

import json
import struct

import pytest
import torch

from baton.model import safetensors_io as sio


def make_file(tmp_path, name="model.safetensors", tensors=None, metadata=None):
    tensors = tensors or {
        "model.layers.0.self_attn.q_proj.weight": torch.arange(12, dtype=torch.float32).reshape(
            3, 4
        ),
        "model.layers.0.mlp.down_proj.weight": torch.full((2, 5), -1.5, dtype=torch.bfloat16),
    }
    path = sio.save_safetensors(tmp_path / name, tensors, metadata)
    return path, tensors


# --------------------------------------------------------------------------- #
# Header parsing
# --------------------------------------------------------------------------- #


def test_round_trip_write_then_read(tmp_path):
    path, tensors = make_file(tmp_path)
    header, header_len = sio.read_header(path)
    refs = sio.refs_from_header(path.name, header, header_len)

    assert set(refs) == set(tensors)
    got = sio.load_tensors(list(refs.values()), sio.LocalSource(tmp_path))
    for name, want in tensors.items():
        assert got[name].dtype == want.dtype
        assert got[name].shape == want.shape
        assert torch.equal(got[name], want)


def test_header_offsets_are_absolute(tmp_path):
    path, _ = make_file(tmp_path)
    header, header_len = sio.read_header(path)
    refs = sio.refs_from_header(path.name, header, header_len)
    first = min(refs.values(), key=lambda r: r.start)
    assert first.start == sio.HEADER_LEN_BYTES + header_len
    assert path.stat().st_size == max(r.end for r in refs.values())


def test_reading_one_tensor_touches_only_its_bytes(tmp_path):
    """The point of partial loading: read the range, never the file."""
    path, tensors = make_file(tmp_path)
    header, header_len = sio.read_header(path)
    ref = sio.refs_from_header(path.name, header, header_len)["model.layers.0.mlp.down_proj.weight"]

    with open(path, "rb") as fh:
        fh.seek(ref.start)
        blob = fh.read(ref.nbytes)

    assert len(blob) < path.stat().st_size
    assert torch.equal(
        sio.slice_tensor(blob, ref.start, ref),
        tensors["model.layers.0.mlp.down_proj.weight"],
    )


def test_metadata_key_is_not_a_tensor(tmp_path):
    path, tensors = make_file(tmp_path, metadata={"quant": "int4", "format": "pt"})
    header, header_len = sio.read_header(path)
    assert header["__metadata__"]["quant"] == "int4"
    assert set(sio.refs_from_header(path.name, header, header_len)) == set(tensors)


def test_truncated_prefix_raises():
    with pytest.raises(sio.SafetensorsError):
        sio.parse_header(b"\x01\x02")


def test_header_shorter_than_declared_raises():
    blob = struct.pack("<Q", 4096) + b"{}"
    with pytest.raises(sio.SafetensorsError):
        sio.parse_header(blob)


def test_bad_json_raises():
    body = b"not json"
    with pytest.raises(sio.SafetensorsError):
        sio.parse_header(struct.pack("<Q", len(body)) + body)


def test_absurd_header_length_raises():
    with pytest.raises(sio.SafetensorsError):
        sio.parse_header(struct.pack("<Q", sio.MAX_HEADER_BYTES + 1) + b"{}")


def test_unknown_dtype_raises(tmp_path):
    ref = sio.TensorRef("w", "f.safetensors", "F8_E4M3", (2,), 0, 2)
    with pytest.raises(sio.SafetensorsError):
        _ = ref.torch_dtype


# --------------------------------------------------------------------------- #
# resolve
# --------------------------------------------------------------------------- #


def _headers(tmp_path, *names):
    return {n: sio.read_header(tmp_path / n) for n in names}


def test_resolve_uses_the_index_weight_map(tmp_path):
    make_file(tmp_path, "a.safetensors", {"x": torch.ones(2)})
    make_file(tmp_path, "b.safetensors", {"y": torch.ones(3)})
    weight_map = {"x": "a.safetensors", "y": "b.safetensors"}
    refs = sio.resolve(weight_map, _headers(tmp_path, "a.safetensors", "b.safetensors"), ["y", "x"])
    assert [r.file for r in refs] == ["b.safetensors", "a.safetensors"]


def test_resolve_falls_back_to_a_single_file_checkpoint(tmp_path):
    make_file(tmp_path, "only.safetensors", {"x": torch.ones(2)})
    refs = sio.resolve({}, _headers(tmp_path, "only.safetensors"), ["x"])
    assert refs[0].file == "only.safetensors"


def test_resolve_missing_name_raises(tmp_path):
    make_file(tmp_path, "only.safetensors", {"x": torch.ones(2)})
    with pytest.raises(sio.SafetensorsError):
        sio.resolve({}, _headers(tmp_path, "only.safetensors"), ["absent"])


def test_resolve_rejects_an_index_that_lies(tmp_path):
    make_file(tmp_path, "a.safetensors", {"x": torch.ones(2)})
    with pytest.raises(sio.SafetensorsError):
        sio.resolve({"y": "a.safetensors"}, _headers(tmp_path, "a.safetensors"), ["y"])


# --------------------------------------------------------------------------- #
# coalesce
# --------------------------------------------------------------------------- #


def ref(name, file, start, end):
    return sio.TensorRef(name, file, "U8", (end - start,), start, end)


def test_adjacent_ranges_merge_into_one_fetch():
    spans = sio.coalesce([ref("a", "f", 0, 100), ref("b", "f", 100, 200)])
    assert len(spans) == 1
    assert (spans[0].start, spans[0].end) == (0, 200)
    assert [r.name for r in spans[0].refs] == ["a", "b"]


def test_a_gap_over_the_limit_splits_the_fetch():
    spans = sio.coalesce([ref("a", "f", 0, 100), ref("b", "f", 100 + (1 << 21), 200 + (1 << 21))])
    assert len(spans) == 2


def test_a_gap_under_the_limit_merges():
    spans = sio.coalesce([ref("a", "f", 0, 100), ref("b", "f", 1000, 1100)])
    assert len(spans) == 1
    assert spans[0].nbytes == 1100


def test_different_files_never_merge():
    spans = sio.coalesce([ref("a", "f1", 0, 100), ref("b", "f2", 100, 200)])
    assert {s.file for s in spans} == {"f1", "f2"}


def test_coalesce_stops_a_span_at_the_cap():
    refs = [ref(n, "f", i * 100, (i + 1) * 100) for i, n in enumerate("abcde")]
    spans = sio.coalesce(refs, max_gap=0, max_span=250)
    assert [(s.start, s.end) for s in spans] == [(0, 200), (200, 400), (400, 500)]


def test_coalesce_keeps_one_tensor_over_the_cap_whole():
    spans = sio.coalesce([ref("a", "f", 0, 1000), ref("b", "f", 1000, 1010)], 0, max_span=100)
    assert [(s.start, s.end) for s in spans] == [(0, 1000), (1000, 1010)]


def test_coalesce_sorts_out_of_order_input():
    spans = sio.coalesce([ref("b", "f", 100, 200), ref("a", "f", 0, 100)])
    assert [r.name for r in spans[0].refs] == ["a", "b"]


def test_nine_layer_tensors_in_one_file_become_one_request():
    """PRD 12.1: a layer's tensors typically coalesce into one or two requests."""
    refs = [ref(f"t{i}", "f", i * 1000, i * 1000 + 900) for i in range(9)]
    assert len(sio.coalesce(refs)) == 1


# --------------------------------------------------------------------------- #
# Streaming and verification
# --------------------------------------------------------------------------- #


class CountingSource:
    def __init__(self, root):
        self.inner = sio.LocalSource(root)
        self.calls = []

    def fetch(self, file, start, end):
        self.calls.append((file, start, end))
        return self.inner.fetch(file, start, end)


def test_iter_tensors_makes_one_request_per_span(tmp_path):
    path, tensors = make_file(tmp_path)
    header, header_len = sio.read_header(path)
    refs = list(sio.refs_from_header(path.name, header, header_len).values())
    src = CountingSource(tmp_path)

    got = dict(sio.iter_tensors(refs, src))

    assert len(src.calls) == 1  # the two tensors are adjacent
    assert set(got) == set(tensors)


def test_local_source_short_read_raises(tmp_path):
    path, _ = make_file(tmp_path)
    with pytest.raises(sio.SafetensorsError):
        sio.LocalSource(tmp_path).fetch(path.name, 0, path.stat().st_size + 64)


def test_nan_in_a_tensor_is_caught(tmp_path):
    bad = torch.tensor([1.0, float("nan")], dtype=torch.float32)
    path = sio.save_safetensors(tmp_path / "bad.safetensors", {"w": bad})
    header, header_len = sio.read_header(path)
    refs = list(sio.refs_from_header(path.name, header, header_len).values())
    with pytest.raises(sio.SafetensorsError):
        sio.load_tensors(refs, sio.LocalSource(tmp_path))
    # The same read passes when verification is off.
    got = sio.load_tensors(refs, sio.LocalSource(tmp_path), verify=False)
    assert torch.isnan(got["w"][1])


def test_slice_outside_the_span_raises():
    r = ref("a", "f", 100, 200)
    with pytest.raises(sio.SafetensorsError):
        sio.slice_tensor(b"\x00" * 50, 100, r)


def test_shape_that_disagrees_with_the_byte_range_raises():
    r = sio.TensorRef("a", "f", "F32", (99,), 0, 8)
    with pytest.raises(sio.SafetensorsError):
        sio.slice_tensor(b"\x00" * 8, 0, r)


# --------------------------------------------------------------------------- #
# RangeSource
# --------------------------------------------------------------------------- #


class FakeResponse:
    def __init__(self, status_code, content, headers=None, history=()):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}
        self.history = history


class FakeClient:
    """Serves ranges out of an in-memory file and records the headers it saw."""

    def __init__(self, data: bytes, statuses=None):
        self.data = data
        self.statuses = list(statuses or [])
        self.seen = []

    def get(self, url, headers):
        self.seen.append((url, headers))
        status = self.statuses.pop(0) if self.statuses else 206
        lo, hi = headers["Range"].removeprefix("bytes=").split("-")
        body = self.data[int(lo) : int(hi) + 1]
        return FakeResponse(status, body if status in (200, 206) else b"")


def test_range_header_is_inclusive_of_the_last_byte():
    client = FakeClient(bytes(range(256)))
    src = sio.RangeSource("https://hf.co/repo/resolve/main", client=client)
    assert src.fetch("f.safetensors", 10, 20) == bytes(range(10, 20))
    url, headers = client.seen[0]
    assert url == "https://hf.co/repo/resolve/main/f.safetensors"
    assert headers["Range"] == "bytes=10-19"


def test_bearer_token_is_sent_when_given():
    client = FakeClient(bytes(range(64)))
    src = sio.RangeSource("https://hf.co/r/resolve/main", token="hf_abc", client=client)
    src.fetch("f", 0, 4)
    assert client.seen[0][1]["Authorization"] == "Bearer hf_abc"


def test_no_authorization_header_without_a_token():
    client = FakeClient(bytes(range(64)))
    sio.RangeSource("https://hf.co/r/resolve/main", client=client).fetch("f", 0, 4)
    assert "Authorization" not in client.seen[0][1]


def test_retries_then_succeeds():
    client = FakeClient(bytes(range(64)), statuses=[500, 503, 206])
    src = sio.RangeSource("https://hf.co/r/resolve/main", client=client, backoff=0.0)
    assert src.fetch("f", 0, 8) == bytes(range(8))
    assert len(client.seen) == 3


def test_gives_up_after_the_retry_budget():
    client = FakeClient(bytes(range(64)), statuses=[500] * 10)
    src = sio.RangeSource("https://hf.co/r/resolve/main", client=client, retries=5, backoff=0.0)
    with pytest.raises(sio.SafetensorsError):
        src.fetch("f", 0, 8)
    assert len(client.seen) == 5


def test_a_short_body_is_rejected():
    client = FakeClient(bytes(range(4)))  # asked for 8 bytes, file holds 4
    src = sio.RangeSource("https://hf.co/r/resolve/main", client=client, retries=1, backoff=0.0)
    with pytest.raises(sio.SafetensorsError):
        src.fetch("f", 0, 8)


def test_range_source_feeds_iter_tensors(tmp_path):
    path, tensors = make_file(tmp_path)
    client = FakeClient(path.read_bytes())
    src = sio.RangeSource("https://hf.co/r/resolve/main", client=client)
    header, header_len = sio.read_header(path)
    refs = list(sio.refs_from_header(path.name, header, header_len).values())

    got = dict(sio.iter_tensors(refs, src))
    for name, want in tensors.items():
        assert torch.equal(got[name], want)


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #


def test_written_header_is_eight_byte_aligned(tmp_path):
    path, _ = make_file(tmp_path)
    raw = path.read_bytes()
    (header_len,) = struct.unpack_from("<Q", raw, 0)
    assert header_len % 8 == 0


def test_written_file_is_readable_by_the_safetensors_library(tmp_path):
    safetensors_torch = pytest.importorskip("safetensors.torch")
    tensors = {"w": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
    path = sio.save_safetensors(tmp_path / "x.safetensors", tensors, {"quant": "bf16"})
    loaded = safetensors_torch.load_file(str(path))
    assert torch.equal(loaded["w"], tensors["w"])


def test_save_is_atomic_and_leaves_no_part_file(tmp_path):
    path = sio.save_safetensors(tmp_path / "x.safetensors", {"w": torch.ones(2)})
    assert path.exists()
    assert not list(tmp_path.glob("*.part"))


def test_int_and_bool_dtypes_round_trip(tmp_path):
    tensors = {
        "q": torch.randint(0, 255, (4, 8), dtype=torch.uint8),
        "s": torch.randn(4, dtype=torch.float16),
        "z": torch.tensor([True, False, True]),
        "i": torch.tensor([-3, 4], dtype=torch.int32),
    }
    path = sio.save_safetensors(tmp_path / "q.safetensors", tensors)
    header, header_len = sio.read_header(path)
    refs = list(sio.refs_from_header(path.name, header, header_len).values())
    got = sio.load_tensors(refs, sio.LocalSource(tmp_path))
    for name, want in tensors.items():
        assert torch.equal(got[name], want), name


def test_unstorable_dtype_raises(tmp_path):
    with pytest.raises(sio.SafetensorsError):
        sio.save_safetensors(
            tmp_path / "c.safetensors", {"w": torch.ones(2, dtype=torch.complex64)}
        )


def test_index_json_shape_is_what_resolve_expects(tmp_path):
    """Guards the contract with the head, which ships index.json verbatim (PRD 5.3)."""
    make_file(tmp_path, "a.safetensors", {"x": torch.ones(2)})
    index = json.loads(json.dumps({"metadata": {}, "weight_map": {"x": "a.safetensors"}}))
    refs = sio.resolve(index["weight_map"], _headers(tmp_path, "a.safetensors"), ["x"])
    assert refs[0].name == "x"


def test_range_source_learns_the_commit_from_the_redirect():
    """Hugging Face names the commit on the redirect, not on the CDN reply (BAT-11)."""

    class Redirecting(FakeClient):
        def get(self, url, headers):
            reply = super().get(url, headers)
            reply.history = (FakeResponse(307, b"", {"x-repo-commit": "abc123"}),)
            return reply

    src = sio.RangeSource("https://h/r/resolve/main", client=Redirecting(bytes(64)))
    src.fetch("f", 0, 8)
    assert src.commit == "abc123"
