"""Safetensors partial loading (PRD 5.3) and HTTP Range fetch (PRD 12.1).

A safetensors file is: 8-byte little-endian u64 header length, a JSON header of
that length, then raw tensor bytes. The header maps a tensor name to
``{dtype, shape, data_offsets: [start, end]}``, where the offsets are relative
to the end of the header. Reading one tensor therefore needs only the absolute
range ``[8 + header_len + start, 8 + header_len + end)``.

Nothing here loads a whole file. The worker resolves the tensors of its own
layer range, coalesces the ranges, and streams them one tensor at a time.
"""

from __future__ import annotations

import json
import struct
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Self

import torch

HEADER_LEN_BYTES = 8
"""Width of the little-endian u64 that prefixes every safetensors file."""

MAX_HEADER_BYTES = 100_000_000
"""Refuse headers larger than this. A real header is a few MB at most."""

COALESCE_GAP = 1 << 20
"""Tensors in one file closer than this are fetched in a single request (PRD 12.1)."""

DTYPES: dict[str, torch.dtype] = {
    "BOOL": torch.bool,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "F64": torch.float64,
}

DTYPE_NAMES: dict[torch.dtype, str] = {v: k for k, v in DTYPES.items()}


class SafetensorsError(Exception):
    """A safetensors file or header is malformed."""


# --------------------------------------------------------------------------- #
# Header parsing
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TensorRef:
    """Where one tensor lives: a file, an absolute byte range, a dtype, a shape."""

    name: str
    file: str
    dtype: str
    shape: tuple[int, ...]
    start: int
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.start

    @property
    def torch_dtype(self) -> torch.dtype:
        try:
            return DTYPES[self.dtype]
        except KeyError:
            raise SafetensorsError(f"unsupported dtype {self.dtype!r} for {self.name!r}") from None

    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n


def parse_header(blob: bytes) -> tuple[dict, int]:
    """Parse the 8-byte length prefix and the JSON header from the start of a file.

    ``blob`` must hold at least ``8 + header_len`` bytes. Returns the decoded
    header and its length in bytes.
    """
    if len(blob) < HEADER_LEN_BYTES:
        raise SafetensorsError("file shorter than the 8-byte header length prefix")
    (header_len,) = struct.unpack_from("<Q", blob, 0)
    if header_len == 0 or header_len > MAX_HEADER_BYTES:
        raise SafetensorsError(f"implausible header length {header_len}")
    stop = HEADER_LEN_BYTES + header_len
    if len(blob) < stop:
        raise SafetensorsError(f"need {stop} bytes of header, got {len(blob)}")
    try:
        header = json.loads(blob[HEADER_LEN_BYTES:stop])
    except json.JSONDecodeError as exc:
        raise SafetensorsError(f"header is not valid JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise SafetensorsError("header is not a JSON object")
    return header, header_len


def read_header(path: str | Path) -> tuple[dict, int]:
    """Read the JSON header of a local safetensors file. Reads the header only."""
    with open(path, "rb") as fh:
        prefix = fh.read(HEADER_LEN_BYTES)
        if len(prefix) < HEADER_LEN_BYTES:
            raise SafetensorsError(f"{path}: shorter than the header length prefix")
        (header_len,) = struct.unpack_from("<Q", prefix, 0)
        if header_len == 0 or header_len > MAX_HEADER_BYTES:
            raise SafetensorsError(f"{path}: implausible header length {header_len}")
        return parse_header(prefix + fh.read(header_len))


def refs_from_header(file: str, header: Mapping, header_len: int) -> dict[str, TensorRef]:
    """Turn one parsed header into absolute byte ranges, keyed by tensor name."""
    base = HEADER_LEN_BYTES + header_len
    refs: dict[str, TensorRef] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        try:
            start, end = meta["data_offsets"]
            dtype = meta["dtype"]
            shape = tuple(int(d) for d in meta["shape"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SafetensorsError(f"{file}: bad header entry for {name!r}") from exc
        if end < start:
            raise SafetensorsError(f"{file}: {name!r} has end {end} before start {start}")
        refs[name] = TensorRef(name, file, dtype, shape, base + int(start), base + int(end))
    return refs


def resolve(
    weight_map: Mapping[str, str],
    headers: Mapping[str, tuple[Mapping, int]],
    names: Iterable[str],
) -> list[TensorRef]:
    """Resolve tensor names to byte ranges (PRD 5.3 step 2).

    ``weight_map`` is ``model.safetensors.index.json["weight_map"]``: tensor name
    to shard file. ``headers`` maps a shard file to its parsed header and header
    length. For a single-file checkpoint, pass an empty ``weight_map``; the one
    header in ``headers`` is then searched directly.
    """
    tables = {f: refs_from_header(f, h, n) for f, (h, n) in headers.items()}
    out: list[TensorRef] = []
    for name in names:
        file = weight_map.get(name)
        if file is None:
            hits = [f for f, t in tables.items() if name in t]
            if len(hits) != 1:
                raise SafetensorsError(f"{name!r} not found in the index or in any header")
            file = hits[0]
        table = tables.get(file)
        if table is None:
            raise SafetensorsError(f"no header supplied for shard file {file!r}")
        if name not in table:
            raise SafetensorsError(f"{name!r} is mapped to {file!r} but absent from its header")
        out.append(table[name])
    return out


# --------------------------------------------------------------------------- #
# Range planning
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ByteRange:
    """One fetch: a contiguous span of one file that covers several tensors."""

    file: str
    start: int
    end: int
    refs: tuple[TensorRef, ...]

    @property
    def nbytes(self) -> int:
        return self.end - self.start


def coalesce(refs: Sequence[TensorRef], max_gap: int = COALESCE_GAP) -> list[ByteRange]:
    """Group refs of the same file into fetches, merging gaps under ``max_gap``.

    A 70B layer's nine tensors usually sit in one file and merge into one or two
    requests (PRD 12.1).
    """
    ordered = sorted(refs, key=lambda r: (r.file, r.start, r.end))
    out: list[ByteRange] = []
    group: list[TensorRef] = []
    for ref in ordered:
        if group and ref.file == group[0].file and ref.start - group[-1].end <= max_gap:
            group.append(ref)
            continue
        if group:
            out.append(_range_of(group))
        group = [ref]
    if group:
        out.append(_range_of(group))
    return out


def _range_of(group: list[TensorRef]) -> ByteRange:
    return ByteRange(
        file=group[0].file,
        start=group[0].start,
        end=max(r.end for r in group),
        refs=tuple(group),
    )


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #


class ByteSource(Protocol):
    """Anything that can return the bytes ``[start, end)`` of a named file."""

    def fetch(self, file: str, start: int, end: int) -> bytes: ...


class LocalSource:
    """Read ranges from safetensors files in a local directory."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def fetch(self, file: str, start: int, end: int) -> bytes:
        with open(self.root / file, "rb") as fh:
            fh.seek(start)
            blob = fh.read(end - start)
        if len(blob) != end - start:
            raise SafetensorsError(f"{file}: read {len(blob)} bytes, expected {end - start}")
        return blob


class RangeSource:
    """HTTP Range fetch from Hugging Face, or from the head proxy (PRD 12.1, 12.2).

    ``base_url`` is everything before the file name, for example
    ``https://huggingface.co/<repo>/resolve/<revision>``. Retries five times with
    exponential backoff from one second, as the PRD requires.
    """

    def __init__(
        self,
        base_url: str,
        token: str | None = None,
        client=None,
        retries: int = 5,
        backoff: float = 1.0,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.retries = retries
        self.backoff = backoff
        self._timeout = timeout
        self._client = client
        self._owned = client is None

    @property
    def client(self):
        if self._client is None:
            import httpx  # imported lazily: the local path needs no HTTP stack

            self._client = httpx.Client(timeout=self._timeout, follow_redirects=True)
        return self._client

    def close(self) -> None:
        if self._owned and self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def fetch(self, file: str, start: int, end: int) -> bytes:
        want = end - start
        headers = {"Range": f"bytes={start}-{end - 1}"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        url = f"{self.base_url}/{file}"
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                resp = self.client.get(url, headers=headers)
                if resp.status_code not in (200, 206):
                    raise SafetensorsError(f"{url}: HTTP {resp.status_code}")
                blob = resp.content
                # Safetensors carries no per-tensor hash. Byte length is the only
                # check available at this layer (PRD 12.1).
                if len(blob) != want:
                    raise SafetensorsError(f"{url}: got {len(blob)} bytes, expected {want}")
                return blob
            except Exception as exc:  # noqa: BLE001 - retried, then re-raised below
                last = exc
                if attempt + 1 < self.retries:
                    time.sleep(self.backoff * (2**attempt))
        raise SafetensorsError(f"{url}: range fetch failed after {self.retries} tries: {last}")


# --------------------------------------------------------------------------- #
# Tensor materialization
# --------------------------------------------------------------------------- #


def slice_tensor(blob: bytes | bytearray, blob_start: int, ref: TensorRef) -> torch.Tensor:
    """Cut one tensor out of a fetched span. ``blob_start`` is the span's file offset."""
    lo = ref.start - blob_start
    hi = ref.end - blob_start
    if lo < 0 or hi > len(blob):
        raise SafetensorsError(f"{ref.name!r} lies outside the fetched span")
    raw = bytearray(blob[lo:hi])  # copy: frombuffer needs a writable buffer
    flat = torch.frombuffer(raw, dtype=ref.torch_dtype)
    if flat.numel() != ref.numel():
        raise SafetensorsError(
            f"{ref.name!r}: {flat.numel()} elements in {ref.nbytes} bytes, "
            f"shape {ref.shape} wants {ref.numel()}"
        )
    return flat.reshape(ref.shape)


def check_finite(name: str, tensor: torch.Tensor) -> torch.Tensor:
    """Scan for NaN and Inf after the cast (PRD 12.1). Returns the tensor unchanged."""
    if tensor.is_floating_point() and not torch.isfinite(tensor).all():
        raise SafetensorsError(f"{name!r} holds NaN or Inf after the cast")
    return tensor


def iter_tensors(
    refs: Sequence[TensorRef],
    source: ByteSource,
    max_gap: int = COALESCE_GAP,
    verify: bool = True,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Stream the given tensors, one at a time, in file order.

    One span is held at a time, so peak host memory is one coalesced fetch. The
    caller quantizes and frees each tensor before the next arrives (PRD 5.3 step 5).
    """
    for span in coalesce(refs, max_gap):
        blob = source.fetch(span.file, span.start, span.end)
        for ref in span.refs:
            tensor = slice_tensor(blob, span.start, ref)
            if verify:
                check_finite(ref.name, tensor)
            yield ref.name, tensor
        del blob


def load_tensors(
    refs: Sequence[TensorRef],
    source: ByteSource,
    max_gap: int = COALESCE_GAP,
    verify: bool = True,
) -> dict[str, torch.Tensor]:
    """Eager form of :func:`iter_tensors`. Holds every requested tensor at once."""
    return dict(iter_tensors(refs, source, max_gap, verify))


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #


def _tensor_bytes(t: torch.Tensor) -> bytes:
    return t.detach().cpu().contiguous().view(torch.uint8).reshape(-1).numpy().tobytes()


def save_safetensors(
    path: str | Path,
    tensors: Mapping[str, torch.Tensor],
    metadata: Mapping[str, str] | None = None,
) -> Path:
    """Write a safetensors file. The shard cache uses this for quantized ranges (PRD 5.4)."""
    header: dict[str, object] = {}
    if metadata:
        header["__metadata__"] = dict(metadata)
    blobs: list[bytes] = []
    offset = 0
    for name, t in tensors.items():
        if t.dtype not in DTYPE_NAMES:
            raise SafetensorsError(f"{name!r}: cannot store dtype {t.dtype}")
        raw = _tensor_bytes(t)
        header[name] = {
            "dtype": DTYPE_NAMES[t.dtype],
            "shape": list(t.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        blobs.append(raw)
        offset += len(raw)

    body = json.dumps(header, separators=(",", ":")).encode()
    pad = (-len(body)) % 8  # the data block must start 8-byte aligned
    body += b" " * pad

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with open(tmp, "wb") as fh:
        fh.write(struct.pack("<Q", len(body)))
        fh.write(body)
        for raw in blobs:
            fh.write(raw)
    tmp.replace(path)  # atomic: a reader never sees a half-written shard
    return path
