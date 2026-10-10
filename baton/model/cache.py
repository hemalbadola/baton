"""Local shard cache (PRD 5.4).

Layout under the cache root::

    models/<repo_id>/<revision>/
      config.json, tokenizer.json, index.json, headers/*.json   # head only
      embed.bf16.bin                                            # N1 only, memmapped
      layers/<a>-<b>.<quant>.safetensors                        # one file per assigned range
      layers/<a>-<b>.<quant>.json                               # manifest
      head.<quant>.safetensors                                  # final_norm + lm_head, Nk only

The cache is keyed by tensor name, not by range. A worker asks for the names it
needs; :meth:`ModelCache.gather` returns the ones already on disk, from whatever
range file holds them, and the names still missing. So a cached range ``10-30``
seeds a new range ``10-25`` with no download after a re-partition.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from baton.model import safetensors_io as sio
from baton.model.quant import GROUP, QuantWeight, Tier

CACHE_ENV = "BATON_CACHE"
"""Environment variable that overrides the cache root."""

MANIFEST_VERSION = 1


class CacheError(Exception):
    """The cache is missing something, or holds something it cannot use."""


def default_root() -> Path:
    """``$BATON_CACHE`` if set, else ``~/.cache/baton``."""
    env = os.environ.get(CACHE_ENV)
    return Path(env).expanduser() if env else Path.home() / ".cache" / "baton"


def source_digest(refs: Iterable[sio.TensorRef]) -> str:
    """sha256 over the source byte ranges. A changed checkpoint invalidates the entry."""
    items = sorted((r.name, r.file, r.start, r.end) for r in refs)
    return hashlib.sha256(json.dumps(items, separators=(",", ":")).encode()).hexdigest()


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #


@dataclass
class WeightMeta:
    """How to rebuild one :class:`QuantWeight` from the tensors in the range file."""

    tier: Tier
    shape: tuple[int, int]
    group: int = GROUP


@dataclass
class Manifest:
    """The sidecar JSON beside a range file (PRD 5.4)."""

    repo_id: str
    revision: str
    start: int
    stop: int
    quant: Tier
    weights: dict[str, WeightMeta] = field(default_factory=dict)
    source_sha256: str = ""
    version: int = MANIFEST_VERSION
    created: float = 0.0

    def to_json(self) -> str:
        d = asdict(self)
        d["weights"] = {k: asdict(v) for k, v in self.weights.items()}
        return json.dumps(d, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> Manifest:
        d = json.loads(text)
        weights = {
            k: WeightMeta(v["tier"], tuple(v["shape"]), v.get("group", GROUP))
            for k, v in d.pop("weights", {}).items()
        }
        return cls(weights=weights, **d)


# --------------------------------------------------------------------------- #
# The cache
# --------------------------------------------------------------------------- #


class ModelCache:
    """One model revision on one machine's disk."""

    def __init__(self, repo_id: str, revision: str, root: str | Path | None = None) -> None:
        self.repo_id = repo_id
        self.revision = revision
        self.root = Path(root) if root is not None else default_root()

    # -- paths -------------------------------------------------------------- #

    @property
    def dir(self) -> Path:
        """``<root>/models/<repo_id>/<revision>``. A slash in the repo id becomes a directory."""
        return self.root / "models" / self.repo_id / self.revision

    @property
    def layers_dir(self) -> Path:
        return self.dir / "layers"

    @property
    def headers_dir(self) -> Path:
        return self.dir / "headers"

    def range_path(self, start: int, stop: int, quant: Tier) -> Path:
        return self.layers_dir / f"{start}-{stop}.{quant}.safetensors"

    def manifest_path(self, start: int, stop: int, quant: Tier) -> Path:
        return self.layers_dir / f"{start}-{stop}.{quant}.json"

    def head_path(self, quant: Tier) -> Path:
        return self.dir / f"head.{quant}.safetensors"

    @property
    def embed_path(self) -> Path:
        return self.dir / "embed.bf16.bin"

    # -- head-only metadata ------------------------------------------------- #

    def put_meta(self, name: str, obj: object) -> Path:
        """Store ``config.json``, ``index.json`` or a parsed shard header."""
        path = self.dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(path, json.dumps(obj).encode())
        return path

    def get_meta(self, name: str) -> object:
        path = self.dir / name
        if not path.exists():
            raise CacheError(f"{path} is not cached")
        return json.loads(path.read_text())

    def has_meta(self, name: str) -> bool:
        return (self.dir / name).exists()

    # -- range files -------------------------------------------------------- #

    def manifests(self, quant: Tier | None = None) -> list[Manifest]:
        """Every valid manifest for this revision, newest last. Bad files are skipped."""
        out: list[Manifest] = []
        if not self.layers_dir.is_dir():
            return out
        for path in sorted(self.layers_dir.glob("*.json")):
            try:
                man = Manifest.from_json(path.read_text())
            except (json.JSONDecodeError, TypeError, KeyError, ValueError):
                continue  # a truncated or older manifest is treated as absent
            if man.version != MANIFEST_VERSION or man.revision != self.revision:
                continue
            if quant is not None and man.quant != quant:
                continue
            if not self.range_path(man.start, man.stop, man.quant).exists():
                continue
            out.append(man)
        return out

    def has_range(self, start: int, stop: int, quant: Tier, digest: str | None = None) -> bool:
        """The PRD 5.4 hit rule: start, stop, quant and revision must all match.

        Pass ``digest`` to also require that the source byte ranges are unchanged.
        """
        path = self.manifest_path(start, stop, quant)
        if not path.exists() or not self.range_path(start, stop, quant).exists():
            return False
        try:
            man = Manifest.from_json(path.read_text())
        except (json.JSONDecodeError, TypeError, KeyError, ValueError):
            return False
        if (man.revision, man.start, man.stop, man.quant) != (self.revision, start, stop, quant):
            return False
        return digest is None or man.source_sha256 == digest

    def put_range(
        self,
        start: int,
        stop: int,
        quant: Tier,
        weights: Mapping[str, QuantWeight],
        extra: Mapping[str, torch.Tensor] | None = None,
        digest: str = "",
    ) -> Path:
        """Write one quantized layer range plus its manifest.

        ``extra`` carries tensors that are not quantized matrices, such as the
        RMSNorm weights and the attention biases. They are stored as they are.
        """
        tensors: dict[str, torch.Tensor] = {}
        meta: dict[str, WeightMeta] = {}
        for name, qw in weights.items():
            for suffix, t in qw.to_tensors().items():
                tensors[f"{name}.{suffix}"] = t
            meta[name] = WeightMeta(qw.tier, qw.shape, qw.group)
        for name, t in (extra or {}).items():
            if name in tensors:
                raise CacheError(f"{name!r} collides with a quantized weight")
            tensors[name] = t

        man = Manifest(
            repo_id=self.repo_id,
            revision=self.revision,
            start=start,
            stop=stop,
            quant=quant,
            weights=meta,
            source_sha256=digest,
            created=time.time(),
        )
        path = self.range_path(start, stop, quant)
        sio.save_safetensors(
            path, tensors, {"repo_id": self.repo_id, "revision": self.revision, "quant": quant}
        )
        _atomic_write(self.manifest_path(start, stop, quant), man.to_json().encode())
        return path

    def read_range(self, start: int, stop: int, quant: Tier) -> dict[str, QuantWeight]:
        """Load one whole range file back into quantized weights."""
        if not self.has_range(start, stop, quant):
            raise CacheError(f"no cached range {start}-{stop}.{quant} for {self.revision}")
        man = Manifest.from_json(self.manifest_path(start, stop, quant).read_text())
        tensors = _read_all(self.range_path(start, stop, quant))
        return _rebuild(man, tensors, list(man.weights))

    # -- per-name lookup across every cached range -------------------------- #

    def locate(self, names: Sequence[str], quant: Tier) -> dict[str, tuple[int, int]]:
        """Map each requested weight name to the cached range that holds it."""
        found: dict[str, tuple[int, int]] = {}
        for man in self.manifests(quant):
            for name in names:
                if name not in found and name in man.weights:
                    found[name] = (man.start, man.stop)
        return found

    def gather(self, names: Sequence[str], quant: Tier) -> tuple[dict[str, QuantWeight], list[str]]:
        """Collect what is cached and report what is not (PRD 5.4, last paragraph).

        This is the check a worker runs before it fetches anything. A re-partition
        that narrows a range downloads nothing.
        """
        where = self.locate(names, quant)
        by_range: dict[tuple[int, int], list[str]] = {}
        for name, key in where.items():
            by_range.setdefault(key, []).append(name)

        out: dict[str, QuantWeight] = {}
        for (start, stop), wanted in by_range.items():
            man = Manifest.from_json(self.manifest_path(start, stop, quant).read_text())
            tensors = _read_all(self.range_path(start, stop, quant))
            out.update(_rebuild(man, tensors, wanted))
        return out, [n for n in names if n not in out]

    # -- embedding table ---------------------------------------------------- #

    def put_embed(self, table: torch.Tensor) -> Path:
        """Write the bf16 embedding table as a flat file for memmap (PRD 5.3, D5)."""
        if table.dtype != torch.bfloat16:
            table = table.to(torch.bfloat16)
        self.dir.mkdir(parents=True, exist_ok=True)
        raw = table.contiguous().view(torch.uint8).reshape(-1).numpy().tobytes()
        _atomic_write(self.embed_path, raw)
        return self.embed_path

    def open_embed(self, vocab: int, hidden: int):
        """Memmap the embedding table. Costs no resident memory; the page cache holds hot rows."""
        import numpy as np

        if not self.embed_path.exists():
            raise CacheError(f"{self.embed_path} is not cached")
        want = vocab * hidden * 2
        got = self.embed_path.stat().st_size
        if got != want:
            raise CacheError(f"{self.embed_path}: {got} bytes, expected {want}")
        return np.memmap(self.embed_path, dtype="uint8", mode="r", shape=(vocab, hidden * 2))

    @staticmethod
    def embed_rows(mm, ids: Sequence[int], dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        """Read the given token rows out of a memmapped table and cast them."""
        raw = bytearray(mm[list(ids)].tobytes())
        rows = torch.frombuffer(raw, dtype=torch.bfloat16).reshape(len(ids), -1)
        return rows.to(dtype)

    # -- housekeeping ------------------------------------------------------- #

    def disk_bytes(self) -> int:
        """Total bytes this revision occupies."""
        if not self.dir.exists():
            return 0
        return sum(p.stat().st_size for p in self.dir.rglob("*") if p.is_file())

    def evict_range(self, start: int, stop: int, quant: Tier) -> None:
        """Drop one range file and its manifest. Missing files are not an error."""
        self.range_path(start, stop, quant).unlink(missing_ok=True)
        self.manifest_path(start, stop, quant).unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Dense tensors, as fetched (BAT-11)
# --------------------------------------------------------------------------- #

FETCH_SPAN_BYTES = 64 << 20
"""Largest ranged request for tensors that sit back to back. A request costs about
one second before the first byte, so one request per tensor made a 1 GB model take
five minutes. The span is held in memory once, beside the model."""


class BlobCache:
    """Keeps every tensor a worker fetched, one file per tensor.

    Layout: ``<root>/models/<repo_id>/<commit>/blobs/<file>/<start>-<end>``, the
    raw bytes of that range. The key is the tensor, not the layer range, so a
    new plan with a different range fetches only the tensors it does not have.

    The directory is named by the repo commit, so a checkpoint that changes
    upstream never mixes with the old bytes. With no network, the last commit
    seen for the revision is used, and a cached shard loads offline.
    """

    def __init__(
        self, root: str | Path, repo_id: str, revision: str, source: sio.ByteSource
    ) -> None:
        self.root = Path(root)
        self.repo_id = repo_id
        self.revision = revision
        self.source = source
        self.commit: str | None = None
        self.fetched_bytes = 0
        """Bytes that came from the network in this load. Zero means a full cache hit."""

    @property
    def _pointer(self) -> Path:
        return self.root / "models" / self.repo_id / f"{self.revision}.commit"

    def _dir(self) -> Path:
        assert self.commit is not None
        return ModelCache(self.repo_id, self.commit, self.root).dir

    def header(self, file: str) -> tuple[dict, int]:
        """The shard header, from the network when there is one, else from disk."""
        try:
            header, header_len = sio.fetch_header(self.source, file)
        except sio.SafetensorsError:
            if self.commit is None and self._pointer.exists():
                self.commit = self._pointer.read_text().strip()
            path = self._dir() / "headers" / f"{file}.json" if self.commit else None
            if path is None or not path.exists():
                raise
            saved = json.loads(path.read_text())
            return saved["header"], saved["header_len"]
        if self.commit is None:
            # A source that names no commit still gets a stable directory.
            self.commit = getattr(self.source, "commit", None) or self.revision
            _try_write(self._pointer, self.commit.encode())
        _try_write(
            self._dir() / "headers" / f"{file}.json",
            json.dumps({"header": header, "header_len": header_len}).encode(),
        )
        return header, header_len

    def _blob(self, ref: sio.TensorRef) -> Path:
        return self._dir() / "blobs" / ref.file / f"{ref.start}-{ref.end}"

    def iter_tensors(
        self, refs: Sequence[sio.TensorRef], verify: bool = True
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Stream the tensors: the cached ones from disk, the rest from the source.

        Call `header` for each file first. That is what learns the commit.
        """
        missing: list[sio.TensorRef] = []
        for ref in refs:
            path = self._blob(ref)
            try:
                hit = path.stat().st_size == ref.nbytes
            except OSError:
                hit = False
            if not hit:
                missing.append(ref)
                continue
            tensor = sio.slice_tensor(path.read_bytes(), ref.start, ref)
            if verify:
                sio.check_finite(ref.name, tensor)
            yield ref.name, tensor
        for span in sio.coalesce(missing, max_gap=0, max_span=FETCH_SPAN_BYTES):
            blob = self.source.fetch(span.file, span.start, span.end)
            self.fetched_bytes += len(blob)
            for ref in span.refs:
                tensor = sio.slice_tensor(blob, span.start, ref)
                if verify:
                    sio.check_finite(ref.name, tensor)
                _try_write(self._blob(ref), blob[ref.start - span.start : ref.end - span.start])
                yield ref.name, tensor
            del blob


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _try_write(path: Path, data: bytes) -> None:
    """Cache one file. A full disk costs the cache, never the load."""
    try:
        _atomic_write(path, data)
    except OSError:
        path.with_suffix(path.suffix + ".part").unlink(missing_ok=True)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)


def _read_all(path: Path) -> dict[str, torch.Tensor]:
    header, header_len = sio.read_header(path)
    refs = list(sio.refs_from_header(path.name, header, header_len).values())
    return sio.load_tensors(refs, sio.LocalSource(path.parent), verify=False)


def _rebuild(
    man: Manifest, tensors: Mapping[str, torch.Tensor], names: Sequence[str]
) -> dict[str, QuantWeight]:
    out: dict[str, QuantWeight] = {}
    for name in names:
        meta = man.weights.get(name)
        if meta is None:
            raise CacheError(f"{name!r} is not in the manifest for {man.start}-{man.stop}")
        parts = {
            suffix: tensors[f"{name}.{suffix}"]
            for suffix in ("qweight", "scale", "zero")
            if f"{name}.{suffix}" in tensors
        }
        if "qweight" not in parts:
            raise CacheError(f"{name!r}: the range file has no qweight")
        out[name] = QuantWeight.from_tensors(meta.tier, meta.shape, parts, meta.group)
    return out
