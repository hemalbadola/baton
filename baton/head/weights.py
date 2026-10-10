"""The model's files on the head's disk, served to the workers (PRD 12.2, BAT-43).

Only the head talks to Hugging Face. It downloads each shard file once, and
every worker range-fetches its layers from the head over the LAN. A guest laptop
needs no route to the internet, and the second start downloads nothing.

The whole model is on the head's disk, never in its memory: disk is the cheap
resource, and memory is the one that the cluster shares.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import httpx

from baton.model.cache import ModelCache, default_root

RETRIES = 5


class WeightStore:
    """`ensure()` puts every file on disk. `read()` returns a byte range of one."""

    def __init__(
        self,
        model: str,
        files: Iterable[str],
        token: str | None = None,
        revision: str = "main",
        root: str | Path | None = None,
    ) -> None:
        self.model = model
        self.files = frozenset(files)
        self.token = token
        self.revision = revision
        self.root = Path(root) if root is not None else default_root()
        self.local = Path(model) if Path(model).is_dir() else None
        self.commit: str | None = None
        self.ready = self.local is not None

    @property
    def _pointer(self) -> Path:
        return self.root / "models" / self.model / f"{self.revision}.commit"

    def path(self, file: str) -> Path:
        """Where `file` is on disk. KeyError for a name that is not part of the model:
        the name comes from a URL, and nothing outside the model may be read."""
        if file not in self.files:
            raise KeyError(file)
        if self.local is not None:
            return self.local / file
        if self.commit is None:
            raise KeyError(file)
        return ModelCache(self.model, self.commit, self.root).dir / "files" / file

    def read(self, file: str, start: int, end: int) -> bytes:
        with open(self.path(file), "rb") as fh:
            fh.seek(start)
            return fh.read(end - start)

    async def ensure(self, echo: Callable[[str], None] = print) -> None:
        """Download what is missing. Returns when every file is whole on disk."""
        if self.local is None:
            for file in sorted(self.files):
                await asyncio.to_thread(self._download, file, echo)
        self.ready = True

    def _download(self, file: str, echo: Callable[[str], None]) -> None:
        url = f"https://huggingface.co/{self.model}/resolve/{self.revision}/{file}"
        auth = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        with httpx.Client(follow_redirects=True, timeout=60.0) as client:
            try:
                # One byte tells the size and the commit. The commit is named on the
                # redirect, so a HEAD that is not followed would miss it.
                probe = client.get(url, headers={"Range": "bytes=0-0", **auth})
                probe.raise_for_status()
            except httpx.HTTPError as exc:
                # No network. A model that is already whole on disk still starts.
                if self._pointer.exists():
                    self.commit = self._pointer.read_text().strip()
                    if self.path(file).exists():
                        echo(f"offline: using {file} from the disk cache")
                        return
                raise RuntimeError(f"cannot download {file}: {exc}") from exc
            for hop in (*probe.history, probe):
                self.commit = hop.headers.get("x-repo-commit") or self.commit
            self.commit = self.commit or self.revision
            total = int(probe.headers["content-range"].rsplit("/", 1)[1])
            dest = self.path(file)
            dest.parent.mkdir(parents=True, exist_ok=True)
            self._pointer.write_text(self.commit)
            if dest.exists() and dest.stat().st_size == total:
                echo(f"{file}: on disk already ({total / 1e6:.0f} MB)")
                return

            part = dest.with_name(dest.name + ".part")
            started = time.monotonic()
            last = None
            for attempt in range(RETRIES):
                have = part.stat().st_size if part.exists() else 0
                if have >= total:
                    break
                try:
                    # A broken download continues from the byte where it stopped.
                    with client.stream(
                        "GET", url, headers={"Range": f"bytes={have}-", **auth}
                    ) as response:
                        response.raise_for_status()
                        with open(part, "ab" if response.status_code == 206 else "wb") as fh:
                            if response.status_code != 206:
                                have = 0
                            tenth = 10 * have // total
                            for chunk in response.iter_bytes(1 << 20):
                                fh.write(chunk)
                                have += len(chunk)
                                if 10 * have // total != tenth:
                                    tenth = 10 * have // total
                                    rate = have / 1e6 / max(time.monotonic() - started, 1e-3)
                                    echo(
                                        f"downloading {file}: {have / 1e6:.0f} of "
                                        f"{total / 1e6:.0f} MB ({rate:.1f} MB/s)"
                                    )
                except httpx.HTTPError as exc:
                    last = exc
                    time.sleep(2**attempt)
            if not part.exists() or part.stat().st_size != total:
                raise RuntimeError(f"cannot download {file}: {last}")
            part.replace(dest)
