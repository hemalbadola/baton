"""Baton: LAN-distributed LLM inference."""

import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _version() -> str:
    try:
        return version("baton-cluster")
    except PackageNotFoundError:
        # Running from a source tree that is not installed: read the file next to it.
        pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
        match = (
            re.search(r'(?m)^version\s*=\s*"([^"]+)"', pyproject.read_text(encoding="utf-8"))
            if pyproject.exists()
            else None
        )
        return match[1] if match else "0.0.0"


__version__ = _version()
