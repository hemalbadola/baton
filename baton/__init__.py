"""Baton: LAN-distributed LLM inference."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("baton-cluster")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0"
