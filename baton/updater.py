"""Check for a newer Baton and replace the installed copy (BAT-44).

Where the truth lives: `version` in `pyproject.toml` on the `main` branch. Every
change that users must get raises that number. The check reads one small raw file.

The update replaces only the `baton-cluster` package inside the tool environment
that the installer made. It keeps the rest of that environment, so a CUDA build of
PyTorch that the Windows installer swapped in survives. Then it starts `baton app`
again. The running program has to end first, because Windows locks a running exe.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx

from baton import __version__

RAW_PYPROJECT = "https://raw.githubusercontent.com/hemalbadola/baton/main/pyproject.toml"
SOURCE_ZIP = "https://github.com/hemalbadola/baton/archive/refs/heads/main.zip"
CHECK_EVERY_S = 600.0

_cache: tuple[float, str | None] = (0.0, None)


def parse(text: str) -> tuple[int, ...]:
    """`0.2.1` to `(0, 2, 1)`. Anything that is not numbers sorts as zero."""
    return tuple(int(p) if p.isdigit() else 0 for p in re.split(r"[.\-+]", text.strip())[:4])


def is_newer(latest: str | None, current: str = __version__) -> bool:
    return latest is not None and parse(latest) > parse(current)


def latest_version(force: bool = False) -> str | None:
    """The version on `main`, or None when the network is not there. Cached 10 minutes."""
    global _cache
    at, value = _cache
    if not force and time.monotonic() - at < CHECK_EVERY_S and at:
        return value
    found: str | None = None
    try:
        reply = httpx.get(RAW_PYPROJECT, timeout=4.0, headers={"Cache-Control": "no-cache"})
        match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', reply.text)
        found = match[1] if reply.status_code == 200 and match else None
    except httpx.HTTPError:
        pass
    _cache = (time.monotonic(), found)
    return found


def _uv() -> str | None:
    found = shutil.which("uv")
    if found:
        return found
    home = Path.home() / ".local" / "bin"
    for name in ("uv", "uv.exe"):
        if (home / name).exists():
            return str(home / name)
    return None


def tool_python() -> Path | None:
    """The interpreter of the uv tool environment that this process runs in, or None
    when Baton came from pip, brew or a source tree. Those update their own way."""
    prefix = Path(sys.prefix)
    if prefix.parent.name == "tools" and (prefix / "uv-receipt.toml").exists():
        return Path(sys.executable)
    return None


def can_apply() -> bool:
    return tool_python() is not None and _uv() is not None


def how_to_update() -> str:
    """What to tell a user whose install this module cannot replace."""
    if tool_python() is not None:
        return "Install uv, then run the install line again."
    return "pip install -U baton-cluster   (or: brew upgrade baton)"


def apply(args: list[str] | None = None) -> None:
    """Start a helper that replaces the package after this process ends, then relaunches.

    The caller must end the process right after this returns.
    """
    python, uv = tool_python(), _uv()
    if python is None or uv is None:
        raise RuntimeError(how_to_update())
    relaunch = [str(python), "-m", "baton.cli", "app", "--no-update", *(args or [])]
    install = [uv, "pip", "install", "--quiet", "--python", str(python),
               "--reinstall-package", "baton-cluster", SOURCE_ZIP]  # fmt: skip
    if sys.platform == "win32":
        quote = subprocess.list2cmdline
        script = f"timeout /t 3 /nobreak >nul & {quote(install)} & {quote(relaunch)}"
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        subprocess.Popen(["cmd", "/c", script], creationflags=flags, close_fds=True)
    else:
        script = f"sleep 2; {shlex.join(install)} ; exec {shlex.join(relaunch)}"
        subprocess.Popen(
            ["sh", "-c", script], start_new_session=True, close_fds=True,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )  # fmt: skip


def update_at_start() -> bool:
    """Replace the install before the page opens, when a newer one exists.

    True when a helper took over: the caller must end the process at once. Any
    failure here is silent. A missing network must never stop a demo.
    """
    try:
        if can_apply() and is_newer(latest_version(force=True)):
            apply(sys.argv[sys.argv.index("app") + 1 :] if "app" in sys.argv else [])
            return True
    except Exception:  # noqa: BLE001 - the page opens with the version that is installed
        return False
    return False


def exit_now() -> None:
    os._exit(0)
