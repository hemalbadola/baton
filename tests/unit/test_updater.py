"""The updater (BAT-44): version order, and the commands that replace the install."""

import sys

import pytest

from baton import updater


@pytest.mark.parametrize(
    ("latest", "current", "newer"),
    [
        ("0.2.0", "0.1.0", True),
        ("0.10.0", "0.9.0", True),  # numbers, not text
        ("0.2.0", "0.2.0", False),
        ("0.1.9", "0.2.0", False),
        (None, "0.1.0", False),
    ],
)
def test_version_order(latest, current, newer) -> None:
    assert updater.is_newer(latest, current) is newer


def test_a_source_tree_cannot_replace_itself() -> None:
    # The test venv is not a uv tool environment.
    assert updater.tool_python() is None
    assert not updater.can_apply()
    with pytest.raises(RuntimeError, match="pip install -U"):
        updater.apply()


def test_the_latest_version_is_read_from_main(monkeypatch) -> None:
    class Reply:
        status_code = 200
        text = '[project]\nname = "baton-cluster"\nversion = "9.8.7"\n'

    monkeypatch.setattr(updater.httpx, "get", lambda *a, **k: Reply())
    assert updater.latest_version(force=True) == "9.8.7"


def test_no_network_is_no_update(monkeypatch) -> None:
    def down(*a, **k):
        raise updater.httpx.ConnectError("down")

    monkeypatch.setattr(updater.httpx, "get", down)
    assert updater.latest_version(force=True) is None
    assert updater.update_at_start() is False


def test_apply_replaces_only_the_package_and_starts_again(monkeypatch, tmp_path) -> None:
    """A CUDA torch that the Windows installer swapped in must survive the update."""
    tools = tmp_path / "tools" / "baton-cluster"
    (tools / "bin").mkdir(parents=True)
    (tools / "uv-receipt.toml").write_text("")
    monkeypatch.setattr(sys, "prefix", str(tools))
    monkeypatch.setattr(sys, "executable", str(tools / "bin" / "python"))
    monkeypatch.setattr(updater, "_uv", lambda: "/u/uv")
    seen = {}
    monkeypatch.setattr(updater.subprocess, "Popen", lambda argv, **k: seen.update(argv=argv))
    monkeypatch.setattr(updater.sys, "platform", "linux")
    updater.apply(["--no-browser"])
    script = seen["argv"][-1]
    assert "uv pip install --quiet --python" in script
    assert "--reinstall-package baton-cluster" in script
    assert "--reinstall-package torch" not in script
    assert script.rstrip().endswith("app --no-update --no-browser")
