"""The CLI surface is a contract (PRD 15.1). These tests pin the names."""

import click
import pytest
from typer.testing import CliRunner

from baton.cli import app

runner = CliRunner()

COMMANDS = ["worker", "serve", "status", "bench", "selftest", "replan", "prefetch", "cache"]


def test_root_help_lists_every_command():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in COMMANDS:
        assert name in result.output


@pytest.mark.parametrize("name", COMMANDS)
def test_command_help_exits_clean(name):
    result = runner.invoke(app, [name, "--help"])
    assert result.exit_code == 0


@pytest.mark.parametrize("sub", ["ls", "rm"])
def test_cache_subcommands(sub):
    result = runner.invoke(app, ["cache", sub, "--help"])
    assert result.exit_code == 0


@pytest.mark.parametrize(
    "argv",
    [
        ["status"],
        ["bench"],
        ["selftest"],
        ["replan"],
        ["prefetch", "--model", "meta-llama/Llama-3.1-8B-Instruct"],
        ["cache", "ls"],
        ["cache", "rm", "--all"],
    ],
)
def test_body_is_not_implemented_yet(argv):
    """Every body raises NotImplementedError until its lane fills it in."""
    result = runner.invoke(app, argv)
    assert isinstance(result.exception, NotImplementedError)


def test_serve_refuses_a_quant_tier_that_cannot_load_yet():
    """int4 is the PRD default and is not wired into the decoder (BAT-10)."""
    result = runner.invoke(
        app, ["serve", "--model", "some/model", "--no-local-worker", "--quant", "int4"]
    )
    assert result.exit_code == 1
    assert "--quant none" in click.unstyle(result.output)


def test_worker_without_mdns_needs_a_head():
    result = runner.invoke(app, ["worker", "--no-mdns"])
    assert result.exit_code == 2
    assert "--head" in click.unstyle(result.output)  # rich colours the usage error


def test_required_model_flag():
    result = runner.invoke(app, ["serve"])
    assert result.exit_code != 0


def test_the_page_reads_the_same_under_a_non_utf8_locale():
    """Windows defaults to cp1252. The page showed `0â€“14` for `0–14` there."""
    import os
    import subprocess
    import sys

    env = os.environ | {"LC_ALL": "C", "PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0"}
    code = "import sys; from baton.agent import page_html; sys.stdout.buffer.write(page_html().encode())"
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, check=True)
    assert "–".encode() in out.stdout and "·".encode() in out.stdout
    assert "â€".encode() not in out.stdout
