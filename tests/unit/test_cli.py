"""The CLI surface is a contract (PRD 15.1). These tests pin the names."""

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
        ["worker"],
        ["serve", "--model", "meta-llama/Llama-3.1-8B-Instruct"],
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


def test_required_model_flag():
    result = runner.invoke(app, ["serve"])
    assert result.exit_code != 0
