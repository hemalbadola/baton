#!/bin/sh
# Install the `baton` command on macOS or Linux, then run it with any arguments given.
#
#   curl -LsSf https://raw.githubusercontent.com/hemalbadola/baton/main/install.sh | sh
#   curl -LsSf https://raw.githubusercontent.com/hemalbadola/baton/main/install.sh | sh -s -- worker
#
# The second form is what the web page hands out: one line installs and starts.
# BATON_SOURCE overrides where Baton comes from: a directory, a URL, or a git+https address.
set -eu

SOURCE="${BATON_SOURCE:-https://github.com/hemalbadola/baton/archive/refs/heads/main.zip}"

if ! command -v uv >/dev/null 2>&1; then
    echo "baton: installing uv, which brings its own Python"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    PATH="$HOME/.local/bin:$PATH"
    export PATH
fi

echo "baton: installing from $SOURCE"
uv tool install --quiet --force --python 3.11 --from "$SOURCE" baton
BIN="$(uv --color never tool dir --bin)"

# Put the command on PATH for new terminals, but only when it is not there yet:
# this edits the shell profile, and a profile that already works is left alone.
case ":$PATH:" in
    *":$BIN:"*) ;;
    *) uv tool update-shell >/dev/null 2>&1 || true ;;
esac

if [ "$#" -gt 0 ]; then
    exec "$BIN/baton" "$@"
fi
echo "baton: installed. Open a new terminal, then run: baton --help"
