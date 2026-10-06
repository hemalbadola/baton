# Install the `baton` command on Windows, then run it with any arguments given.
#
#   irm https://raw.githubusercontent.com/hemalbadola/baton/main/install.ps1 | iex
#   & ([scriptblock]::Create((irm https://raw.githubusercontent.com/hemalbadola/baton/main/install.ps1))) worker
#
# The second form is what the web page hands out: one line installs and starts.
# $env:BATON_SOURCE overrides where Baton comes from.
#
# NOT YET RUN ON WINDOWS (ticket BAT-31). Written from the uv documentation.
$ErrorActionPreference = "Stop"
$source = if ($env:BATON_SOURCE) { $env:BATON_SOURCE } else { "https://github.com/hemalbadola/baton/archive/refs/heads/main.zip" }

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "baton: installing uv, which brings its own Python"
    irm https://astral.sh/uv/install.ps1 | iex
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

Write-Host "baton: installing from $source"
uv tool install --quiet --force --python 3.11 --from $source baton
if ($LASTEXITCODE -ne 0) { throw "baton: uv tool install failed" }

# PyPI serves a CPU-only torch for Windows. With an NVIDIA card, swap in the
# CUDA build that matches the installed driver, or the GPU sits idle.
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host "baton: NVIDIA GPU found, installing the CUDA build of torch"
    $python = Join-Path (uv --color never tool dir) "baton\Scripts\python.exe"
    uv pip install --quiet --python $python --torch-backend auto --reinstall-package torch torch
    if ($LASTEXITCODE -ne 0) { Write-Warning "baton: CUDA torch did not install. Baton will run on the CPU." }
}

$bin = uv --color never tool dir --bin
uv tool update-shell | Out-Null

# No `exit` here: under `irm | iex` it would close the user's PowerShell window.
if ($args.Count -gt 0) {
    & (Join-Path $bin "baton.exe") @args
} else {
    Write-Host "baton: installed. Open a new terminal, then run: baton --help"
}
