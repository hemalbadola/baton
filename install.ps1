# Install the `baton` command on Windows, then run it with any arguments given.
#
#   irm https://raw.githubusercontent.com/hemalbadola/baton/main/install.ps1 | iex
#   & ([scriptblock]::Create((irm https://raw.githubusercontent.com/hemalbadola/baton/main/install.ps1))) worker
#
# The second form is what the web page hands out: one line installs and starts.
# $env:BATON_SOURCE overrides where Baton comes from.
#
# First Windows run (BAT-33) showed that a fresh Windows lacks the Visual C++ runtime that
# PyTorch needs. This script installs it when it is missing.
$ErrorActionPreference = "Stop"
$source = if ($env:BATON_SOURCE) { $env:BATON_SOURCE } else { "https://github.com/hemalbadola/baton/archive/refs/heads/main.zip" }

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "baton: installing uv, which brings its own Python"
    irm https://astral.sh/uv/install.ps1 | iex
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}

Write-Host "baton: installing from $source"
uv tool install --quiet --force --python 3.11 --from $source baton-cluster
if ($LASTEXITCODE -ne 0) { throw "baton: uv tool install failed" }

# PyPI serves a CPU-only torch for Windows. With an NVIDIA card, swap in the
# CUDA build that matches the installed driver, or the GPU sits idle.
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host "baton: NVIDIA GPU found, installing the CUDA build of torch"
    $python = Join-Path (uv --color never tool dir) "baton-cluster\Scripts\python.exe"
    uv pip install --quiet --python $python --torch-backend auto --reinstall-package torch torch
    if ($LASTEXITCODE -ne 0) { Write-Warning "baton: CUDA torch did not install. Baton will run on the CPU." }
}

# PyTorch loads DLLs that need the Microsoft Visual C++ runtime. Without it, `import torch`
# fails with "WinError 126 ... c10.dll". Windows asks for permission once.
$vc = Get-ItemProperty "HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" -ErrorAction SilentlyContinue
if (-not ($vc -and $vc.Installed -eq 1)) {
    Write-Host "baton: installing the Microsoft Visual C++ runtime. Click Yes when Windows asks."
    $exe = Join-Path $env:TEMP "vc_redist.x64.exe"
    Invoke-WebRequest "https://aka.ms/vs/17/release/vc_redist.x64.exe" -OutFile $exe
    $run = Start-Process $exe -ArgumentList "/install", "/quiet", "/norestart" -Verb RunAs -Wait -PassThru
    # 0 installed, 3010 installed and a restart is advised, 1638 a newer one is present.
    if ($run.ExitCode -notin 0, 1638, 3010) {
        Write-Warning "baton: the Visual C++ runtime did not install (exit $($run.ExitCode)). Install it from https://aka.ms/vs/17/release/vc_redist.x64.exe, then run baton again."
    }
}

$bin = uv --color never tool dir --bin
uv tool update-shell | Out-Null

# No `exit` here: under `irm | iex` it would close the user's PowerShell window.
if ($args.Count -gt 0) {
    & (Join-Path $bin "baton.exe") @args
} else {
    Write-Host "baton: installed. Open a new terminal, then run: baton app"
}
