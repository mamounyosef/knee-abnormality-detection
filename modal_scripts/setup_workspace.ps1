<#
setup_workspace.ps1 — bring one Modal workspace from empty to training-ready.

Modal volumes are per-workspace and there is no cross-workspace copy, so each
workspace that will run training has to build its own data. Doing it from
Kaggle costs no home bandwidth: the download runs inside Modal.

Per workspace, unattended:
    1. download the competition zip to the `knee-data` volume   (~1.5 h, 247 GiB)
    2. decode it into per-study shards on `knee-shards`         (~10 min, ~130 GiB)
    3. verify

Usage:
    .\modal_scripts\setup_workspace.ps1 -Session 4
    .\modal_scripts\setup_workspace.ps1 -Session 4 -SkipDownload   # zip already there

Session numbers map to workspaces in modal_setup_commands.txt.
Workspaces 1 and 3 already hold the zip.
#>
param(
    [Parameter(Mandatory = $true)][int]$Session,
    [switch]$SkipDownload,
    [switch]$SkipPreprocess
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$env:MODAL_CONFIG_PATH = "$HOME\.modal_session$Session.toml"
# Avast intercepts TLS on this machine; without this every HTTPS call fails.
$env:REQUESTS_CA_BUNDLE = "$HOME\.certs\ca-bundle.pem"
$env:SSL_CERT_FILE = $env:REQUESTS_CA_BUNDLE

Write-Host "=== workspace session $Session ===" -ForegroundColor Cyan
python -m modal profile list

if (-not $SkipDownload) {
    Write-Host "`n--- 1/3 downloading competition data (~1.5 h) ---" -ForegroundColor Cyan
    python -m modal run --detach modal_scripts/modal_download.py
}

if (-not $SkipPreprocess) {
    Write-Host "`n--- 2/3 preprocessing into shards (~10 min) ---" -ForegroundColor Cyan
    python -m modal run --detach modal_scripts/modal_preprocess.py
}

Write-Host "`n--- 3/3 verifying ---" -ForegroundColor Cyan
python -m modal run modal_scripts/modal_preprocess.py::verify

Write-Host "`nworkspace $Session ready" -ForegroundColor Green
