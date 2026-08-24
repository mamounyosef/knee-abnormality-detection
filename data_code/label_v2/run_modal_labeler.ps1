# run_modal_labeler.ps1 - point the labeler at a Modal endpoint and test it, in one go.
#
# Smoke test FIRST (one call, prints the raw reply), then the held-out 29 only if it
# parsed. A batch against an unverified endpoint costs a quota to learn what one call
# shows for free.
#
#   .\data_code\label_v2\run_modal_labeler.ps1 `
#       -Url "https://....modal.direct" -Key "wk-..." -Secret "ws-..." `
#       -Model "Qwen/Qwen3.8-27B" -Tag qwen27b-newprompt
#
# Add -FewShot to prepend the calibration block. Add -Workers N (default 4).
# Add -Mode hard for 0/1 labels instead of probabilities - note that a hard run
# cannot be compared to a soft one on AUROC, only on balanced accuracy / F1.

param(
  [Parameter(Mandatory=$true)][string]$Url,
  [Parameter(Mandatory=$true)][string]$Key,
  [Parameter(Mandatory=$true)][string]$Secret,
  [string]$Model   = "Qwen/Qwen3.8-27B",
  [string]$Tag     = "modal-run",
  [int]$Workers    = 4,
  [int]$TimeoutSec = 600,
  [switch]$FewShot,
  [ValidateSet("soft","hard")][string]$Mode = "soft"
)

$ErrorActionPreference = "Stop"
Set-Location "d:\My Projects\knee_abnormality_detection"

# Avast intercepts HTTPS on this machine; without these every request fails at TLS
$env:REQUESTS_CA_BUNDLE = "$HOME\.certs\ca-bundle.pem"
$env:SSL_CERT_FILE      = $env:REQUESTS_CA_BUNDLE
$env:PYTHONUTF8         = "1"

if (Test-Path .env) { Copy-Item .env ".env.before-$Tag.bak" -Force }

# /v1 is what the OpenAI-compatible path lives under; tolerate it being given either way
$base = $Url.TrimEnd('/')
if (-not $base.EndsWith('/v1')) { $base = "$base/v1" }

# OLLAMA_API_KEY must go: load_env prefers it for auth whenever it is present
(Get-Content .env) |
  Where-Object { $_ -notmatch '^(MODAL_LLM_BASE_URL|MODAL_LLM_MODEL|MODAL_KEY|MODAL_SECRET|OLLAMA_API_KEY)=' } |
  Set-Content .env -Encoding utf8
Add-Content .env -Encoding utf8 "MODAL_LLM_BASE_URL=$base"
Add-Content .env -Encoding utf8 "MODAL_LLM_MODEL=$Model"
Add-Content .env -Encoding utf8 "MODAL_KEY=$Key"
Add-Content .env -Encoding utf8 "MODAL_SECRET=$Secret"

Write-Host "`n=== endpoint ===" -ForegroundColor Cyan
Write-Host "  $base"
Write-Host "  $Model`n"

Write-Host "=== smoke test (1 call) ===" -ForegroundColor Cyan
python -u data_code/label_v2/smoke.py --mode $Mode
if ($LASTEXITCODE -ne 0) {
  Write-Host "`nSMOKE TEST FAILED - not spending the batch." -ForegroundColor Red
  exit 1
}

Write-Host "`n=== held-out 29 ===" -ForegroundColor Cyan
$fs = @()
if ($FewShot) { $fs = @("--fewshot") }
python data_code/03_label_reports.py --validate --holdout @fs --mode $Mode --tag $Tag --workers $Workers --timeout $TimeoutSec

Write-Host "`n=== comparison ===" -ForegroundColor Cyan
python data_code/label_v2/compare_labelers.py
