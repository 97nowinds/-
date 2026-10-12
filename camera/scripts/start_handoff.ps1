param(
    [string]$Python,
    [string]$BackendUrl = "http://127.0.0.1:15001",
    [string]$CredentialsFile
)
$ErrorActionPreference = "Stop"
$handoffRoot = Split-Path -Parent $PSScriptRoot
if (-not $Python) {
    $Python = Join-Path $handoffRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $Python)) {
        $Python = Join-Path (Split-Path -Parent $handoffRoot) ".venv\Scripts\python.exe"
    }
}
if (-not $CredentialsFile) {
    $CredentialsFile = Join-Path $handoffRoot "runtime\handoff-credentials.json"
}
if (-not (Test-Path -LiteralPath $CredentialsFile)) {
    throw "Missing server credentials file; provision image_read_token and sync_read_token privately first."
}
& $Python (Join-Path $handoffRoot "map_handoff.py") --api $BackendUrl --port 5175 --credentials-file $CredentialsFile --sample-archive (Join-Path $handoffRoot "runtime\handoff-samples\archive")
