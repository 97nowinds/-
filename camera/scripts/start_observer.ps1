param(
    [string]$LabHost = "100.126.39.4",
    [int]$BackendPort = 5000,
    [int]$FrontendPort = 5173
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvRoot = Join-Path $projectRoot ".venv"
if (-not (Test-Path -LiteralPath $venvRoot)) {
    $venvRoot = Join-Path (Split-Path -Parent $projectRoot) ".venv"
}
$python = Join-Path $venvRoot "Scripts\python.exe"
$server = Join-Path $projectRoot "frontend_server.py"
$backendUrl = "http://${LabHost}:$BackendPort"
$observerUrl = "http://127.0.0.1:${FrontendPort}/?api=$([Uri]::EscapeDataString($backendUrl))&remote=1"

Set-Location $projectRoot
Write-Host "Observer-only laptop mode" -ForegroundColor Cyan
Write-Host "All camera and AI computation runs at $backendUrl" -ForegroundColor Yellow
Write-Host "Open: $observerUrl" -ForegroundColor Green
& $python $server --host 127.0.0.1 --port $FrontendPort --api $backendUrl
