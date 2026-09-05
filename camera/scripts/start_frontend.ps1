param(
    [string]$BackendUrl = "http://127.0.0.1:5000",
    [int]$Port = 5173
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvRoot = Join-Path $projectRoot ".venv"
if (-not (Test-Path -LiteralPath $venvRoot)) {
    $venvRoot = Join-Path (Split-Path -Parent $projectRoot) ".venv"
}
$python = Join-Path $venvRoot "Scripts\python.exe"
$server = Join-Path $projectRoot "frontend_server.py"

Set-Location $projectRoot
Write-Host "Standalone laboratory frontend" -ForegroundColor Cyan
Write-Host "Python: $python" -ForegroundColor DarkGray
Write-Host "Backend API: $BackendUrl" -ForegroundColor Yellow
Write-Host "Frontend: http://127.0.0.1:$Port/?api=$BackendUrl" -ForegroundColor Green
& $python $server --host 0.0.0.0 --port $Port --api $BackendUrl
