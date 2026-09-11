param(
    [Parameter(Mandatory = $true)]
    [string]$Python311Exe
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pluginRoot = Join-Path $projectRoot "plugins\instrument_interaction"
# Locked ONNX test assets exceed MAX_PATH under a deeply nested checkout.
$venvRoot = Join-Path $env:LOCALAPPDATA "camera-interaction-venv"
$venvPython = Join-Path $venvRoot "Scripts\python.exe"
$pluginInstaller = Get-ChildItem -LiteralPath $pluginRoot -File -Filter "*.ps1" |
    Select-Object -First 1 -ExpandProperty FullName
$pluginCheck = Get-ChildItem -LiteralPath $pluginRoot -File -Filter "*.py" |
    Select-Object -First 1 -ExpandProperty FullName
$serverRequirements = Join-Path $projectRoot "requirements-interaction-server.txt"

if (-not (Test-Path -LiteralPath $Python311Exe)) {
    throw "Python 3.11 executable was not found: $Python311Exe"
}
if ([string]::IsNullOrWhiteSpace($pluginInstaller) -or -not (Test-Path -LiteralPath $pluginInstaller)) {
    throw "Instrument interaction plugin is incomplete: $pluginInstaller"
}
if ([string]::IsNullOrWhiteSpace($pluginCheck) -or -not (Test-Path -LiteralPath $pluginCheck)) {
    throw "Instrument interaction environment check is missing: $pluginCheck"
}

& $Python311Exe -c "import sys; assert sys.version_info[:2] == (3, 11), sys.version; print(sys.executable, sys.version)"
if ($LASTEXITCODE -ne 0) { throw "The module requires Python 3.11." }

if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Host "Creating Python 3.11 interaction environment: $venvRoot" -ForegroundColor Cyan
    & $Python311Exe -m venv $venvRoot
    if ($LASTEXITCODE -ne 0) { throw "Failed to create the interaction virtual environment." }
}

Write-Host "Installing the module's locked CUDA 12.8 environment..." -ForegroundColor Cyan
& $pluginInstaller -PythonExe $venvPython
if ($LASTEXITCODE -ne 0) { throw "The instrument interaction dependency installation failed." }

Write-Host "Installing interaction server packages..." -ForegroundColor Cyan
& $venvPython -m pip install -r $serverRequirements
if ($LASTEXITCODE -ne 0) { throw "The interaction server dependency installation failed." }
& $venvPython -m pip check
if ($LASTEXITCODE -ne 0) { throw "The combined environment has dependency conflicts." }

Write-Host "Running lightweight module health check..." -ForegroundColor Cyan
& $venvPython $pluginCheck
if ($LASTEXITCODE -ne 0) { throw "The lightweight module health check failed." }

Write-Host "Loading MiniCPM and YOLO for the required GPU health check..." -ForegroundColor Cyan
& $venvPython $pluginCheck --load-models
if ($LASTEXITCODE -ne 0) {
    throw "The model-load health check failed. Check the NVIDIA driver, CUDA compatibility, and available VRAM."
}

Write-Host "Interaction inference server environment is ready. Start with scripts\start_interaction_server.ps1." -ForegroundColor Green
