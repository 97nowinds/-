param(
    [string]$ListenAddress = "0.0.0.0",
    [int]$Port = 6000,
    [string]$Token = $env:LAB_INTERACTION_TOKEN
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvRoot = Join-Path $env:LOCALAPPDATA "camera-interaction-venv"
$python = Join-Path $venvRoot "Scripts\python.exe"
$pluginRoot = Join-Path $projectRoot "plugins\instrument_interaction"
$app = Join-Path $projectRoot "interaction_server.py"

if (-not (Test-Path -LiteralPath $python)) {
    throw "交互推理服务器环境不存在，请先运行 scripts\setup_instrument_interaction.ps1。"
}
if (-not (Test-Path -LiteralPath $app)) {
    throw "交互推理服务器入口不存在：$app"
}

$env:LAB_INTERACTION_PLUGIN_DIR = $pluginRoot
$env:LAB_INTERACTION_SERVER_HOST = $ListenAddress
$env:LAB_INTERACTION_SERVER_PORT = [string]$Port
if ($Token) {
    $env:LAB_INTERACTION_TOKEN = $Token
}

Set-Location $projectRoot
Write-Host "启动远程仪器交互推理服务器：${ListenAddress}:$Port" -ForegroundColor Green
Write-Host "本机应配置 LAB_INTERACTION_SERVER_URL=http://服务器IP:$Port" -ForegroundColor Cyan
& $python $app
