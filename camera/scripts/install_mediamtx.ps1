param(
    [string]$InstallDir = "C:\codex1\tools\mediamtx",
    [string]$Version = "v1.18.2"
)

$ErrorActionPreference = "Stop"
$assetUrl = "https://github.com/bluenviron/mediamtx/releases/download/${Version}/mediamtx_${Version}_windows_amd64.zip"

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
$zipPath = Join-Path $env:TEMP ("mediamtx-" + [Guid]::NewGuid().ToString("N") + ".zip")
$extractDir = Join-Path $env:TEMP ("mediamtx-" + [Guid]::NewGuid().ToString("N"))
try {
    Write-Host "Downloading MediaMTX $Version..."
    Invoke-WebRequest -Uri $assetUrl -OutFile $zipPath
    Expand-Archive -LiteralPath $zipPath -DestinationPath $extractDir -Force
    $binary = Get-ChildItem -LiteralPath $extractDir -Filter "mediamtx.exe" -Recurse | Select-Object -First 1
    if (-not $binary) { throw "MediaMTX executable was not found in the downloaded archive." }
    Copy-Item -LiteralPath $binary.FullName -Destination (Join-Path $InstallDir "mediamtx.exe") -Force
    Write-Host "Installed MediaMTX at $(Join-Path $InstallDir 'mediamtx.exe')" -ForegroundColor Green
}
finally {
    Remove-Item -LiteralPath $zipPath -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $extractDir -Recurse -Force -ErrorAction SilentlyContinue
}
