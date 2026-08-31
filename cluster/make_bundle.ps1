param(
    [string]$Output = "C:\codex1\camera_cluster_bundle.zip"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$staging = Join-Path $env:TEMP ("camera-cluster-bundle-" + [Guid]::NewGuid().ToString("N"))
try {
    New-Item -ItemType Directory -Force -Path $staging | Out-Null
    Copy-Item -LiteralPath (Join-Path $root "camera") -Destination $staging -Recurse -Force
    Copy-Item -LiteralPath (Join-Path $root "person_track") -Destination $staging -Recurse -Force
    Copy-Item -LiteralPath (Join-Path $root "cluster") -Destination $staging -Recurse -Force
    $linuxMtx = Join-Path $root "tools\mediamtx\mediamtx_v1.18.2_linux_amd64.tar.gz"
    if (Test-Path -LiteralPath $linuxMtx) {
        New-Item -ItemType Directory -Force -Path (Join-Path $staging "tools\mediamtx") | Out-Null
        Copy-Item -LiteralPath $linuxMtx -Destination (Join-Path $staging "tools\mediamtx") -Force
    }
    Get-ChildItem -LiteralPath $staging -Recurse -Directory -Force |
        Where-Object { $_.Name -in @(".venv", "__pycache__") } |
        Sort-Object FullName -Descending |
        Remove-Item -Recurse -Force
    Remove-Item -LiteralPath $Output -Force -ErrorAction SilentlyContinue
    $paths = @(
        (Join-Path $staging "camera"),
        (Join-Path $staging "person_track"),
        (Join-Path $staging "cluster")
    )
    if (Test-Path -LiteralPath (Join-Path $staging "tools")) { $paths += Join-Path $staging "tools" }
    Compress-Archive -Path $paths -DestinationPath $Output -CompressionLevel Optimal
    Write-Host "Bundle: $Output" -ForegroundColor Green
    Write-Host "No camera passwords are included."
}
finally {
    Remove-Item -LiteralPath $staging -Recurse -Force -ErrorAction SilentlyContinue
}
