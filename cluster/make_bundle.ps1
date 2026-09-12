param(
    [string]$Output = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Output)) {
    $Output = Join-Path $root "camera_cluster_bundle.tar.gz"
}
$staging = Join-Path $env:TEMP ("camera-cluster-bundle-" + [Guid]::NewGuid().ToString("N"))
try {
    New-Item -ItemType Directory -Force -Path $staging | Out-Null
    $cameraSource = Join-Path $root "camera"
    $cameraDestination = Join-Path $staging "camera"
    # Runtime data may contain recordings, registered faces or association
    # history. Deploy code and model weights, never operational biometric data.
    $excludedTopLevel = @(".venv", "__pycache__", "runtime", "data", "plugins")
    New-Item -ItemType Directory -Force -Path $cameraDestination | Out-Null
    Get-ChildItem -LiteralPath $cameraSource -Force |
        Where-Object { $_.Name -notin $excludedTopLevel } |
        Copy-Item -Destination $cameraDestination -Recurse -Force
    Copy-Item -LiteralPath (Join-Path $root "person_track") -Destination $staging -Recurse -Force
    Copy-Item -LiteralPath (Join-Path $root "cluster") -Destination $staging -Recurse -Force
    $linuxMtx = Join-Path $root "tools\mediamtx\mediamtx_v1.18.2_linux_amd64.tar.gz"
    if (Test-Path -LiteralPath $linuxMtx) {
        New-Item -ItemType Directory -Force -Path (Join-Path $staging "tools\mediamtx") | Out-Null
        Copy-Item -LiteralPath $linuxMtx -Destination (Join-Path $staging "tools\mediamtx") -Force
    }
    Get-ChildItem -LiteralPath $staging -Recurse -Directory -Force |
        Where-Object { $_.Name -in @(".venv", "__pycache__", "runtime", "cache", "outputs", "known_faces", "recordings") } |
        Sort-Object FullName -Descending |
        Remove-Item -Recurse -Force
    Get-ChildItem -LiteralPath $staging -Recurse -File -Force |
        Where-Object { $_.Extension -in @(".log", ".jsonl", ".bak", ".tmp") } |
        Remove-Item -Force
    Remove-Item -LiteralPath $Output -Force -ErrorAction SilentlyContinue
    $paths = @(
        (Join-Path $staging "camera"),
        (Join-Path $staging "person_track"),
        (Join-Path $staging "cluster")
    )
    if (Test-Path -LiteralPath (Join-Path $staging "tools")) { $paths += Join-Path $staging "tools" }
    if ($Output.EndsWith(".tar.gz", [StringComparison]::OrdinalIgnoreCase)) {
        $entries = @("camera", "person_track", "cluster")
        if (Test-Path -LiteralPath (Join-Path $staging "tools")) { $entries += "tools" }
        Push-Location $staging
        try {
            & tar.exe -czf $Output @entries
            if ($LASTEXITCODE -ne 0) {
                throw "tar.exe failed with exit code $LASTEXITCODE"
            }
        }
        finally {
            Pop-Location
        }
    }
    else {
        Compress-Archive -Path $paths -DestinationPath $Output -CompressionLevel Optimal
    }
    Write-Host "Bundle: $Output" -ForegroundColor Green
    Write-Host "No camera passwords are included."
}
finally {
    Remove-Item -LiteralPath $staging -Recurse -Force -ErrorAction SilentlyContinue
}
