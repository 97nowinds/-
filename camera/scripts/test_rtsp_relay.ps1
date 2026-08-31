param(
    [int]$RelayPort = 8554
)

$ErrorActionPreference = "Stop"
$ffmpeg = Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) ".venv\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe"
if (-not (Test-Path -LiteralPath $ffmpeg)) { throw "FFmpeg was not found at $ffmpeg" }
$listener = Get-NetTCPConnection -LocalPort $RelayPort -State Listen -ErrorAction SilentlyContinue
if (-not $listener) {
    throw "MediaMTX is not listening on TCP $RelayPort. Start start_rtsp_gateway.ps1 first."
}

foreach ($camera in @("cam_1", "cam_2")) {
    $url = "rtsp://127.0.0.1:${RelayPort}/${camera}"
    $arguments = @(
        "-hide_banner", "-loglevel", "warning", "-nostdin",
        "-rtsp_transport", "tcp",
        "-analyzeduration", "10000000", "-probesize", "10000000",
        "-fflags", "+discardcorrupt", "-skip_frame", "nokey", "-i", $url,
        "-map", "0:v:0", "-frames:v", "1", "-f", "null", "-"
    )
    $output = @(& $ffmpeg @arguments 2>&1)
    if ($LASTEXITCODE -ne 0) { throw "Local relay frame check failed for /$camera" }
    Write-Host "local_${camera}=ok" -ForegroundColor Green
}

$remote = Test-NetConnection 100.126.39.4 -Port $RelayPort -InformationLevel Quiet
Write-Host "tailscale_tcp_${RelayPort}=$remote"
if (-not $remote) { throw "Tailscale TCP port $RelayPort is not reachable locally." }
