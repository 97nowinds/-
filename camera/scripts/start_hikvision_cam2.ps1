param(
    [string]$CameraIp = "192.168.1.64",
    [string]$Username = "admin",
    [ValidateSet("101", "102")]
    [string]$Channel = "102",
    [ValidateSet("tcp", "udp")]
    [string]$RtspTransport = "tcp"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$probe = Join-Path $PSScriptRoot "probe_rtsp.py"
$app = Join-Path $projectRoot "app.py"

Write-Host "Hikvision cam_2 secure launcher" -ForegroundColor Cyan
Write-Host "Camera: $CameraIp, channel: $Channel"
Write-Host "RTSP transport: $RtspTransport (use -RtspTransport udp for the lowest LAN latency)"
Write-Host "The password is used only in this process and is not written to disk."

$securePassword = Read-Host "Enter the password for camera user '$Username'" -AsSecureString
$passwordPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePassword)

try {
    $plainPassword = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($passwordPointer)
    $encodedUsername = [Uri]::EscapeDataString($Username)
    $encodedPassword = [Uri]::EscapeDataString($plainPassword)
    $env:LAB_CAM_2_RTSP = "rtsp://${encodedUsername}:${encodedPassword}@${CameraIp}:554/Streaming/Channels/${Channel}"
    $env:LAB_RTSP_TRANSPORT = $RtspTransport
    $env:OPENCV_FFMPEG_CAPTURE_OPTIONS = "rtsp_transport;$RtspTransport|stimeout;5000000|fflags;nobuffer|flags;low_delay|max_delay;0|analyzeduration;0|probesize;32768"
    $plainPassword = $null

    Set-Location $projectRoot
    Write-Host "Testing RTSP stream..." -ForegroundColor Yellow
    & $python $probe --url-env LAB_CAM_2_RTSP
    if ($LASTEXITCODE -ne 0) {
        throw "RTSP test failed. Check the account, password, stream settings, and camera encoding."
    }

    Write-Host "RTSP is available. Starting http://127.0.0.1:5000/" -ForegroundColor Green
    & $python $app
}
finally {
    if ($passwordPointer -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($passwordPointer)
    }
    Remove-Item Env:LAB_CAM_2_RTSP -ErrorAction SilentlyContinue
    Remove-Item Env:LAB_RTSP_TRANSPORT -ErrorAction SilentlyContinue
    Remove-Item Env:OPENCV_FFMPEG_CAPTURE_OPTIONS -ErrorAction SilentlyContinue
}
