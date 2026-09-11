param(
    [string]$CameraIp = "192.168.31.190",
    [string]$Username = "admin",
    [ValidateSet("101", "102")]
    [string]$Channel = "102",
    [ValidateSet("tcp", "udp")]
    [string]$RtspTransport = "tcp",
    [switch]$DisableInstrumentInteraction,
    [string]$InteractionServerUrl = $env:LAB_INTERACTION_SERVER_URL
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$InteractionServerUrl = if ($InteractionServerUrl) { $InteractionServerUrl.TrimEnd('/') } else { "" }
if (-not $DisableInstrumentInteraction -and [string]::IsNullOrWhiteSpace($InteractionServerUrl)) {
    throw "未配置远程交互推理服务器。请传入 -InteractionServerUrl http://服务器IP:6000，或使用 -DisableInstrumentInteraction。"
}
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
$probe = Join-Path $PSScriptRoot "probe_rtsp.py"
$app = Join-Path $projectRoot "app.py"

function Read-SecureStringInWindow([string]$Prompt) {
    $tempFile = [System.IO.Path]::GetTempFileName()
    try {
        $escapedPrompt = $Prompt.Replace("'", "''")
        $command = @"
`$secure = Read-Host -Prompt '$escapedPrompt' -AsSecureString
if (-not `$secure) { exit 1 }
`$secure | ConvertFrom-SecureString | Set-Content -LiteralPath '$tempFile' -NoNewline
"@
        $process = Start-Process powershell.exe -ArgumentList @(
            '-NoLogo',
            '-NoProfile',
            '-ExecutionPolicy',
            'Bypass',
            '-Command',
            $command
        ) -WindowStyle Normal -PassThru
        Wait-Process -Id $process.Id
        if (-not (Test-Path -LiteralPath $tempFile)) {
            throw "Password prompt window was closed before a password was submitted."
        }
        $cipher = Get-Content -LiteralPath $tempFile -Raw
        if ([string]::IsNullOrWhiteSpace($cipher)) {
            throw "Password prompt returned an empty value."
        }
        return ConvertTo-SecureString $cipher
    }
    finally {
        Remove-Item $tempFile -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "Hikvision cam_2 secure launcher" -ForegroundColor Cyan
Write-Host "Python: $python" -ForegroundColor DarkGray
Write-Host "Camera: $CameraIp, channel: $Channel"
Write-Host "RTSP transport: $RtspTransport (use -RtspTransport udp for the lowest LAN latency)"
Write-Host "The password is used only in this process and is not written to disk."

$securePassword = Read-SecureStringInWindow "Enter the password for camera user '$Username'"
$passwordPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePassword)

try {
    $plainPassword = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($passwordPointer)
    $encodedUsername = [Uri]::EscapeDataString($Username)
    $encodedPassword = [Uri]::EscapeDataString($plainPassword)
    $env:LAB_CAM_2_RTSP = "rtsp://${encodedUsername}:${encodedPassword}@${CameraIp}:554/Streaming/Channels/${Channel}"
    $env:LAB_RTSP_TRANSPORT = $RtspTransport
    if ($DisableInstrumentInteraction) {
        $env:LAB_INTERACTION_ENABLED = "0"
    } else {
        $env:LAB_INTERACTION_ENABLED = "1"
        $env:LAB_INTERACTION_SERVER_URL = $InteractionServerUrl
    }
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
    Remove-Item Env:LAB_INTERACTION_ENABLED,Env:LAB_INTERACTION_SERVER_URL -ErrorAction SilentlyContinue
}
