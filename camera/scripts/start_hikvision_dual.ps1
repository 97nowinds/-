param(
    [string]$Camera1Ip = "192.168.1.64",
    [string]$Camera2Ip = "192.168.1.65",
    [string]$Camera1Username = "admin",
    [string]$Camera2Username = "admin",
    [ValidateSet("101", "102")]
    [string]$Channel = "102",
    [ValidateSet("tcp", "udp")]
    [string]$RtspTransport = "tcp",
    [switch]$StandaloneFrontend
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvRoot = Join-Path $projectRoot ".venv"
if (-not (Test-Path -LiteralPath $venvRoot)) {
    $venvRoot = Join-Path (Split-Path -Parent $projectRoot) ".venv"
}
$python = Join-Path $venvRoot "Scripts\python.exe"
$probe = Join-Path $PSScriptRoot "probe_rtsp.py"
$app = Join-Path $projectRoot "app.py"
$nvidiaBinDirs = Get-ChildItem -LiteralPath (Join-Path $venvRoot "Lib\site-packages\nvidia") -Recurse -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -eq "bin" } |
    Select-Object -ExpandProperty FullName
foreach ($nvidiaBinDir in $nvidiaBinDirs) {
    $env:Path = "$nvidiaBinDir;$env:Path"
}
$passwordPointer1 = [IntPtr]::Zero
$passwordPointer2 = [IntPtr]::Zero

Write-Host "Dual Hikvision secure launcher" -ForegroundColor Cyan
Write-Host "Cam1: $Camera1Ip, Cam2: $Camera2Ip, channel: $Channel"
Write-Host "RTSP transport: $RtspTransport (use -RtspTransport udp for the lowest LAN latency)"
Write-Host "Passwords are used only by this process and are not written to disk."

$securePassword1 = Read-Host "Enter password for Cam1 user '$Camera1Username'" -AsSecureString
$securePassword2 = Read-Host "Enter password for Cam2 user '$Camera2Username'" -AsSecureString

try {
    $passwordPointer1 = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePassword1)
    $passwordPointer2 = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($securePassword2)
    $plainPassword1 = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($passwordPointer1)
    $plainPassword2 = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($passwordPointer2)
    $user1 = [Uri]::EscapeDataString($Camera1Username)
    $user2 = [Uri]::EscapeDataString($Camera2Username)
    $pass1 = [Uri]::EscapeDataString($plainPassword1)
    $pass2 = [Uri]::EscapeDataString($plainPassword2)
    $env:LAB_CAM_1_RTSP = "rtsp://${user1}:${pass1}@${Camera1Ip}:554/Streaming/Channels/${Channel}"
    $env:LAB_CAM_2_RTSP = "rtsp://${user2}:${pass2}@${Camera2Ip}:554/Streaming/Channels/${Channel}"
    $env:LAB_RTSP_TRANSPORT = $RtspTransport
    $env:OPENCV_FFMPEG_CAPTURE_OPTIONS = "rtsp_transport;$RtspTransport|stimeout;5000000|fflags;nobuffer|flags;low_delay|max_delay;0|analyzeduration;0|probesize;32768"
    $plainPassword1 = $null
    $plainPassword2 = $null

    Set-Location $projectRoot
    Write-Host "Testing Cam1 RTSP..." -ForegroundColor Yellow
    & $python $probe --url-env LAB_CAM_1_RTSP
    if ($LASTEXITCODE -ne 0) {
        throw "Cam1 RTSP test failed."
    }
    Write-Host "Testing Cam2 RTSP..." -ForegroundColor Yellow
    & $python $probe --url-env LAB_CAM_2_RTSP
    if ($LASTEXITCODE -ne 0) {
        throw "Cam2 RTSP test failed."
    }

    Write-Host "Both streams are available. Starting backend API http://127.0.0.1:5000/" -ForegroundColor Green
    if ($StandaloneFrontend) {
        $env:LAB_API_ONLY = "1"
        $frontendScript = Join-Path $PSScriptRoot "start_frontend.ps1"
        Start-Process powershell.exe -ArgumentList '-NoExit','-ExecutionPolicy','Bypass','-File',$frontendScript,'-BackendUrl','http://127.0.0.1:5000' -WorkingDirectory $projectRoot
        Write-Host "Standalone frontend: http://127.0.0.1:5173/?api=http://127.0.0.1:5000" -ForegroundColor Green
    }
    & $python $app
}
finally {
    if ($passwordPointer1 -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($passwordPointer1)
    }
    if ($passwordPointer2 -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($passwordPointer2)
    }
    Remove-Item Env:LAB_CAM_1_RTSP -ErrorAction SilentlyContinue
    Remove-Item Env:LAB_CAM_2_RTSP -ErrorAction SilentlyContinue
    Remove-Item Env:LAB_API_ONLY -ErrorAction SilentlyContinue
    Remove-Item Env:LAB_RTSP_TRANSPORT -ErrorAction SilentlyContinue
    Remove-Item Env:OPENCV_FFMPEG_CAPTURE_OPTIONS -ErrorAction SilentlyContinue
}
