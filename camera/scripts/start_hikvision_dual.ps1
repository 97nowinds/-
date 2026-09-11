param(
    [string]$Camera1Ip = "192.168.31.191",
    [string]$Camera2Ip = "192.168.31.190",
    [string]$Camera1Username = "admin",
    [string]$Camera2Username = "admin",
    [ValidateSet("101", "102")]
    [string]$Channel = "102",
    [ValidateSet("tcp", "udp")]
    [string]$RtspTransport = "tcp",
    [switch]$StandaloneFrontend,
    [switch]$DisableInstrumentInteraction,
    [string]$InteractionServerUrl = $env:LAB_INTERACTION_SERVER_URL
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$InteractionServerUrl = if ($InteractionServerUrl) { $InteractionServerUrl.TrimEnd('/') } else { "" }
if (-not $DisableInstrumentInteraction -and [string]::IsNullOrWhiteSpace($InteractionServerUrl)) {
    throw "未配置远程交互推理服务器。请传入 -InteractionServerUrl http://服务器IP:6000，或使用 -DisableInstrumentInteraction。"
}
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

Write-Host "Dual Hikvision secure launcher" -ForegroundColor Cyan
Write-Host "Cam1: $Camera1Ip, Cam2: $Camera2Ip, channel: $Channel"
Write-Host "RTSP transport: $RtspTransport (use -RtspTransport udp for the lowest LAN latency)"
Write-Host "Passwords are used only by this process and are not written to disk."

$securePassword1 = Read-SecureStringInWindow "Enter password for Cam1 user '$Camera1Username'"
$securePassword2 = Read-SecureStringInWindow "Enter password for Cam2 user '$Camera2Username'"

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
    if ($DisableInstrumentInteraction) {
        $env:LAB_INTERACTION_ENABLED = "0"
    } else {
        $env:LAB_INTERACTION_ENABLED = "1"
        $env:LAB_INTERACTION_SERVER_URL = $InteractionServerUrl
    }
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
    Remove-Item Env:LAB_INTERACTION_ENABLED,Env:LAB_INTERACTION_SERVER_URL -ErrorAction SilentlyContinue
}
