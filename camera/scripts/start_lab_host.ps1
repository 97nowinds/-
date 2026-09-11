param(
    [string]$Camera1Ip = "192.168.31.191",
    [string]$Camera2Ip = "192.168.31.190",
    [string]$EntranceCameraIp = "192.168.31.192",
    [string]$Camera1Username = "admin",
    [string]$Camera2Username = "admin",
    [string]$EntranceCameraUsername = "admin",
    [string]$EntranceStream = "stream1",
    [ValidateSet("101", "102")]
    [string]$Channel = "102",
    [ValidateSet("tcp", "udp")]
    [string]$RtspTransport = "udp",
    [string]$ListenAddress = "0.0.0.0",
    [int]$Port = 5000,
    [string]$FrontendOrigin = "*",
    [switch]$PromptInConsole,
    [switch]$DisableInstrumentInteraction,
    [string]$InteractionServerUrl = $env:LAB_INTERACTION_SERVER_URL
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvRoot = Join-Path $projectRoot ".venv"
if (-not (Test-Path -LiteralPath $venvRoot)) {
    $venvRoot = Join-Path (Split-Path -Parent $projectRoot) ".venv"
}
if (-not $DisableInstrumentInteraction -and [string]::IsNullOrWhiteSpace($InteractionServerUrl)) {
    throw "Remote interaction server is not configured. Pass -InteractionServerUrl http://SERVER_IP:6000 or use -DisableInstrumentInteraction."
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
$arcfaceRoot = Join-Path $projectRoot "models\insightface\models\buffalo_l"
$requiredModels = @("det_10g.onnx", "w600k_r50.onnx")
$passwordPointers = @()

function Read-SecureStringInWindow([string]$Prompt) {
    if ($PromptInConsole) {
        $secure = Read-Host -Prompt $Prompt -AsSecureString
        if (-not $secure) {
            throw "Password prompt returned an empty value."
        }
        return $secure
    }

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

foreach ($model in $requiredModels) {
    if (-not (Test-Path (Join-Path $arcfaceRoot $model))) {
        throw "ArcFace model '$model' is missing. Run scripts\install_arcface_models.py first."
    }
}

Write-Host "Laboratory compute host: ArcFace tracking" -ForegroundColor Cyan
if ($Camera1Ip) {
    Write-Host "Indoor camera 1: $Camera1Ip"
} else {
    Write-Host "Indoor camera 1: skipped" -ForegroundColor Yellow
}
Write-Host "Indoor camera 2: $Camera2Ip"
if ($EntranceCameraIp) {
    Write-Host "Entrance ArcFace camera: $EntranceCameraIp"
} else {
    Write-Host "Entrance ArcFace camera: not enabled; using two indoor cameras" -ForegroundColor Yellow
}
Write-Host "API: http://${ListenAddress}:$Port, RTSP transport: $RtspTransport"
Write-Host ("Instrument interaction: " + $(if ($DisableInstrumentInteraction) { "disabled" } else { "remote -> $InteractionServerUrl, cam_2 -> lab_camera_view_2" }))
Write-Host "Passwords stay in this process and are never written to disk."

$password1 = $null
$camera1Enabled = -not [string]::IsNullOrWhiteSpace($Camera1Ip)
if ($camera1Enabled) {
    $password1 = Read-SecureStringInWindow "Password for Cam1 user '$Camera1Username'"
}
$password2 = Read-SecureStringInWindow "Password for Cam2 user '$Camera2Username'"
$entrancePassword = $null
if ($EntranceCameraIp) {
    $entrancePassword = Read-SecureStringInWindow "Password for entrance camera user '$EntranceCameraUsername'"
}

try {
    if ($camera1Enabled) {
        $passwordPointers += [Runtime.InteropServices.Marshal]::SecureStringToBSTR($password1)
    }
    $passwordPointers += [Runtime.InteropServices.Marshal]::SecureStringToBSTR($password2)
    if ($EntranceCameraIp) {
        $passwordPointers += [Runtime.InteropServices.Marshal]::SecureStringToBSTR($entrancePassword)
    }
    $user2 = [Uri]::EscapeDataString($Camera2Username)
    $passwordIndex = 0
    if ($camera1Enabled) {
        $user1 = [Uri]::EscapeDataString($Camera1Username)
        $pass1 = [Uri]::EscapeDataString([Runtime.InteropServices.Marshal]::PtrToStringBSTR($passwordPointers[$passwordIndex]))
        $env:LAB_CAM_1_RTSP = "rtsp://${user1}:${pass1}@${Camera1Ip}:554/Streaming/Channels/${Channel}"
        $passwordIndex++
    }
    $pass2 = [Uri]::EscapeDataString([Runtime.InteropServices.Marshal]::PtrToStringBSTR($passwordPointers[$passwordIndex]))
    $env:LAB_CAM_2_RTSP = "rtsp://${user2}:${pass2}@${Camera2Ip}:554/Streaming/Channels/${Channel}"
    if ($EntranceCameraIp) {
        $entranceUser = [Uri]::EscapeDataString($EntranceCameraUsername)
        $entrancePass = [Uri]::EscapeDataString([Runtime.InteropServices.Marshal]::PtrToStringBSTR($passwordPointers[$passwordPointers.Count - 1]))
        $env:LAB_CAM_ENTRANCE_RTSP = "rtsp://${entranceUser}:${entrancePass}@${EntranceCameraIp}:554/${EntranceStream}"
        $entrancePass = $null
    }
    $pass1 = $null
    $pass2 = $null
    $env:LAB_RTSP_TRANSPORT = $RtspTransport
    $env:OPENCV_FFMPEG_CAPTURE_OPTIONS = "rtsp_transport;$RtspTransport|stimeout;5000000|fflags;nobuffer|flags;low_delay|max_delay;0|analyzeduration;0|probesize;32768"
    $env:LAB_FACE_ENGINE = "arcface"
    $env:LAB_API_ONLY = "1"
    $env:LAB_APP_HOST = $ListenAddress
    $env:LAB_APP_PORT = [string]$Port
    $env:LAB_FRONTEND_ORIGIN = $FrontendOrigin
    if ($DisableInstrumentInteraction) {
        $env:LAB_INTERACTION_ENABLED = "0"
        Remove-Item Env:LAB_INTERACTION_SERVER_URL -ErrorAction SilentlyContinue
    }
    else {
        $env:LAB_INTERACTION_ENABLED = "1"
        $env:LAB_INTERACTION_SERVER_URL = $InteractionServerUrl.TrimEnd('/')
    }

    Set-Location $projectRoot
    $cameraEnvironments = @()
    if ($camera1Enabled) {
        $cameraEnvironments += "LAB_CAM_1_RTSP"
    }
    $cameraEnvironments += "LAB_CAM_2_RTSP"
    if ($EntranceCameraIp) {
        $cameraEnvironments += "LAB_CAM_ENTRANCE_RTSP"
    }
    foreach ($cameraEnvironment in $cameraEnvironments) {
        Write-Host "Testing $cameraEnvironment..." -ForegroundColor Yellow
        & $python $probe --url-env $cameraEnvironment
        if ($LASTEXITCODE -ne 0) {
            throw "$cameraEnvironment RTSP test failed."
        }
    }
    & $python -c "import onnxruntime as ort; print('ONNX providers:', ort.get_available_providers())"
    Write-Host "Starting laboratory compute API on port $Port" -ForegroundColor Green
    & $python $app
}
finally {
    foreach ($pointer in $passwordPointers) {
        if ($pointer -ne [IntPtr]::Zero) {
            [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
        }
    }
    foreach ($name in @(
        "LAB_CAM_1_RTSP", "LAB_CAM_2_RTSP", "LAB_CAM_ENTRANCE_RTSP",
        "LAB_RTSP_TRANSPORT", "OPENCV_FFMPEG_CAPTURE_OPTIONS", "LAB_FACE_ENGINE",
        "LAB_API_ONLY", "LAB_APP_HOST", "LAB_APP_PORT", "LAB_FRONTEND_ORIGIN",
        "LAB_INTERACTION_ENABLED", "LAB_INTERACTION_SERVER_URL"
    )) {
        Remove-Item "Env:$name" -ErrorAction SilentlyContinue
    }
}
