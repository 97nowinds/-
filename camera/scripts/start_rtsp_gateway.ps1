param(
    [string]$Camera1Ip = "192.168.1.64",
    [string]$Camera2Ip = "192.168.1.65",
    [string]$Camera1Username = "admin",
    [string]$Camera2Username = "admin",
    [ValidateSet("101", "102")]
    [string]$Channel = "102",
    [ValidateSet("tcp", "udp")]
    [string]$RtspTransport = "tcp",
    [int]$RelayPort = 8554
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$installCandidates = @(
    (Join-Path "C:\codex1" "tools\mediamtx"),
    (Join-Path (Split-Path -Parent $projectRoot) "tools\mediamtx"),
    (Join-Path $projectRoot "tools\mediamtx")
)
$installRoot = $installCandidates[0]
$mediaMtx = $null
$configPath = $null
foreach ($candidateRoot in $installCandidates) {
    $candidateExe = Join-Path $candidateRoot "mediamtx.exe"
    if (Test-Path -LiteralPath $candidateExe) {
        $installRoot = $candidateRoot
        $mediaMtx = $candidateExe
        $configPath = Join-Path $candidateRoot "mediamtx.yml"
        break
    }
}
if (-not $mediaMtx) {
    $mediaMtx = Join-Path $installRoot "mediamtx.exe"
    $configPath = Join-Path $installRoot "mediamtx.yml"
}
function Find-FfmpegPath {
    $candidates = @(
        (Join-Path "C:\codex1" ".venv\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe"),
        (Join-Path $projectRoot ".venv\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe"),
        (Join-Path (Split-Path -Parent $projectRoot) ".venv\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) { return $candidate }
    }
    throw "FFmpeg was not found. Install imageio-ffmpeg in C:\codex1\.venv or camera\.venv first."
}
$ffmpeg = Find-FfmpegPath
if (-not (Test-Path -LiteralPath $mediaMtx)) {
    & (Join-Path $PSScriptRoot "install_mediamtx.ps1") -InstallDir $installRoot
    if (-not (Test-Path -LiteralPath $mediaMtx)) { throw "MediaMTX installation failed." }
}

$passwordPointers = @()
$mediaMtxProcess = $null
$relayProcesses = @{}

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

function New-HikvisionUrl([string]$ip, [string]$username, [string]$password) {
    $escapedUser = [Uri]::EscapeDataString($username)
    $escapedPassword = [Uri]::EscapeDataString($password)
    return "rtsp://${escapedUser}:${escapedPassword}@${ip}:554/Streaming/Channels/${Channel}"
}

function Start-Relay([string]$name, [string]$sourceUrl, [string]$path) {
    $publishUrl = "rtsp://127.0.0.1:${RelayPort}/${path}"
    $argumentLine = "-hide_banner -loglevel error -nostdin -rtsp_transport $RtspTransport -fflags nobuffer -flags low_delay -i `"$sourceUrl`" -map 0:v:0 -map 0:a? -c:v copy -c:a copy -f rtsp `"$publishUrl`""
    # Do not let FFmpeg print a credential-bearing input URL to the console.
    $process = Start-Process -FilePath $ffmpeg -ArgumentList $argumentLine -WindowStyle Hidden -PassThru
    $relayProcesses[$name] = $process
    Write-Host "$name relay started on /$path (PID $($process.Id))" -ForegroundColor Green
}

Write-Host "Laboratory RTSP relay" -ForegroundColor Cyan
Write-Host "Cam1: $Camera1Ip, Cam2: $Camera2Ip, channel: $Channel, transport: $RtspTransport"
Write-Host "Published streams: rtsp://100.126.39.4:${RelayPort}/cam_1 and /cam_2"
Write-Host "The lab host runs MediaMTX and FFmpeg only. No Python or AI process is started."
Write-Host "Passwords are requested at runtime and are not written to disk or logs."

$password1 = Read-SecureStringInWindow "Password for Cam1 user '$Camera1Username'"
$password2 = Read-SecureStringInWindow "Password for Cam2 user '$Camera2Username'"

try {
    $passwordPointers += [Runtime.InteropServices.Marshal]::SecureStringToBSTR($password1)
    $passwordPointers += [Runtime.InteropServices.Marshal]::SecureStringToBSTR($password2)
    $source1 = New-HikvisionUrl $Camera1Ip $Camera1Username ([Runtime.InteropServices.Marshal]::PtrToStringBSTR($passwordPointers[0]))
    $source2 = New-HikvisionUrl $Camera2Ip $Camera2Username ([Runtime.InteropServices.Marshal]::PtrToStringBSTR($passwordPointers[1]))
    $password1 = $null
    $password2 = $null

    $mediaMtxProcess = Start-Process -FilePath $mediaMtx -ArgumentList "`"$configPath`"" -WorkingDirectory $installRoot -PassThru
    Start-Sleep -Seconds 2

    Start-Relay "Cam1" $source1 "cam_1"
    Start-Relay "Cam2" $source2 "cam_2"
    $source1 = $null
    $source2 = $null

    while ($true) {
        foreach ($name in @("Cam1", "Cam2")) {
            $process = $relayProcesses[$name]
            if ($process.HasExited) {
                Write-Host "$name relay exited with code $($process.ExitCode); restarting automatically." -ForegroundColor Yellow
                if ($name -eq "Cam1") { Start-Relay $name (New-HikvisionUrl $Camera1Ip $Camera1Username ([Runtime.InteropServices.Marshal]::PtrToStringBSTR($passwordPointers[0]))) "cam_1" }
                if ($name -eq "Cam2") { Start-Relay $name (New-HikvisionUrl $Camera2Ip $Camera2Username ([Runtime.InteropServices.Marshal]::PtrToStringBSTR($passwordPointers[1]))) "cam_2" }
            }
        }
        Start-Sleep -Seconds 5
    }
}
finally {
    foreach ($process in $relayProcesses.Values) {
        if ($process -and -not $process.HasExited) { Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue }
    }
    if ($mediaMtxProcess -and -not $mediaMtxProcess.HasExited) { Stop-Process -Id $mediaMtxProcess.Id -Force -ErrorAction SilentlyContinue }
    foreach ($pointer in $passwordPointers) {
        if ($pointer -ne [IntPtr]::Zero) { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer) }
    }
}
