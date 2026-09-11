param(
    [string]$CameraIp = "192.168.31.192",
    [string]$Username = "admin",
    [ValidateSet("tcp", "udp")]
    [string]$RtspTransport = "tcp"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$ffmpeg = Join-Path (Split-Path -Parent $projectRoot) ".venv\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe"
if (-not (Test-Path -LiteralPath $ffmpeg)) {
    $ffmpeg = Join-Path $projectRoot ".venv\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe"
}
if (-not (Test-Path -LiteralPath $ffmpeg)) {
    throw "FFmpeg was not found. Install imageio-ffmpeg in the repository virtual environment first."
}

if (-not (Test-Connection -ComputerName $CameraIp -Count 1 -Quiet)) {
    throw "Camera $CameraIp is not reachable."
}

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

$passwordSecure = Read-SecureStringInWindow "Password for TP-LINK camera user '$Username'"
$passwordPointer = [IntPtr]::Zero

try {
    $passwordPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($passwordSecure)
    $password = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($passwordPointer)
    $user = [Uri]::EscapeDataString($Username)
    $pass = [Uri]::EscapeDataString($password)

    foreach ($stream in @("stream1", "stream2")) {
        $url = "rtsp://${user}:${pass}@${CameraIp}:554/${stream}"
        $arguments = @(
            "-hide_banner", "-loglevel", "error", "-nostdin",
            "-rtsp_transport", $RtspTransport,
            "-fflags", "+discardcorrupt", "-i", $url,
            "-map", "0:v:0", "-frames:v", "1", "-f", "null", "-"
        )
        $null = @(& $ffmpeg @arguments 2>&1)
        if ($LASTEXITCODE -ne 0) {
            throw "TP-LINK RTSP frame check failed for /$stream. Verify the camera password and codec settings."
        }
        Write-Host "tplink_${stream}=ok" -ForegroundColor Green
    }

    Write-Host "TP-LINK RTSP main and sub streams passed." -ForegroundColor Green
}
finally {
    $password = $null
    if ($passwordPointer -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($passwordPointer)
    }
}
