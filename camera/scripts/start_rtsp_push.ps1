param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^rtsps?://')]
    [string]$RemoteBaseUrl,
    [string]$Camera1Ip = "192.168.31.191",
    [string]$Camera2Ip = "192.168.31.190",
    [string]$EntranceCameraIp = "192.168.31.192",
    [string]$Camera1Username = "admin",
    [string]$Camera2Username = "admin",
    [string]$EntranceCameraUsername = "admin",
    [string]$RemoteUsername = "",
    [ValidateSet("101", "102")]
    [string]$Channel = "102",
    [ValidateSet("tcp", "udp")]
    [string]$RtspTransport = "tcp",
    [int]$RestartSeconds = 5,
    [int]$PublisherConnectGraceSeconds = 15,
    [int]$PublisherMissLimit = 3,
    [switch]$PromptInConsole
)

$ErrorActionPreference = "Stop"
if ($RemoteBaseUrl -match '@') {
    throw "Do not put a remote password in RemoteBaseUrl. Use -RemoteUsername and the runtime prompt."
}
$projectRoot = Split-Path -Parent $PSScriptRoot
function Find-FfmpegPath {
    $candidates = @(
        (Join-Path $projectRoot ".venv\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe"),
        (Join-Path (Split-Path -Parent $projectRoot) ".venv\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) { return $candidate }
    }
    throw "FFmpeg was not found. Install imageio-ffmpeg in camera\.venv or the repository virtual environment first."
}
$ffmpeg = Find-FfmpegPath

$passwordPointers = @()
$processes = @{}
$processStartedAt = @{}
$publisherMisses = @{}
$remotePassword = ""
$remotePort = ([Uri]$RemoteBaseUrl).Port

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

function New-HikvisionUrl([string]$ip, [string]$username, [string]$password) {
    $user = [Uri]::EscapeDataString($username)
    $pass = [Uri]::EscapeDataString($password)
    return "rtsp://${user}:${pass}@${ip}:554/Streaming/Channels/${Channel}"
}

function New-TplinkUrl([string]$ip, [string]$username, [string]$password) {
    $user = [Uri]::EscapeDataString($username)
    $pass = [Uri]::EscapeDataString($password)
    return "rtsp://${user}:${pass}@${ip}:554/stream1"
}

function Add-Credentials([string]$url, [string]$username, [string]$password) {
    if ([string]::IsNullOrWhiteSpace($username)) { return $url }
    $uri = [Uri]$url
    $user = [Uri]::EscapeDataString($username)
    $pass = [Uri]::EscapeDataString($password)
    return "$($uri.Scheme)://${user}:${pass}@$($uri.Host):$($uri.Port)$($uri.PathAndQuery)"
}

function Start-Push([string]$name, [string]$sourceUrl, [string]$destinationUrl) {
    $argumentLine = "-hide_banner -loglevel error -nostdin -rtsp_transport $RtspTransport -fflags nobuffer -flags low_delay -i `"$sourceUrl`" -map 0:v:0 -map 0:a? -c:v copy -c:a copy -f rtsp -rtsp_transport tcp `"$destinationUrl`""
    $processes[$name] = Start-Process -FilePath $ffmpeg -ArgumentList $argumentLine -WindowStyle Hidden -PassThru
    $processStartedAt[$name] = [DateTime]::UtcNow
    $publisherMisses[$name] = 0
    Write-Host "$name push process started (PID $($processes[$name].Id))" -ForegroundColor Green
}

function Test-PublisherConnection([Diagnostics.Process]$process) {
    if (-not $process -or $process.HasExited) { return $false }
    $connections = Get-NetTCPConnection -OwningProcess $process.Id -State Established -ErrorAction SilentlyContinue
    return [bool]($connections | Where-Object { $_.RemotePort -eq $remotePort } | Select-Object -First 1)
}

$password1 = Read-SecureStringInWindow "Password for Cam1 user '$Camera1Username'"
$password2 = Read-SecureStringInWindow "Password for Cam2 user '$Camera2Username'"
$entrancePassword = $null
if (-not [string]::IsNullOrWhiteSpace($EntranceCameraIp)) {
    $entrancePassword = Read-SecureStringInWindow "Password for TP-LINK entrance camera user '$EntranceCameraUsername'"
}
if (-not [string]::IsNullOrWhiteSpace($RemoteUsername)) {
    $remotePasswordSecure = Read-SecureStringInWindow "Password for remote RTSP ingest user '$RemoteUsername'"
}

try {
    $passwordPointers += [Runtime.InteropServices.Marshal]::SecureStringToBSTR($password1)
    $passwordPointers += [Runtime.InteropServices.Marshal]::SecureStringToBSTR($password2)
    $entrancePointerIndex = -1
    if ($entrancePassword) {
        $entrancePointerIndex = $passwordPointers.Count
        $passwordPointers += [Runtime.InteropServices.Marshal]::SecureStringToBSTR($entrancePassword)
    }
    if ($remotePasswordSecure) {
        $remotePointerIndex = $passwordPointers.Count
        $passwordPointers += [Runtime.InteropServices.Marshal]::SecureStringToBSTR($remotePasswordSecure)
        $remotePassword = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($passwordPointers[$remotePointerIndex])
    }
    $remoteCam1 = Add-Credentials (($RemoteBaseUrl.TrimEnd('/') + "/cam_1")) $RemoteUsername $remotePassword
    $remoteCam2 = Add-Credentials (($RemoteBaseUrl.TrimEnd('/') + "/cam_2")) $RemoteUsername $remotePassword
    $source1 = New-HikvisionUrl $Camera1Ip $Camera1Username ([Runtime.InteropServices.Marshal]::PtrToStringBSTR($passwordPointers[0]))
    $source2 = New-HikvisionUrl $Camera2Ip $Camera2Username ([Runtime.InteropServices.Marshal]::PtrToStringBSTR($passwordPointers[1]))
    if ($entrancePointerIndex -ge 0) {
        $remoteEntrance = Add-Credentials (($RemoteBaseUrl.TrimEnd('/') + "/cam_entrance")) $RemoteUsername $remotePassword
        $sourceEntrance = New-TplinkUrl $EntranceCameraIp $EntranceCameraUsername ([Runtime.InteropServices.Marshal]::PtrToStringBSTR($passwordPointers[$entrancePointerIndex]))
    }

    Write-Host "Laboratory RTSP push gateway" -ForegroundColor Cyan
    Write-Host "Cam1: $Camera1Ip, Cam2: $Camera2Ip, entrance: $EntranceCameraIp, input transport: $RtspTransport"
    Write-Host "Remote paths: /cam_1, /cam_2, and /cam_entrance"
    Write-Host "The lab host runs FFmpeg only; no Python, AI, recording, or Flask."
    Write-Host "All passwords are runtime-only and are not written to disk or logs."

    Start-Push "Cam1" $source1 $remoteCam1
    Start-Push "Cam2" $source2 $remoteCam2
    if ($entrancePointerIndex -ge 0) {
        Start-Push "Entrance" $sourceEntrance $remoteEntrance
    }

    while ($true) {
        foreach ($name in @($processes.Keys)) {
            $process = $processes[$name]
            if ($process.HasExited) {
                Write-Host "$name push exited with code $($process.ExitCode); retrying in ${RestartSeconds}s." -ForegroundColor Yellow
                Start-Sleep -Seconds $RestartSeconds
                if ($name -eq "Cam1") { Start-Push $name $source1 $remoteCam1 }
                if ($name -eq "Cam2") { Start-Push $name $source2 $remoteCam2 }
                if ($name -eq "Entrance") { Start-Push $name $sourceEntrance $remoteEntrance }
            }
            elseif (([DateTime]::UtcNow - $processStartedAt[$name]).TotalSeconds -ge $PublisherConnectGraceSeconds) {
                if (Test-PublisherConnection $process) {
                    $publisherMisses[$name] = 0
                }
                else {
                    $publisherMisses[$name]++
                    if ($publisherMisses[$name] -ge $PublisherMissLimit) {
                        Write-Host "$name push process is alive but no longer publishing; forcing restart." -ForegroundColor Yellow
                        Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
                        Start-Sleep -Seconds $RestartSeconds
                        if ($name -eq "Cam1") { Start-Push $name $source1 $remoteCam1 }
                        if ($name -eq "Cam2") { Start-Push $name $source2 $remoteCam2 }
                        if ($name -eq "Entrance") { Start-Push $name $sourceEntrance $remoteEntrance }
                    }
                }
            }
        }
        Start-Sleep -Seconds 2
    }
}
finally {
    foreach ($process in $processes.Values) {
        if ($process -and -not $process.HasExited) { Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue }
    }
    foreach ($pointer in $passwordPointers) {
        if ($pointer -ne [IntPtr]::Zero) { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer) }
    }
}
