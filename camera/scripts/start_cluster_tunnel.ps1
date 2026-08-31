param(
    [string]$JumpHost = "xqyi@10.137.145.22",
    [string]$RemoteHost = "10.10.0.10",
    [int]$LocalPort = 18554,
    [int]$RemotePort = 8554,
    [int]$ApiLocalPort = 5000,
    [int]$ApiRemotePort = 5000,
    [int]$RestartSeconds = 5,
    [string]$IdentityFile = "$env:USERPROFILE\.ssh\lab_rtsp_ed25519"
)

$ErrorActionPreference = "Continue"
$ssh = Join-Path $env:SystemRoot "System32\OpenSSH\ssh.exe"
if (-not (Test-Path -LiteralPath $ssh)) {
    throw "OpenSSH client was not found at $ssh"
}
if (-not (Test-Path -LiteralPath $IdentityFile)) {
    throw "SSH identity file was not found: $IdentityFile"
}

while ($true) {
    $forward = "${LocalPort}:${RemoteHost}:${RemotePort}"
    $arguments = @(
        "-N",
        "-i", $IdentityFile,
        "-o", "IdentitiesOnly=yes",
        "-o", "BatchMode=yes",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=3",
        "-L", $forward,
        "-L", "${ApiLocalPort}:${RemoteHost}:${ApiRemotePort}",
        $JumpHost
    )

    Write-Host "Opening RTSP tunnel to $JumpHost with the local SSH key." -ForegroundColor Cyan
    & $ssh @arguments
    Write-Host "Tunnel disconnected or authentication failed; retrying in ${RestartSeconds}s." -ForegroundColor Yellow
    Start-Sleep -Seconds $RestartSeconds
}
