$ErrorActionPreference = "Stop"

$key = "$env:USERPROFILE\.ssh\lab_rtsp_ed25519"
$publicKeyFile = "$key.pub"
$ssh = Join-Path $env:SystemRoot "System32\OpenSSH\ssh.exe"
$jumpHost = "xqyi@10.137.145.22"

if (-not (Test-Path -LiteralPath $publicKeyFile)) {
    throw "Public key was not found: $publicKeyFile"
}
if (-not (Test-Path -LiteralPath $ssh)) {
    throw "OpenSSH client was not found at $ssh"
}

$publicKey = (Get-Content -LiteralPath $publicKeyFile -Raw).Trim()
if ($publicKey -notmatch '^ssh-ed25519 [A-Za-z0-9+/=]+(?: .*)?$') {
    throw "The public key format is invalid."
}

$remoteCommand = "umask 077; mkdir -p ~/.ssh; touch ~/.ssh/authorized_keys; grep -qxF '$publicKey' ~/.ssh/authorized_keys || printf '%s\n' '$publicKey' >> ~/.ssh/authorized_keys; chmod 700 ~/.ssh; chmod 600 ~/.ssh/authorized_keys"
Write-Host "Adding the lab host public key to $jumpHost ~/.ssh/authorized_keys." -ForegroundColor Cyan
Write-Host "Enter the SSH password once when prompted. The password is not saved." -ForegroundColor Yellow
& $ssh -o StrictHostKeyChecking=accept-new $jumpHost $remoteCommand
if ($LASTEXITCODE -ne 0) {
    throw "The public key was not installed. SSH exit code: $LASTEXITCODE"
}
Write-Host "Public key installation completed." -ForegroundColor Green
