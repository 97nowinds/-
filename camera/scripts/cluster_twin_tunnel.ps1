$ErrorActionPreference = 'Stop'
$configPath = Join-Path $PSScriptRoot '..\config\resident_services.json'
$config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
$identity = [Environment]::ExpandEnvironmentVariables($config.ssh_identity)
if ($identity.StartsWith('~/')) { $identity = Join-Path $env:USERPROFILE $identity.Substring(2) }
$proxy = 'ssh -i "' + $identity + '" -o BatchMode=yes -o ConnectTimeout=8 -W %h:%p ' + $config.ssh_jump
$ssh = Join-Path $env:WINDIR 'System32\OpenSSH\ssh.exe'
while ($true) {
    & $ssh -NT -i $identity -o BatchMode=yes -o ConnectTimeout=8 -o ExitOnForwardFailure=yes `
        -o ServerAliveInterval=15 -o ServerAliveCountMax=2 -o "ProxyCommand=$proxy" `
        -L '127.0.0.1:5182:127.0.0.1:25182' $config.ssh_target
    Start-Sleep -Seconds 5
}
