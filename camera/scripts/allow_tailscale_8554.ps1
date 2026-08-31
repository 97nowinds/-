#requires -RunAsAdministrator

$ErrorActionPreference = "Stop"
$ruleName = "HAFS RTSP Relay via Tailscale"
$existing = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue
if ($existing) {
    Set-NetFirewallRule -DisplayName $ruleName -Enabled True -Direction Inbound -Action Allow -Profile Any
} else {
    New-NetFirewallRule `
        -DisplayName $ruleName `
        -Direction Inbound `
        -Action Allow `
        -Protocol TCP `
        -LocalPort 8554 `
        -RemoteAddress "100.64.0.0/10" `
        -InterfaceAlias "Tailscale" `
        -Profile Any | Out-Null
}
Write-Host "RTSP port 8554 is allowed from Tailscale CGNAT addresses only." -ForegroundColor Green
Get-NetFirewallRule -DisplayName $ruleName | Get-NetFirewallPortFilter | Select-Object Protocol, LocalPort
