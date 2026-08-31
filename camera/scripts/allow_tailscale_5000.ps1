#requires -RunAsAdministrator

$ErrorActionPreference = "Stop"
$ruleName = "HAFS Dashboard via Tailscale"
$existing = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue

if ($existing) {
    Set-NetFirewallRule -DisplayName $ruleName -Enabled True -Direction Inbound -Action Allow -Profile Any
    Write-Host "Updated firewall rule: $ruleName" -ForegroundColor Green
} else {
    New-NetFirewallRule `
        -DisplayName $ruleName `
        -Direction Inbound `
        -Action Allow `
        -Protocol TCP `
        -LocalPort 5000 `
        -RemoteAddress "100.64.0.0/10" `
        -InterfaceAlias "Tailscale" `
        -Profile Any | Out-Null
    Write-Host "Created firewall rule: $ruleName" -ForegroundColor Green
}

Get-NetFirewallRule -DisplayName $ruleName |
    Select-Object DisplayName, Enabled, Direction, Action, Profile
Get-NetFirewallRule -DisplayName $ruleName |
    Get-NetFirewallPortFilter |
    Select-Object Protocol, LocalPort
Get-NetFirewallRule -DisplayName $ruleName |
    Get-NetFirewallAddressFilter |
    Select-Object RemoteAddress, InterfaceAlias
