$ErrorActionPreference = "Stop"

$powershell = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
$tunnelScript = Join-Path $PSScriptRoot "start_cluster_tunnel.ps1"
$pushScript = Join-Path $PSScriptRoot "start_rtsp_push.ps1"
$identityFile = "$env:USERPROFILE\.ssh\lab_rtsp_ed25519"
$taskUser = "$env:USERDOMAIN\$env:USERNAME"

foreach ($path in @($tunnelScript, $pushScript, $identityFile)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required script was not found: $path"
    }
}

$principal = New-ScheduledTaskPrincipal -UserId $taskUser -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1)
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $taskUser

$tunnelAction = New-ScheduledTaskAction `
    -Execute $powershell `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$tunnelScript`""
$pushAction = New-ScheduledTaskAction `
    -Execute $powershell `
    -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Normal -File `"$pushScript`" -RemoteBaseUrl rtsp://127.0.0.1:18554"

Register-ScheduledTask `
    -TaskName "Lab Safety RTSP Tunnel" `
    -Action $tunnelAction `
    -Trigger $trigger `
    -Principal $principal `
    -Settings $settings `
    -Description "Reconnect the SSH tunnel from the lab host to the lnx RTSP receiver." `
    -Force | Out-Null

Register-ScheduledTask `
    -TaskName "Lab Safety RTSP Push" `
    -Action $pushAction `
    -Trigger $trigger `
    -Principal $principal `
    -Settings $settings `
    -Description "Prompt for camera passwords and start the three FFmpeg RTSP pushes." `
    -Force | Out-Null

Write-Host "Registered: Lab Safety RTSP Tunnel" -ForegroundColor Green
Write-Host "Registered: Lab Safety RTSP Push" -ForegroundColor Green
Write-Host "The push task prompts for all three camera passwords after each Windows login."
