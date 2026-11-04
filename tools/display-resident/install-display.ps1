$ErrorActionPreference = 'Stop'
$destination = Join-Path $env:LOCALAPPDATA 'LabSafetyDisplay'
New-Item -ItemType Directory -Path $destination -Force | Out-Null
$watcher = Join-Path $destination 'watch-display.ps1'
Copy-Item -LiteralPath (Join-Path $PSScriptRoot 'watch-display.ps1') -Destination $watcher -Force
$taskName = 'Lab Safety Display Watcher'
$taskUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$powershellPath = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$action = New-ScheduledTaskAction -Execute $powershellPath -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$watcher`""
$principal = New-ScheduledTaskPrincipal -UserId $taskUser -LogonType Interactive -RunLevel Limited
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $taskUser
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1)
if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
    Stop-ScheduledTask -TaskName $taskName
}
Register-ScheduledTask -TaskName $taskName -Action $action -Principal $principal -Trigger $trigger -Settings $settings -Description 'Open the laboratory display after login, restart its browser on exit and reopen after frontend recovery.' -Force | Out-Null
Start-ScheduledTask -TaskName $taskName
Write-Host 'Installed and started. Sign in after reboot to open the display.' -ForegroundColor Green
Write-Host "Logs: $destination\watch.log"
