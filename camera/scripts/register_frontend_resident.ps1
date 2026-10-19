param(
    [string]$PythonPath,
    [string]$BackendUrl = 'http://127.0.0.1:15001'
)
$ErrorActionPreference = 'Stop'
if (-not $PythonPath) { $PythonPath = Join-Path (Split-Path -Parent $PSScriptRoot) '.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $PythonPath)) { throw 'Pass -PythonPath with the installed frontend Python executable.' }
$taskName = 'Lab Safety Frontend Watcher'
$taskUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$watcher = Join-Path $PSScriptRoot 'watch_frontend.ps1'
$action = New-ScheduledTaskAction -Execute "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$watcher`" -PythonPath `"$PythonPath`" -BackendUrl `"$BackendUrl`""
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1)
$principal = New-ScheduledTaskPrincipal -UserId $taskUser -LogonType Interactive -RunLevel Limited
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $taskUser
Register-ScheduledTask -TaskName $taskName -Action $action -Settings $settings -Principal $principal -Trigger $trigger -Description 'Keep the existing lab frontend running after user login.' -Force | Out-Null
Start-ScheduledTask -TaskName $taskName
Write-Host 'Frontend watcher registered and started. Existing services are left running.'
