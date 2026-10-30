$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\pythonw.exe'
$script = Join-Path $PSScriptRoot 'face_adaptation_bridge.py'
$taskUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$name = 'Lab Face Adaptation Bridge'
$action = New-ScheduledTaskAction -Execute $python -Argument "`"$script`"" -WorkingDirectory $root
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -Hidden
$principal = New-ScheduledTaskPrincipal -UserId $taskUser -LogonType Interactive -RunLevel Limited
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $taskUser
$retry = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName $name -Action $action -Settings $settings -Principal $principal -Trigger @($trigger,$retry) -Description 'Authenticated loopback site face sample service' -Force | Out-Null
Start-ScheduledTask -TaskName $name
Write-Host 'Site sample bridge started.'
