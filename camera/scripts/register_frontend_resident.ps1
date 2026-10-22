param([string]$PythonPath)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $PythonPath) { $PythonPath = Join-Path $projectRoot '.venv\Scripts\pythonw.exe' }
if (-not (Test-Path -LiteralPath $PythonPath)) { throw 'Frontend Python environment was not found.' }
$taskName = 'Lab Safety Frontend Watcher'
$taskUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$script = Join-Path $PSScriptRoot 'resident_services.py'
$config = Join-Path $projectRoot 'config\resident_services.json'
$action = New-ScheduledTaskAction -Execute $PythonPath -Argument "`"$script`" --config `"$config`"" -WorkingDirectory $projectRoot
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -Hidden
$principal = New-ScheduledTaskPrincipal -UserId $taskUser -LogonType Interactive -RunLevel Limited
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $taskUser
$retryTrigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 1)
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing) { Stop-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue }
Register-ScheduledTask -TaskName $taskName -Action $action -Settings $settings -Principal $principal -Trigger @($trigger, $retryTrigger) -Description 'Own SSH tunnels, camera publishers, frontend and notifications; retry every minute after login.' -Force | Out-Null
Start-ScheduledTask -TaskName $taskName
Write-Host 'Resident service task registered and started.'
