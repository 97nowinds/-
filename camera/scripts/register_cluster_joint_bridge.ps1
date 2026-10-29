param([string]$PythonPath)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $PythonPath) { $PythonPath = Join-Path $projectRoot '.venv\Scripts\pythonw.exe' }
if (-not (Test-Path -LiteralPath $PythonPath)) { throw 'Python environment was not found.' }
$taskName = 'Lab Interaction Cluster Bridge'
$taskUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$script = Join-Path $PSScriptRoot 'cluster_joint_bridge.py'
$action = New-ScheduledTaskAction -Execute $PythonPath -Argument "`"$script`"" -WorkingDirectory $projectRoot
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -Hidden
$principal = New-ScheduledTaskPrincipal -UserId $taskUser -LogonType Interactive -RunLevel Limited
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $taskUser
$retry = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 1)
if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
    Stop-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
}
Register-ScheduledTask -TaskName $taskName -Action $action -Settings $settings -Principal $principal -Trigger @($trigger, $retry) -Description 'Encrypted loopback forwarding for GPU tracking and instrument interaction; recover owned SSH transport.' -Force | Out-Null
Start-ScheduledTask -TaskName $taskName
Write-Host 'Cluster joint bridge registered and started.'
