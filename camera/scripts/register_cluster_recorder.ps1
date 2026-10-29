param([Parameter(Mandatory=$true)][string]$ReceiverRoot)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$ReceiverRoot = (Resolve-Path -LiteralPath $ReceiverRoot).Path
if (-not (Test-Path -LiteralPath (Join-Path $ReceiverRoot 'runtime\cluster_receiver.enabled'))) {
    throw 'Receiver must be explicitly configured for cluster transport first.'
}
$pythonPath = Join-Path $ReceiverRoot '.venv\Scripts\pythonw.exe'
$script = Join-Path $ReceiverRoot 'resident_joint.py'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Recorder Python environment was not found.' }
$taskName = 'Lab Joint Frame Recorder'
$taskUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$credentials = Join-Path $projectRoot 'runtime\handoff-credentials.json'
$action = New-ScheduledTaskAction -Execute $pythonPath -Argument "`"$script`" --teammate-root `"$projectRoot`" --credentials-file `"$credentials`"" -WorkingDirectory $ReceiverRoot
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero) -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -Hidden
$principal = New-ScheduledTaskPrincipal -UserId $taskUser -LogonType Interactive -RunLevel Limited
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $taskUser
$retry = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 1)
if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) { Stop-ScheduledTask -TaskName $taskName }
Register-ScheduledTask -TaskName $taskName -Action $action -Settings $settings -Principal $principal -Trigger @($trigger, $retry) -Description 'Capture identity-bound frozen frames and send to the GPU receiver through the cluster bridge.' -Force | Out-Null
Start-ScheduledTask -TaskName $taskName
Write-Host 'Cluster frame recorder registered and started.'
