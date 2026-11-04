$ErrorActionPreference = 'Stop'
$taskName = 'Lab Safety Display Watcher'
if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
    Stop-ScheduledTask -TaskName $taskName
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
}
Write-Host 'Display watcher removed. You may close its browser window.'
