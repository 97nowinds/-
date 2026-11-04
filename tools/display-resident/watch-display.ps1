param(
    [string]$Url = 'http://100.106.153.101:18080/display?api=http://100.106.153.101:18081',
    [int]$CheckSeconds = 15
)
$ErrorActionPreference = 'Stop'
$profilePath = Join-Path $env:LOCALAPPDATA 'LabSafetyDisplay\EdgeProfile'
$logPath = Join-Path $env:LOCALAPPDATA 'LabSafetyDisplay\watch.log'
New-Item -ItemType Directory -Path (Split-Path $profilePath) -Force | Out-Null
$browser = @(
    "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe",
    "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe",
    "$env:LOCALAPPDATA\Microsoft\Edge\Application\msedge.exe"
) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $browser) { throw 'Microsoft Edge was not found.' }
function Write-DisplayLog([string]$Message) {
    if ((Test-Path $logPath) -and (Get-Item $logPath).Length -gt 2MB) {
        Move-Item -LiteralPath $logPath -Destination "$logPath.old" -Force
    }
    Add-Content -LiteralPath $logPath -Value "$(Get-Date -Format s) $Message"
}
function Get-DisplayProcess {
    Get-CimInstance Win32_Process -Filter "Name='msedge.exe'" |
        Where-Object { $_.CommandLine -and $_.CommandLine.Contains($profilePath) -and $_.CommandLine -notmatch '--type=' }
}
function Test-DisplayPage {
    $response = $null
    try {
        $request = [System.Net.HttpWebRequest]::Create($Url)
        $request.Proxy = $null
        $request.Timeout = 6000
        $response = $request.GetResponse()
        return ([int]$response.StatusCode -eq 200)
    } catch { return $false }
    finally { if ($response) { $response.Close() } }
}
$hadOutage = $false
$previousState = ''
Write-DisplayLog 'Display watcher started.'
while ($true) {
    try {
        $online = Test-DisplayPage
        $currentState = if ($online) { 'online' } else { 'offline' }
        if ($currentState -ne $previousState) { Write-DisplayLog "Frontend $currentState"; $previousState = $currentState }
        if (-not $online) {
            $hadOutage = $true
        } else {
            $processes = @(Get-DisplayProcess)
            if ($hadOutage -and $processes.Count -gt 0) {
                # Only the dedicated display profile is restarted.
                foreach ($item in $processes) { Stop-Process -Id $item.ProcessId -ErrorAction SilentlyContinue }
                Start-Sleep -Seconds 2
                $processes = @(Get-DisplayProcess)
            }
            if ($processes.Count -eq 0) {
                Start-Process -FilePath $browser -ArgumentList @(
                    "--user-data-dir=`"$profilePath`"", '--no-first-run', '--new-window', '--start-fullscreen', "`"$Url`""
                ) -WindowStyle Normal | Out-Null
                Write-DisplayLog 'Opened display browser.'
            }
            $hadOutage = $false
        }
    } catch { Write-DisplayLog "Watcher error: $($_.Exception.Message)" }
    Start-Sleep -Seconds ([Math]::Max(5, $CheckSeconds))
}
