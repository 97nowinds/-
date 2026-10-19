param(
    [string]$BackendUrl = 'http://127.0.0.1:15001',
    [int]$Port = 5173,
    [string]$PythonPath
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $PythonPath) {
    $PythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path $PythonPath)) { throw 'Pass -PythonPath with the Python environment used by the frontend.' }
}
if (-not (Test-Path -LiteralPath $PythonPath)) { throw 'Python executable not found.' }
$runtime = Join-Path $projectRoot 'runtime'
New-Item -ItemType Directory -Path $runtime -Force | Out-Null
while ($true) {
    try {
        # Leave an existing service alone; take over only when its port is free.
        $listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
        if (-not $listener) {
            $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
            $startArgs = @("`"$(Join-Path $projectRoot 'frontend_server.py')`"", '--host', '0.0.0.0', '--port', "$Port", '--api', "`"$BackendUrl`"")
            $process = Start-Process -FilePath $PythonPath -ArgumentList $startArgs -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $runtime "frontend-$stamp.stdout.log") -RedirectStandardError (Join-Path $runtime "frontend-$stamp.stderr.log")
            $process.WaitForExit()
        }
    } catch { Add-Content -LiteralPath (Join-Path $runtime 'frontend-watcher.log') -Value "$(Get-Date -Format s) $($_.Exception.Message)" }
    Start-Sleep -Seconds 10
}
