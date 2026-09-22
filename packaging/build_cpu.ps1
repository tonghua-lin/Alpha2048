$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $taskRoot
$taskLogDir = Join-Path $taskRoot 'reports\packaging'
New-Item -ItemType Directory -Force -Path $taskLogDir | Out-Null
$taskLog = Join-Path $taskLogDir (('cpu-build-{0}.log' -f (Get-Date -Format 'yyyyMMdd-HHmmss')))
Start-Transcript -Path $taskLog
$taskExit = 1
try {
    $env:PYTHONUTF8 = '1'
    & (Join-Path $taskRoot '.venv\Scripts\python.exe') -u (Join-Path $PSScriptRoot 'build_cpu.py')
    if ($LASTEXITCODE -ne 0) { throw "CPU build failed with exit code $LASTEXITCODE" }
    $taskExit = 0
} catch {
    Write-Host $_ -ForegroundColor Red
} finally {
    Write-Host "Log: $taskLog"
    Stop-Transcript
}
exit $taskExit
