[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$backend = Join-Path $root 'backend'
$frontend = Join-Path $root 'frontend'
$python = (Resolve-Path "$backend\.venv\Scripts\python.exe").Path
$node = (Get-Command node.exe).Source
$vite = (Resolve-Path "$frontend\node_modules\vite\bin\vite.js").Path
foreach ($port in 8000,5175) {
    $listener = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue
    if ($listener) { throw "Port $port already has a listener (PID $($listener.OwningProcess))." }
}
# Both children share this console so Ctrl+C reaches Uvicorn and Vite.
$children = @()
try {
    $children += Start-Process $python -ArgumentList '-m uvicorn app.main:app --host 127.0.0.1 --port 8000' -WorkingDirectory $backend -NoNewWindow -PassThru
    $children += Start-Process $node -ArgumentList ('"{0}" --host localhost --port 5175 --strictPort' -f $vite) -WorkingDirectory $frontend -NoNewWindow -PassThru
    $ready = $false
    foreach ($attempt in 1..60) {
        if ($children.HasExited -contains $true) { throw 'A service exited during startup.' }
        try {
            $status = Invoke-RestMethod http://localhost:5175/api/v1/model/status -TimeoutSec 2
            if ($status.model_ready -and $status.video_ready) { $ready = $true; break }
        } catch {}
        Start-Sleep -Milliseconds 500
    }
    if (!$ready) { throw 'Timed out waiting for classifier and detector readiness.' }
    Write-Host 'MediaLens AI  http://localhost:5175   API  http://127.0.0.1:8000/docs'
    Write-Host 'Press Ctrl+C in this console to stop both services.'
    while ($children.HasExited -notcontains $true) { Start-Sleep -Seconds 1 }
} finally {
    foreach ($child in $children) {
        if (!$child.HasExited -and !$child.WaitForExit(4000)) {
            & taskkill.exe /PID $child.Id /T /F | Out-Null
        }
    }
}
