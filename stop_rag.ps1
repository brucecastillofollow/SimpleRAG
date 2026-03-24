Write-Host "[info] Stopping rag_local.py processes..."
$rag = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'rag_local.py' }
if ($rag) {
    $rag | ForEach-Object {
        taskkill /PID $_.ProcessId /F | Out-Host
    }
} else {
    Write-Host "[info] No rag_local.py process found."
}

Write-Host "[info] Stopping Ollama processes..."
taskkill /IM ollama.exe /F 2>$null | Out-Host
taskkill /IM "ollama app.exe" /F 2>$null | Out-Host

Write-Host "[ok] Stop command completed."
