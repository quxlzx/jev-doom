# restart.ps1 — kill llama-server, relaunch tasks, verify LoRA loaded
Get-Process llama-server -ErrorAction SilentlyContinue | ForEach-Object {
  Write-Output ("killing PID " + $_.Id)
  Stop-Process -Id $_.Id -Force
}
Start-Sleep -Seconds 3
schtasks /run /tn doomspine | Out-Null
schtasks /run /tn doombrain | Out-Null
$deadline = (Get-Date).AddSeconds(90)
$ready = $false
while ((Get-Date) -lt $deadline) {
  Start-Sleep -Seconds 5
  try {
    $h = Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8181/health -TimeoutSec 3
    if ($h.Content -match "ok") { $ready = $true; break }
  } catch {}
}
Write-Output ("spine ready: $ready")
$body = Get-Content D:\doomsrv\test_swap.json -Raw
$r = Invoke-WebRequest -UseBasicParsing -Uri http://127.0.0.1:8181/completion -Method Post -Body $body -ContentType "application/json" -TimeoutSec 60
$j = $r.Content | ConvertFrom-Json
Write-Output ("pick=" + $j.content + " lora=" + ($j.generation_settings.lora -join ","))
